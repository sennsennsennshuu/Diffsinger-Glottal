import json
import shutil
from pathlib import Path
from typing import List, Union, Tuple, Dict

import onnx
import onnxsim
import torch
import yaml

from basics.base_exporter import BaseExporter
from deployment.modules.toplevel import DiffSingerAcousticONNX
from modules.fastspeech.param_adaptor import VARIANCE_CHECKLIST
from utils import load_ckpt, onnx_helper, remove_suffix
from utils.hparams import hparams
from lib.vocabulary import load_phoneme_dictionary


def _get_hparam(key, default=None):
    """Read hparam with v3 nested-config fallback.

    v3 stores: model.embeddings.use_breathiness_embed
    hparams flattens to: embeddings_use_breathiness_embed
    This function checks prefixed versions FIRST, then bare key.
    """
    # Check prefixed versions first (v3 flattening)
    for prefix in ('embeddings_', 'linguistic_encoder_', 'spec_decoder_',
                   'local_upsample_', 'normalization_', 'vocoder_',
                   'model_embeddings_', 'model_linguistic_encoder_'):
        v = hparams.get(prefix + key, None)
        if v is not None:
            return v
    # Try nested model.*
    model = hparams.get('model', {})
    for section in ('embeddings', 'linguistic_encoder', 'spec_decoder',
                    'local_upsample', 'normalization', 'vocoder'):
        sec = model.get(section, {})
        if isinstance(sec, dict) and key in sec:
            return sec[key]
    # Vocoder at top level (dict)
    voc = hparams.get('vocoder', {})
    if isinstance(voc, dict) and key in voc:
        return voc[key]
    # ── Raw config fallback: some experiments don't flatten properly ──
    v = hparams.get(key, default)
    if v is not default or default is None:
        return v
    # Last resort: parse config.yaml directly from work_dir
    try:
        import yaml as _yaml
        cfg_path = Path(hparams.get('work_dir', '')) / 'config.yaml'
        if cfg_path.exists():
            raw = _yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
            if isinstance(raw, dict):
                for section in ('model', 'inference', 'training'):
                    sec = raw.get(section, {})
                    if isinstance(sec, dict):
                        for sub in ('embeddings', 'linguistic_encoder', 'spec_decoder',
                                    'normalization', 'vocoder'):
                            sub_sec = sec.get(sub, {})
                            if isinstance(sub_sec, dict) and key in sub_sec:
                                return sub_sec[key]
                        if key in sec:
                            return sec[key]
    except Exception:
        pass
    return default



class DiffSingerAcousticExporter(BaseExporter):
    def __init__(
            self,
            device: Union[str, torch.device] = 'cpu',
            cache_dir: Path = None,
            ckpt_steps: int = None,
            freeze_gender: float = None,
            freeze_velocity: bool = False,
            export_spk: List[Tuple[str, Dict[str, float]]] = None,
            freeze_spk: Tuple[str, Dict[str, float]] = None
    ):
        super().__init__(device=device, cache_dir=cache_dir)
        # Basic attributes
        self.model_name: str = hparams['exp_name']
        self.ckpt_steps: int = ckpt_steps
        # ── v3 hparams access helper ──
        # v3 config nests values under model.* → hparams flattens with prefix.
        # e.g. model.embeddings.use_breathiness_embed → hparams['embeddings_use_breathiness_embed']
        # We try: bare key → nested section → flattened prefix.
        def _cfg(self, key, default=None):
            v = hparams.get(key, None)
            if v is not None:
                return v
            for prefix in ('embeddings_', 'linguistic_encoder_', 'spec_decoder_',
                           'local_upsample_', 'normalization_', 'vocoder_',
                           'vocoder.', 'variance_parameters.'):
                v = hparams.get(prefix + key, None)
                if v is not None:
                    return v
            # Try nested model section
            model = hparams.get('model', {})
            for section in ('embeddings', 'linguistic_encoder', 'spec_decoder',
                            'local_upsample', 'normalization'):
                sec = model.get(section, {})
                if isinstance(sec, dict) and key in sec:
                    return sec[key]
            # Try vocoder nested
            voc = hparams.get('vocoder', {})
            if isinstance(voc, dict) and key in voc:
                return voc[key]
            return default

        self.spk_map: dict = self.build_spk_map()
        self.lang_map: dict = self.build_lang_map()
        from pathlib import Path as _Path
        _root = _Path(__file__).resolve().parents[2]
        config_dicts = hparams.get('dictionaries', {})
        if not isinstance(config_dicts, dict) or not config_dicts:
            raise RuntimeError(
                'hparams.dictionaries is empty or missing — export requires a valid phoneme dictionary.')
        # Normalise language codes: some v3 configs use file basenames (e.g. ARPAbetPlus)
        # → derive a ISO-style code from the filename as a fallback.
        _LANG_FROM_DICT = {'opencpop': 'zh', 'dictionary-ja': 'ja', 'dictionary-en': 'en',
                           'en_dict': 'en', 'arpabet': 'en', 'arpabetplus': 'en'}
        dicts = {}
        for raw_lang, cfg_path in config_dicts.items():
            lang = raw_lang.lower()
            for pattern, code in _LANG_FROM_DICT.items():
                if pattern in lang or pattern in str(cfg_path).lower():
                    lang = code
                    break
            else:
                # If no match, try to extract from filename: dictionary-{code}.txt
                import re as _re
                m = _re.search(r'dictionary[-_]([a-z]{2,3})', str(cfg_path))
                if m:
                    lang = m.group(1)
            # Resolve file path
            work_dict = _Path(hparams['work_dir']) / f'dictionary-{lang}.txt'
            if work_dict.exists():
                dicts[lang] = work_dict
            else:
                src = _Path(cfg_path)
                if not src.is_absolute():
                    src = _root / src
                if src.exists():
                    dicts[lang] = src
                else:
                    print(f'| WARNING: dictionary for lang={lang} not found at {src}, skipping')
        # ── Auto-discover: if only got 1 dictionary, scan project dictionaries/ ──
        if len(dicts) <= 1:
            dict_dir = _root / 'dictionaries'
            if dict_dir.exists():
                for tf in sorted(dict_dir.glob('*.txt')):
                    fname = tf.stem.lower()
                    # Already have this one
                    if any(str(tf.resolve()) == str(d.resolve()) for d in dicts.values()):
                        continue
                    # Derive lang from filename
                    for pattern, code in _LANG_FROM_DICT.items():
                        if pattern in fname:
                            if code not in dicts:
                                dicts[code] = tf
                                print(f'| auto-discovered dictionary: {code} => {tf}')
                            break
                    else:
                        m = _re.search(r'dictionary[-_]([a-z]{2,3})', fname) if '_re' in dir() else None
                        if m:
                            code = m.group(1)
                            if code not in dicts:
                                dicts[code] = tf
                                print(f'| auto-discovered dictionary: {code} => {tf}')
        if not dicts:
            raise RuntimeError(
                f'No dictionary files resolved from hparams dictionaries={config_dicts}. '
                f'Place dictionary-zh.txt / dictionary-ja.txt in work_dir or dictionaries/.')
        dict_paths = dicts
        self.phoneme_dictionary = load_phoneme_dictionary(dict_paths)
        # use_lang_id: v3 uses model.use_lang_id (flattened), fallback to top-level
        self.use_lang_id = _get_hparam('use_lang_id', False)
        self.model = self.build_model()
        self.use_glottal = self.model.glottal_predictor is not None
        self.fs2_aux_cache_path = self.cache_dir / (
            'fs2_aux.onnx' if self.model.use_shallow_diffusion else 'fs2.onnx'
        )
        self.diffusion_cache_path = self.cache_dir / 'diffusion.onnx'

        # Attributes for logging
        self.model_class_name = remove_suffix(self.model.__class__.__name__, 'ONNX')
        fs2_aux_cls_logging = [remove_suffix(self.model.fs2.__class__.__name__, 'ONNX')]
        if self.model.use_shallow_diffusion:
            aux_cls = self.model.aux_decoder
            # main-sfcb wraps aux_decoder in a wrapper with .decoder; v3 directly uses ConvNeXtDecoder
            if hasattr(aux_cls, 'decoder'):
                aux_cls = aux_cls.decoder
            fs2_aux_cls_logging.append(remove_suffix(aux_cls.__class__.__name__, 'ONNX'))
        self.fs2_aux_class_name = ', '.join(fs2_aux_cls_logging)
        if self.model.use_shallow_diffusion:
            aux_cls = self.model.aux_decoder
            if hasattr(aux_cls, 'decoder'):
                aux_cls = aux_cls.decoder
            self.aux_decoder_class_name = remove_suffix(aux_cls.__class__.__name__, 'ONNX')
        else:
            self.aux_decoder_class_name = None
        self.backbone_class_name = remove_suffix(self.model.diffusion.backbone.__class__.__name__, 'ONNX')
        self.diffusion_class_name = remove_suffix(self.model.diffusion.__class__.__name__, 'ONNX')

        # Attributes for exporting
        self.expose_gender = freeze_gender is None
        self.expose_velocity = not freeze_velocity
        self.freeze_spk: Tuple[str, Dict[str, float]] = freeze_spk \
            if hparams['use_spk_id'] else None
        self.export_spk: List[Tuple[str, Dict[str, float]]] = export_spk \
            if hparams['use_spk_id'] and export_spk is not None else []
        if hparams['use_key_shift_embed'] and not self.expose_gender:
            shift_min, shift_max = hparams['augmentation_args']['random_pitch_shifting']['range']
            key_shift = freeze_gender * shift_max if freeze_gender >= 0. else freeze_gender * abs(shift_min)
            key_shift = max(min(key_shift, shift_max), shift_min)  # clip key shift
            self.model.fs2.register_buffer('frozen_key_shift', torch.FloatTensor([key_shift]).to(self.device), persistent=False)
        if hparams['use_spk_id']:
            if not self.export_spk and self.freeze_spk is None:
                # In case the user did not specify any speaker settings:
                if len(self.spk_map) == 1:
                    # If there is only one speaker, freeze him/her.
                    first_spk = next(iter(self.spk_map.keys()))
                    self.freeze_spk = (first_spk, {first_spk: 1.0})
                else:
                    # If there are multiple speakers, export them all.
                    self.export_spk = [(name, {name: 1.0}) for name in self.spk_map.keys()]
            if self.freeze_spk is not None:
                self.model.fs2.register_buffer('frozen_spk_embed', self._perform_spk_mix(self.freeze_spk[1]))

    def build_model(self) -> DiffSingerAcousticONNX:
        # Detect checkpoint metadata before building model
        ckpt_path = hparams['work_dir']
        ckpt_files = sorted(Path(ckpt_path).glob('model*.ckpt'))
        use_glottal = False
        vocab_size_from_ckpt = None
        if ckpt_files:
            ckpt = torch.load(ckpt_files[-1], map_location='cpu', weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt)
            use_glottal = any('glottal' in k for k in state_dict.keys())
            print(f'| Glottal weights detected: {use_glottal}')
            # Read vocab_size from checkpoint embedding layer
            for embed_key in (
                'model.linguistic_encoder.token_embedding.weight',
                'model.fs2.txt_embed.weight',
                'linguistic_encoder.token_embedding.weight',
                'fs2.txt_embed.weight',
            ):
                if embed_key in state_dict:
                    vocab_size_from_ckpt = state_dict[embed_key].shape[0]
                    print(f'| vocab_size from checkpoint ({embed_key}): {vocab_size_from_ckpt}')
                    break
        # ── Load ph_map.json to get correct token IDs (eliminates rebuild mismatch) ──
        ph_map_path = Path(ckpt_path) / 'ph_map.json'
        ph_map = {}
        cross_lingual_tokens = set()
        if ph_map_path.exists():
            with open(ph_map_path, encoding='utf-8') as f:
                ph_map = json.load(f)
            # Find cross-lingual tokens: same ID shared by zh/* and ja/*
            id_count = {}
            id_phones = {}
            for ph, idx in ph_map.items():
                id_count[idx] = id_count.get(idx, 0) + 1
                id_phones.setdefault(idx, []).append(ph)
            for idx, count in id_count.items():
                if count >= 2:
                    phones = id_phones[idx]
                    langs = {p.split('/')[0] for p in phones if '/' in p}
                    if len(langs) >= 2:
                        cross_lingual_tokens.add(idx)
            if not vocab_size_from_ckpt:
                vocab_size_from_ckpt = len(ph_map) + 1  # +1 for PAD=0
                print(f'| vocab_size from ph_map.json: {vocab_size_from_ckpt}')
        if vocab_size_from_ckpt is None:
            vocab_size_from_ckpt = len(self.phoneme_dictionary)
            print(f'| vocab_size from dictionary: {vocab_size_from_ckpt}')
            cross_lingual_tokens = sorted({
                self.phoneme_dictionary.encode_one(p)
                for p in self.phoneme_dictionary.cross_lingual_phonemes
            })
        else:
            cross_lingual_tokens = sorted(cross_lingual_tokens)
        if vocab_size_from_ckpt != len(self.phoneme_dictionary):
            print(f'| vocab_size: checkpoint={vocab_size_from_ckpt}, '
                  f'dictionary={len(self.phoneme_dictionary)}, using checkpoint value')

        model = DiffSingerAcousticONNX(
            vocab_size=vocab_size_from_ckpt,
            out_dims=hparams['audio_num_mel_bins'],
            cross_lingual_token_idx=cross_lingual_tokens,
            use_glottal=use_glottal
        ).eval().to(self.device)
        load_ckpt(model, hparams['work_dir'], ckpt_steps=self.ckpt_steps,
                  prefix_in_ckpt='model', strict=False, device=self.device)
        return model

    def export(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        model_name = self.model_name
        if self.freeze_spk is not None:
            model_name += '.' + self.freeze_spk[0]
        self.export_model(path / f'{model_name}.onnx')
        self.export_attachments(path)

    def export_model(self, path: Path):
        self._torch_export_model()
        fs2_aux_onnx = self._optimize_fs2_aux_graph(onnx.load(self.fs2_aux_cache_path))
        diffusion_onnx = self._optimize_diffusion_graph(onnx.load(self.diffusion_cache_path))

        if self.use_glottal:
            glottal_onnx = self._export_model_glottal()
            glottal_onnx = self._optimize_glottal_graph(glottal_onnx)

            # ── Three-graph manual merge ──
            # ONNX compose.merge_models is buggy for nested merges.
            # We merge by simple append: fs2 nodes → glottal nodes → diffusion nodes.
            # Only wiring needed: glottal consumes fs2's 'condition' output,
            # diffusion consumes glottal's 'condition_out' (and fs2's 'aux_mel' if shallow).
            print(f'Merging {self.fs2_aux_class_name} + GlottalConditioner + '
                  f'{self.diffusion_class_name} back into {self.model_class_name}...')

            from onnx import helper

            # 1. Collect all nodes in dependency order: fs2 → glottal → diffusion
            all_nodes = list(fs2_aux_onnx.graph.node)
            all_inits = list(fs2_aux_onnx.graph.initializer)
            all_vinfos = list(fs2_aux_onnx.graph.value_info)
            seen_inputs = {i.name for i in fs2_aux_onnx.graph.input}
            seen_outputs = set()
            all_inputs = list(fs2_aux_onnx.graph.input)
            all_outputs = []

            # Add fs2 outputs
            for o in fs2_aux_onnx.graph.output:
                if o.name not in seen_outputs:
                    seen_outputs.add(o.name)
                    all_outputs.append(o)

            # 2. Append glottal nodes (consume 'condition' from fs2 output)
            for gn in glottal_onnx.graph.node:
                all_nodes.append(gn)
            all_inits.extend(glottal_onnx.graph.initializer)
            all_vinfos.extend(glottal_onnx.graph.value_info)
            # Glottal output 'condition_out' (new)
            for o in glottal_onnx.graph.output:
                if o.name not in seen_outputs:
                    seen_outputs.add(o.name)
                    all_outputs.append(o)
            # Glottal inputs: skip 'condition' (wired from fs2 output),
            #   'f0' (shared with fs2 input), add 'glottal_blend' (new).
            #   OpenUtau 0.1.568+glottal reads this from user expression curve.
            for i in glottal_onnx.graph.input:
                if i.name in ('condition', 'f0'):
                    continue
                if i.name not in seen_inputs:
                    seen_inputs.add(i.name)
                    all_inputs.append(i)

            # 3. Append diffusion nodes, rename condition→condition_out,
            #    x_aux→aux_mel, depth→depth, steps→steps (most are no-op)
            _wire = {
                'condition': 'condition_out',
            }
            if self.model.use_shallow_diffusion:
                _wire['x_aux'] = 'aux_mel'
            for dn in diffusion_onnx.graph.node:
                dn.input[:] = [_wire.get(inp, inp) for inp in dn.input]
                all_nodes.append(dn)
            all_inits.extend(diffusion_onnx.graph.initializer)
            all_vinfos.extend(diffusion_onnx.graph.value_info)
            for o in diffusion_onnx.graph.output:
                if o.name not in seen_outputs:
                    seen_outputs.add(o.name)
                    all_outputs.append(o)
            _skip_diff_inputs = {'condition', 'x_aux'}
            for i in diffusion_onnx.graph.input:
                if i.name in _skip_diff_inputs:
                    continue
                if i.name not in seen_inputs:
                    seen_inputs.add(i.name)
                    all_inputs.append(i)

            # 4. Build final graph and model
            final_graph = helper.make_graph(
                nodes=all_nodes,
                name=fs2_aux_onnx.graph.name,
                inputs=all_inputs,
                outputs=[o for o in all_outputs
                          if o.name not in ('condition', 'condition_out', 'aux_mel')],
                initializer=all_inits,
                value_info=all_vinfos,
            )
            merged = helper.make_model(final_graph,
                                       producer_name=fs2_aux_onnx.producer_name,
                                       producer_version=fs2_aux_onnx.producer_version,
                                       domain=fs2_aux_onnx.domain,
                                       model_version=fs2_aux_onnx.model_version)
            merged.ir_version = fs2_aux_onnx.ir_version
            merged.opset_import.extend(fs2_aux_onnx.opset_import)

            # 5. Add dim prefixes
            onnx_helper.model_add_prefixes(
                merged,
                dim_prefix=('fs2aux.' if self.model.use_shallow_diffusion else 'fs2.'),
                ignored_pattern=r'(n_tokens)|(n_frames)'
            )
            # model_add_prefixes also processes subgraphs; diffusion internal
            # dim names don't have 'diffusion.' prefix in this simple merge,
            # which is fine — the dim prefix handles n_frames globally.

            # 6. ONNX simplify
            try:
                merged, check = onnxsim.simplify(merged, include_subgraph=True)
            except Exception as e:
                print(f'| onnxsim failed ({type(e).__name__}), continuing')

            model_onnx = merged
        else:
            model_onnx = self._merge_fs2_aux_diffusion_graphs(fs2_aux_onnx, diffusion_onnx)

        onnx.save(model_onnx, path)
        self.fs2_aux_cache_path.unlink()
        self.diffusion_cache_path.unlink()
        print(f'| export model => {path}')

    def export_attachments(self, path: Path):
        for spk in self.export_spk:
            self._export_spk_embed(
                path / f'{self.model_name}.{spk[0]}.emb',
                self._perform_spk_mix(spk[1])
            )
        self.export_dictionaries(path)
        self._export_phonemes(path)

        model_name = self.model_name
        if self.freeze_spk is not None:
            model_name += '.' + self.freeze_spk[0]
        # vocoder identifier: v3 uses pc-nsf-hifigan 2025.02
        vocoder_type = _get_hparam('vocoder_type', '')
        vocoder_path = _get_hparam('vocoder_path', '')
        if 'pc_nsf_hifigan' in str(vocoder_type) or 'pc_nsf_hifigan' in str(vocoder_path):
            vocoder_name = 'pc_nsf_hifigan_44.1k_hop512_128bin_2025.02'
        elif vocoder_type:
            vocoder_name = vocoder_type
        else:
            vocoder_name = 'nsf_hifigan'
        dsconfig = {
            # basic configs
            'phonemes': f'{self.model_name}.phonemes.json',
            'languages': f'{self.model_name}.languages.json',
            'use_lang_id': False,  # Always False: OpenUTAU 0.1.565.0 doesn't support languages
            'acoustic': f'{model_name}.onnx',
            'hidden_size': hparams['hidden_size'],
            'vocoder': vocoder_name,
        }
        # multi-speaker
        if len(self.export_spk) > 0:
            dsconfig['speakers'] = [f'{self.model_name}.{spk[0]}' for spk in self.export_spk]
        # parameters
        if self.expose_gender:
            dsconfig['augmentation_args'] = {
                'random_pitch_shifting': {
                    'range': hparams['augmentation_args']['random_pitch_shifting']['range']
                }
            }
        dsconfig['use_key_shift_embed'] = self.expose_gender
        dsconfig['use_speed_embed'] = self.expose_velocity
        # Read variance flags from hparams (v3: model.embeddings.* → embeddings_*)
        for variance in VARIANCE_CHECKLIST:
            if variance == 'pitch':
                continue
            key = f'use_{variance}_embed'
            dsconfig[key] = _get_hparam(key, (variance in self.model.fs2.variance_embed_list))
        dsconfig['use_pitch_embed'] = True  # pitch is always embedded in v3
        # Override: OpenUTAU 0.1.565.0 doesn't support breathiness/voicing/tension
        # as dynamic inputs. These were frozen to zero constants during ONNX export.
        # Keep dsconfig flags as False so OpenUTAU doesn't try to generate curves.
        # Re-enable when OpenUTAU adds support for these variance types.
        OU_FROZEN_VARIANCES = {'breathiness', 'voicing', 'tension'}
        for v in OU_FROZEN_VARIANCES & set(VARIANCE_CHECKLIST):
            dsconfig[f'use_{v}_embed'] = False
        # SFC Breathness: glottal blend is read from user expression curve
        # in OpenUtau 0.1.568+glottal.  dsconfig flag tells the renderer to
        # show the glottal blend slider and pass it as ONNX input.
        if self.use_glottal:
            dsconfig['use_glottal_blend'] = True
        # sampling acceleration and shallow diffusion
        dsconfig['use_continuous_acceleration'] = True
        dsconfig['use_variable_depth'] = self.model.use_shallow_diffusion
        dsconfig['max_depth'] = 1 - self.model.diffusion.t_start
        # mel specification
        dsconfig['sample_rate'] = hparams['audio_sample_rate']
        dsconfig['hop_size'] = hparams['hop_size']
        dsconfig['win_size'] = hparams['win_size']
        dsconfig['fft_size'] = hparams['fft_size']
        dsconfig['num_mel_bins'] = hparams['audio_num_mel_bins']
        dsconfig['mel_fmin'] = hparams['fmin']
        dsconfig['mel_fmax'] = hparams['fmax'] if hparams['fmax'] is not None else hparams['audio_sample_rate'] / 2
        dsconfig['mel_base'] = 'e'
        dsconfig['mel_scale'] = 'slaney'
        config_path = path / 'dsconfig.yaml'
        with open(config_path, 'w', encoding='utf8') as fw:
            yaml.safe_dump(dsconfig, fw, sort_keys=False)
        print(f'| export configs => {config_path} **PLEASE EDIT BEFORE USE**')

    @torch.no_grad()
    def _torch_export_model(self):
        # Prepare inputs for FastSpeech2 and aux decoder tracing
        n_frames = 10
        tokens = torch.LongTensor([[1]]).to(self.device)
        durations = torch.LongTensor([[n_frames]]).to(self.device)
        f0 = torch.FloatTensor([[440.] * n_frames]).to(self.device)

        # ── OpenUTAU compatibility: freeze unsupported variance inputs ──
        # OpenUTAU 0.1.565.0 only natively supplies: energy (if use_energy_embed)
        # All other variance names (breathiness, voicing, tension) must be
        # registered as zero-onnx-constants rather than dynamic inputs.
        # languages is also frozen unless use_lang_id is explicitly True.
        OU_SUPPORTED_VARIANCES = {'energy'}
        variance_embed_list = list(self.model.fs2.variance_embed_list)
        ou_variances = [v for v in variance_embed_list if v in OU_SUPPORTED_VARIANCES]
        frozen_variances = [v for v in variance_embed_list if v not in OU_SUPPORTED_VARIANCES]
        # Freeze languages if use_lang_id is False (OpenUTAU won't provide it)
        freeze_languages = True  # Always freeze: OpenUTAU 0.1.565.0 doesn't support languages input

        # Register frozen zero buffers for unsupported variance + languages
        from deployment.modules.toplevel import DiffSingerAcousticONNX

        # ── Build the fs2_aux model FIRST, then register buffers on the clone ──
        fs2_model = self.model.view_as_fs2_aux()
        original_forward = fs2_model.forward_fs2_aux

        # Build variance dict with supported as inputs, frozen as buffers
        variances_input = {
            v_name: torch.zeros(1, n_frames, dtype=torch.float32, device=self.device)
            for v_name in ou_variances
        }
        for v_name in frozen_variances:
            buf = torch.zeros(1, n_frames, dtype=torch.float32)
            fs2_model.register_buffer(f'_frozen_{v_name}', buf, persistent=False)
        if freeze_languages:
            buf = torch.zeros(1, 1, dtype=torch.long)
            fs2_model.register_buffer('_frozen_languages', buf, persistent=False)

        # Monkey-patch forward_fs2_aux on the CLONE to inject frozen values
        def patched_forward(tokens, durations, f0, variances, gender=None,
                           velocity=None, spk_embed=None, languages=None,
                           glottal_blend=None):
            for v_name in frozen_variances:
                buf = getattr(fs2_model, f'_frozen_{v_name}')
                variances[v_name] = buf.expand(-1, f0.shape[1]).to(f0.device)
            if freeze_languages:
                lang_buf = getattr(fs2_model, '_frozen_languages')
                languages = lang_buf.expand(-1, tokens.shape[1]).to(tokens.device)
            return original_forward(tokens, durations, f0, variances,
                                    gender=gender, velocity=velocity,
                                    spk_embed=spk_embed, languages=languages,
                                    glottal_blend=glottal_blend)
        fs2_model.forward_fs2_aux = patched_forward

        # Build arguments and input_names (only OpenUTAU-supported)
        kwargs: Dict[str, torch.Tensor] = {}
        arguments = (tokens, durations, f0, variances_input, kwargs)
        input_names = ['tokens', 'durations', 'f0'] + ou_variances
        dynamix_axes = {
            'tokens': {1: 'n_tokens'},
            'durations': {1: 'n_tokens'},
            'f0': {1: 'n_frames'},
            **{v_name: {1: 'n_frames'} for v_name in ou_variances}
        }
        if hparams['use_key_shift_embed']:
            if self.expose_gender:
                kwargs['gender'] = torch.rand((1, n_frames), dtype=torch.float32, device=self.device)
                input_names.append('gender')
                dynamix_axes['gender'] = {
                    1: 'n_frames'
                }
        if hparams['use_speed_embed']:
            if self.expose_velocity:
                kwargs['velocity'] = torch.rand((1, n_frames), dtype=torch.float32, device=self.device)
                input_names.append('velocity')
                dynamix_axes['velocity'] = {
                    1: 'n_frames'
                }
        if hparams['use_spk_id'] and not self.freeze_spk:
            kwargs['spk_embed'] = torch.rand(
                (1, n_frames, hparams['hidden_size']),
                dtype=torch.float32, device=self.device
            )
            input_names.append('spk_embed')
            dynamix_axes['spk_embed'] = {
                1: 'n_frames'
            }
        # languages: always frozen (OpenUTAU 0.1.565.0 doesn't support)
        # The patched_forward injects a zero buffer internally.
        # SFC Breathness: glottal blend input (frame-level, range [-1, 1])
        # view_as_fs2_aux() nulls glottal_predictor (TransformerEncoder can't be
        # exported to opset 15 by PyTorch 1.13). Glottal injection is applied
        # externally by the SFCB post-processor. Do NOT export as ONNX input.
        dynamix_axes['condition'] = {
            1: 'n_frames'
        }

        # PyTorch ONNX export for FastSpeech2 and aux decoder
        output_names = ['condition']
        if self.model.use_shallow_diffusion:
            output_names.append('aux_mel')
            dynamix_axes['aux_mel'] = {
                1: 'n_frames'
            }
        print(f'Exporting {self.fs2_aux_class_name}...')
        torch.onnx.export(
            fs2_model,
            arguments,
            self.fs2_aux_cache_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamix_axes,
            opset_version=15
        )

        condition = torch.rand((1, n_frames, hparams['hidden_size']), device=self.device)

        # Prepare inputs for backbone tracing and GaussianDiffusion scripting
        shape = (1, 1, hparams['audio_num_mel_bins'], n_frames)
        noise = torch.randn(shape, device=self.device)
        x_aux = torch.randn((1, n_frames, hparams['audio_num_mel_bins']), device=self.device)
        dummy_time = (torch.rand((1,), device=self.device) * self.model.diffusion.time_scale_factor).float()
        dummy_depth = torch.tensor(0.1, device=self.device)
        dummy_steps = 5

        print(f'Tracing {self.backbone_class_name} backbone...')
        if self.model.diffusion_type == 'ddpm':
            major_mel_decoder = self.model.view_as_diffusion()
        elif self.model.diffusion_type == 'reflow':
            major_mel_decoder = self.model.view_as_reflow()
        else:
            raise ValueError(f'Invalid diffusion type: {self.model.diffusion_type}')
        major_mel_decoder.diffusion.set_backbone(
            torch.jit.trace(
                major_mel_decoder.diffusion.backbone,
                (
                    noise,
                    dummy_time,
                    condition.transpose(1, 2)
                )
            )
        )

        print(f'Scripting {self.diffusion_class_name}...')
        diffusion_inputs = [
            condition,
            *([x_aux, dummy_depth] if self.model.use_shallow_diffusion else [])
        ]
        major_mel_decoder = torch.jit.script(
            major_mel_decoder,
            example_inputs=[
                (
                    *diffusion_inputs,
                    1  # p_sample branch
                ),
                (
                    *diffusion_inputs,
                    dummy_steps  # p_sample_plms branch
                )
            ]
        )

        # PyTorch ONNX export for GaussianDiffusion
        print(f'Exporting {self.diffusion_class_name}...')
        torch.onnx.export(
            major_mel_decoder,
            (
                *diffusion_inputs,
                dummy_steps
            ),
            self.diffusion_cache_path,
            input_names=[
                'condition',
                *(['x_aux', 'depth'] if self.model.use_shallow_diffusion else []),
                'steps'
            ],
            output_names=[
                'mel'
            ],
            dynamic_axes={
                'condition': {
                    1: 'n_frames'
                },
                **({'x_aux': {1: 'n_frames'}} if self.model.use_shallow_diffusion else {}),
                'mel': {
                    1: 'n_frames'
                }
            },
            opset_version=15
        )

    @torch.no_grad()
    def _perform_spk_mix(self, spk_mix: Dict[str, float]):
        if not spk_mix:
            return torch.zeros(hparams['hidden_size'], device=self.device)
        spk_mix_ids = []
        spk_mix_values = []
        for name, value in spk_mix.items():
            spk_mix_ids.append(self.spk_map[name])
            assert value >= 0., f'Speaker mix checks failed.\n' \
                                f'Proportion of speaker \'{name}\' is negative.'
            spk_mix_values.append(value)
        spk_mix_id_N = torch.LongTensor(spk_mix_ids).to(self.device)[None]  # => [1, N]
        spk_mix_value_N = torch.FloatTensor(spk_mix_values).to(self.device)[None]  # => [1, N]
        spk_mix_value_sum = spk_mix_value_N.sum()
        assert spk_mix_value_sum > 0., f'Speaker mix checks failed.\n' \
                                       f'Proportions of speaker mix sum to zero.'
        spk_mix_value_N /= spk_mix_value_sum  # normalize
        spk_mix_embed = torch.sum(
            self.model.fs2.spk_embed(spk_mix_id_N) * spk_mix_value_N.unsqueeze(2),  # => [1, N, H]
            dim=1, keepdim=False
        )  # => [1, H]
        return spk_mix_embed

    def _optimize_fs2_aux_graph(self, fs2: onnx.ModelProto) -> onnx.ModelProto:
        print(f'Running ONNX Simplifier on {self.fs2_aux_class_name}...')
        try:
            fs2, check = onnxsim.simplify(fs2, include_subgraph=True)
            if check:
                print(f'| optimize graph: {self.fs2_aux_class_name}')
            else:
                print(f'| simplify failed validation, keeping un-optimized graph')
        except Exception as e:
            print(f'| onnxsim failed ({type(e).__name__}), keeping un-optimized graph')
        onnx_helper.model_reorder_io_list(
            fs2, 'input',
            target_name='languages', insert_after_name='tokens'
        )
        return fs2

    def _optimize_diffusion_graph(self, diffusion: onnx.ModelProto) -> onnx.ModelProto:
        onnx_helper.model_override_io_shapes(diffusion, output_shapes={
            'mel': (1, 'n_frames', hparams['audio_num_mel_bins'])
        })
        print(f'Running ONNX Simplifier #1 on {self.diffusion_class_name}...')
        try:
            diffusion, check = onnxsim.simplify(diffusion, include_subgraph=True)
            if not check:
                print(f'| simplify #1 failed validation')
        except Exception as e:
            print(f'| onnxsim #1 failed ({type(e).__name__}), continuing')
        onnx_helper.graph_fold_back_to_squeeze(diffusion.graph)
        onnx_helper.graph_extract_conditioner_projections(
            graph=diffusion.graph, op_type='Conv',
            weight_pattern=r'diffusion\..*\.conditioner_projection\.weight',
            alias_prefix='/diffusion/backbone/cache'
        )
        onnx_helper.graph_remove_unused_values(diffusion.graph)
        print(f'Running ONNX Simplifier #2 on {self.diffusion_class_name}...')
        try:
            diffusion, check = onnxsim.simplify(
                diffusion,
                include_subgraph=True
            )
            if check:
                print(f'| optimize graph: {self.diffusion_class_name}')
            else:
                print(f'| simplify #2 failed validation')
        except Exception as e:
            print(f'| onnxsim #2 failed ({type(e).__name__}), continuing')
        return diffusion

    def _export_model_glottal(self) -> 'onnx.ModelProto':
        """Export standalone GlottalConditioner ONNX for three-graph merge."""
        from deployment.modules.glottal_export import GlottalConditionerONNX

        n_frames = 10
        condition = torch.randn(1, n_frames, hparams['hidden_size'], device=self.device)
        f0 = torch.full((1, n_frames), 440.0, dtype=torch.float32, device=self.device)
        glottal_blend = torch.zeros(1, n_frames, dtype=torch.float32, device=self.device)

        glottal_model = GlottalConditionerONNX(
            self.model.glottal_predictor,
            self.model.glottal_adapter,
            self.model._glottal_proj,
        ).eval().to(self.device)

        cache_path = self.cache_dir / 'glottal.onnx'
        print(f'Exporting GlottalConditioner...')
        torch.onnx.export(
            glottal_model,
            (condition, f0, glottal_blend),
            str(cache_path),
            input_names=['condition', 'f0', 'glottal_blend'],
            output_names=['condition_out'],
            dynamic_axes={
                'condition': {1: 'n_frames'},
                'f0': {1: 'n_frames'},
                'glottal_blend': {1: 'n_frames'},
                'condition_out': {1: 'n_frames'},
            },
            opset_version=15,
        )
        return onnx.load(str(cache_path))

    def _optimize_glottal_graph(self, glottal: 'onnx.ModelProto') -> 'onnx.ModelProto':
        """Run ONNX Simplifier on glottal graph (best-effort)."""
        print(f'Running ONNX Simplifier on GlottalConditioner...')
        try:
            glottal, check = onnxsim.simplify(glottal, include_subgraph=True)
            if check:
                print(f'| optimize graph: GlottalConditioner')
            else:
                print(f'| simplify failed validation on glottal, keeping un-optimized')
        except Exception as e:
            print(f'| onnxsim failed on glottal ({type(e).__name__}), continuing')
        return glottal

    def _merge_fs2_aux_diffusion_graphs(self, fs2: onnx.ModelProto, diffusion: onnx.ModelProto,
                                         condition_key: str = 'condition') -> onnx.ModelProto:
        onnx_helper.model_add_prefixes(
            fs2, dim_prefix=('fs2aux.' if self.model.use_shallow_diffusion else 'fs2.'),
            ignored_pattern=r'(n_tokens)|(n_frames)'
        )
        onnx_helper.model_add_prefixes(diffusion, dim_prefix='diffusion.', ignored_pattern='n_frames')
        print(f'Merging {self.fs2_aux_class_name} and {self.diffusion_class_name} '
              f'back into {self.model_class_name}...')
        merged = onnx.compose.merge_models(
            fs2, diffusion, io_map=[
                (condition_key, 'condition'),
                *([('aux_mel', 'x_aux')] if self.model.use_shallow_diffusion else []),
            ],
            prefix1='', prefix2='', doc_string='',
            producer_name=fs2.producer_name, producer_version=fs2.producer_version,
            domain=fs2.domain, model_version=fs2.model_version
        )
        merged.graph.name = fs2.graph.name

        print(f'Running ONNX Simplifier on {self.model_class_name}...')
        try:
            merged, check = onnxsim.simplify(
                merged,
                include_subgraph=True
            )
            if check:
                print(f'| optimize graph: {self.model_class_name}')
            else:
                print(f'| simplify failed validation')
        except Exception as e:
            print(f'| onnxsim failed ({type(e).__name__}), continuing')

        return merged

    # noinspection PyMethodMayBeStatic
    def _export_spk_embed(self, path: Path, spk_embed: torch.Tensor):
        with open(path, 'wb') as f:
            f.write(spk_embed.cpu().numpy().tobytes())
        print(f'| export spk embed => {path}')

    def _export_phonemes(self, path: Path):
        # Use experiment's ph_map.json directly — this is the ground-truth
        # mapping the checkpoint was trained with. Rebuilding from dictionary
        # files can produce different IDs if merged_groups config is missing.
        ph_map_src = Path(hparams['work_dir']) / 'ph_map.json'
        ph_path = path / f'{self.model_name}.phonemes.json'
        if ph_map_src.exists():
            shutil.copy(ph_map_src, ph_path)
            print(f'| export phonemes => {ph_path}')
        else:
            # Fallback: rebuild from PhonemeDictionary
            self.phoneme_dictionary.dump(ph_path)
            print(f'| export phonemes => {ph_path} (rebuilt from dictionaries)')
        lang_path = path / f'{self.model_name}.languages.json'
        with open(lang_path, 'w', encoding='utf8') as f:
            json.dump(self.lang_map, f, ensure_ascii=False, indent=2)
        print(f'| export languages => {lang_path}')
