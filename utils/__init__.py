from __future__ import annotations

import pathlib
import re
from collections import OrderedDict

import torch


def load_ckpt(
        cur_model, ckpt_base_dir, ckpt_steps=None,
        prefix_in_ckpt='model', ignored_prefixes=None, key_in_ckpt='state_dict',
        strict=True, device='cpu'
):
    """Load checkpoint with automatic v3→v2 key remapping for ONNX deployment.

    v3 training checkpoints store:
        linguistic_encoder.token_embedding
        linguistic_encoder.duration_embedding
        linguistic_encoder.language_embedding
        linguistic_encoder.encoder.layers.*
        parameter_embeddings.f0_embed / pitch_embed / variance_*
        speaker_embedding

    v2 ONNX deployment models expect:
        fs2.txt_embed
        fs2.dur_embed
        fs2.lang_embed
        fs2.encoder.layers.*
        fs2.f0_embed / fs2.pitch_embed / fs2.variance_*
        fs2.spk_embed

    The key_mapping list defines (pattern, replacement) pairs applied in order."""
    if ignored_prefixes is None:
        ignored_prefixes = ['model.fs2.encoder.embed_tokens']
    if not isinstance(ckpt_base_dir, pathlib.Path):
        ckpt_base_dir = pathlib.Path(ckpt_base_dir)
    if ckpt_base_dir.is_file():
        checkpoint_path = [ckpt_base_dir]
    elif ckpt_steps is not None:
        # Support both naming conventions:
        #   v3: model-temp-steps=0054000-epochs=0818.ckpt
        #   v2: model_ckpt_steps_54000.ckpt
        base_dir = ckpt_base_dir
        steps = int(ckpt_steps)
        # Pad to 7 digits (v3 convention) and try matching
        ckpt_files = sorted(
            [
                ckpt_file for ckpt_file in base_dir.iterdir() if ckpt_file.is_file()
                and re.search(rf'steps[_=]0*{steps}\b', ckpt_file.name)
            ],
            key=lambda x: x.name
        )
        if ckpt_files:
            checkpoint_path = [ckpt_files[-1]]
        else:
            # Fallback: try exact v2 naming
            fallback = ckpt_base_dir / f'model_ckpt_steps_{steps}.ckpt'
            if fallback.exists():
                checkpoint_path = [fallback]
            else:
                # Last resort: list available ckpts and take newest
                all_ckpts = sorted(base_dir.glob('*.ckpt'), key=lambda x: x.name)
                if all_ckpts:
                    print(f'| WARNING: no ckpt matched steps={steps}, using newest: {all_ckpts[-1].name}')
                    checkpoint_path = [all_ckpts[-1]]
                else:
                    available = list(base_dir.glob('*'))
                    raise FileNotFoundError(
                        f'No checkpoint found for steps={steps} in {ckpt_base_dir}.\n'
                        f'Available files: {[f.name for f in available][:20]}'
                    )
    else:
        base_dir = ckpt_base_dir
        checkpoint_path = sorted(
            [
                ckpt_file
                for ckpt_file in base_dir.iterdir()
                if ckpt_file.is_file()
                and re.search(r'model.*[-_]steps[=_]\d+.*\.ckpt', ckpt_file.name)
            ],
            key=lambda x: x.name
        )
    assert len(checkpoint_path) > 0, f'| ckpt not found in {ckpt_base_dir}.'
    checkpoint_path = checkpoint_path[-1]
    ckpt_loaded = torch.load(checkpoint_path, map_location=device)
    # v3 models are plain nn.Module, not CategorizedModule
    if hasattr(cur_model, 'check_category') and 'category' in ckpt_loaded:
        cur_model.check_category(ckpt_loaded.get('category'))
    if key_in_ckpt is None:
        state_dict = ckpt_loaded
    else:
        state_dict = ckpt_loaded[key_in_ckpt]
    if prefix_in_ckpt is not None:
        state_dict = OrderedDict({
            k[len(prefix_in_ckpt) + 1:]: v
            for k, v in state_dict.items() if k.startswith(f'{prefix_in_ckpt}.')
            if all(not k.startswith(p) for p in ignored_prefixes)
        })
    if not strict:
        cur_model_state_dict = cur_model.state_dict()
        unmatched_keys = []
        for key, param in state_dict.items():
            if key in cur_model_state_dict:
                new_param = cur_model_state_dict[key]
                if new_param.shape != param.shape:
                    unmatched_keys.append(key)
                    print('| Unmatched keys: ', key, new_param.shape, param.shape)
        for key in unmatched_keys:
            del state_dict[key]
    # ── v3 → v2 key remapping for ONNX deployment ──
    # Check if this checkpoint uses v3 naming (linguistic_encoder, parameter_embeddings)
    # and the model expects v2 naming (fs2.*). If so, apply automatic remapping.
    v3_keys = [k for k in state_dict if 'linguistic_encoder' in k or 'parameter_embeddings' in k]
    model_keys = set(cur_model.state_dict().keys())
    if v3_keys and any('fs2.' in mk for mk in model_keys):
        print('| Detected v3 checkpoint → applying automatic key remapping for ONNX deployment')
        mappings = [
            # linguistic_encoder → fs2
            ('linguistic_encoder.token_embedding', 'fs2.txt_embed'),
            ('linguistic_encoder.duration_embedding', 'fs2.dur_embed'),
            ('linguistic_encoder.language_embedding', 'fs2.lang_embed'),
            ('linguistic_encoder.encoder', 'fs2.encoder'),
            # parameter_embeddings → fs2
            ('parameter_embeddings.f0_embed', 'fs2.f0_embed'),
            ('parameter_embeddings.pitch_embed', 'fs2.pitch_embed'),
            ('parameter_embeddings.variance_embed_list', 'fs2.variance_embed_list'),
            ('parameter_embeddings.variance_predictor_list', 'fs2.variance_predictor_list'),
            ('parameter_embeddings.key_shift_embed', 'fs2.key_shift_embed'),
            ('parameter_embeddings.speed_embed', 'fs2.speed_embed'),
            ('parameter_embeddings.energy_embed', 'fs2.energy_embed'),
            ('parameter_embeddings.breathiness_embed', 'fs2.breathiness_embed'),
            ('parameter_embeddings.voicing_embed', 'fs2.voicing_embed'),
            ('parameter_embeddings.tension_embed', 'fs2.tension_embed'),
            # speaker_embedding → fs2.spk_embed
            ('speaker_embedding', 'fs2.spk_embed'),
            # spec_decoder → diffusion / aux_decoder (SFC shallow diffusion)
            ('spec_decoder.decoder', 'diffusion'),
            ('spec_decoder.aux_decoder', 'aux_decoder'),
        ]
        new_state_dict = OrderedDict()
        remapped = set()
        for ckpt_key, ckpt_tensor in state_dict.items():
            new_key = ckpt_key
            for old_prefix, new_prefix in mappings:
                if ckpt_key.startswith(old_prefix):
                    new_key = new_prefix + ckpt_key[len(old_prefix):]
                    remapped.add(old_prefix)
                    break
            if new_key in model_keys:
                new_state_dict[new_key] = ckpt_tensor
        state_dict = new_state_dict
        print(f'| Remapped {len(remapped)} key prefixes, {len(state_dict)} keys loaded')

    cur_model.load_state_dict(state_dict, strict=strict)
    shown_model_name = 'state dict'
    if prefix_in_ckpt is not None:
        shown_model_name = f'\'{prefix_in_ckpt}\''
    elif key_in_ckpt is not None:
        shown_model_name = f'\'{key_in_ckpt}\''
    print(f'| load {shown_model_name} from \'{checkpoint_path}\'.')


def remove_suffix(string: str, suffix: str):
    #  Just for Python 3.8 compatibility, since `str.removesuffix()` API of is available since Python 3.9
    if string.endswith(suffix):
        string = string[:-len(suffix)]
    return string
