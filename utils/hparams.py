"""Minimal hparams bridge for ONNX export and inference.

The deployment modules (deployment/modules/, basics/) reference ``utils.hparams.hparams``
as a global dict. This file loads the YAML config and exposes it via a thread-local
namespace, compatible with the original DiffSinger hparams pattern.

Usage:
    from utils.hparams import set_hparams, hparams
    set_hparams()                    # load from sys.argv
    print(hparams['hidden_size'])   # access as dict
"""

import os
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent

class _HParams(dict):
    """Dict subclass that allows attribute-style access, e.g. hparams.use_spk_id."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

hparams = _HParams()

def _deep_flatten(d, parent_key='', sep='_', max_depth=1):
    """Flatten nested dicts up to max_depth levels, e.g. model.hidden_size → hidden_size."""
    items = {}
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict) and max_depth > 0:
            items.update(_deep_flatten(v, new_key, sep, max_depth - 1))
            # Also store the nested dict itself for code that expects it
            items[new_key] = v
        else:
            items[new_key] = v
    return items

# ── Comprehensive defaults for ONNX export ───────────────────
# These cover every hparams key referenced anywhere in the export chain.
# Keys found in the actual config YAML will override these defaults.

_DEFAULTS = {
    # ── Model architecture ──
    'hidden_size': 256,
    'vocab_size': 100,
    'audio_num_mel_bins': 128,
    
    # ── Diffusion ──
    'diffusion_type': 'reflow',
    'backbone_type': 'lynxnet',
    'backbone_args': {
        'num_channels': 1024, 'num_layers': 6, 'kernel_size': 31,
        'dropout_rate': 0.0, 'strong_cond': True,
    },
    'T_start': 0.4,
    'time_scale_factor': 1000,
    'timesteps': 1000,
    'K_step': 400,
    'spec_min': [-6.0],
    'spec_max': [0.0],
    'mel_base': 'e',
    'sampling_algorithm': 'euler',
    'sampling_steps': 20,
    'max_beta': 0.06,
    'schedule_type': 'linear',
    
    # ── Feature flags ──
    'use_spk_id': False,
    'num_spk': 1,
    'use_lang_id': False,
    'num_lang': 1,
    'use_key_shift_embed': True,
    'use_speed_embed': True,
    'f0_embed_type': 'continuous',
    'use_variance_embeds': False,
    'use_breathiness_embed': False,
    'use_voicing_embed': False,
    'use_tension_embed': False,
    'use_energy_embed': False,
    'use_pos_embed': True,
    'rel_pos': True,
    'use_rope': True,
    'rope_interleaved': False,
    'use_shallow_diffusion': True,
    'use_melody_encoder': False,
    'use_glide_embed': False,
    
    # ── Encoder ──
    'enc_layers': 4,
    'enc_ffn_kernel_size': 3,
    'ffn_act': 'gelu',
    'num_heads': 2,
    'dropout': 0.1,
    'predictor_hidden': 256,
    'dur_predictor_kernel': 5,
    
    # ── Shallow diffusion ──
    'shallow_diffusion_args': {
        'train_aux_decoder': True, 'train_diffusion': True,
        'val_gt_start': False, 'aux_decoder_arch': 'convnext',
        'aux_decoder_args': {
            'num_channels': 512, 'num_layers': 6,
            'kernel_size': 7, 'dropout_rate': 0.1,
        },
        'aux_decoder_grad': 0.1,
    },
    
    # ── Augmentation ──
    'augmentation_args': {
        'random_pitch_shifting': {
            'enabled': True, 'range': [-5.0, 5.0], 'scale': 0.75,
        },
        'random_time_stretching': {
            'enabled': True, 'range': [0.5, 2.0], 'scale': 0.75,
        },
    },
    
    # ── Audio / mel spec ──
    'audio_sample_rate': 44100,
    'hop_size': 512,
    'win_size': 2048,
    'fft_size': 2048,
    'fmin': 40,
    'fmax': 16000,
    
    # ── MIDI / pitch ──
    'midi_smooth_width': 0.06,
    'pitch_backbone_type': 'wavenet',
    'pitch_backbone_args': {
        'num_layers': 20, 'num_channels': 256, 'dilation_cycle_length': 5,
    },
    'pitch_prediction_args': {
        'pitd_norm_min': -8.0, 'pitd_norm_max': 8.0,
        'pitd_clip_min': -12.0, 'pitd_clip_max': 12.0,
        'repeat_bins': 64,
        'backbone_type': 'wavenet',
        'backbone_args': {
            'num_layers': 20, 'num_channels': 256, 'dilation_cycle_length': 5,
        },
    },
    
    # ── Training ──
    'dataset_size_key': 'lengths',
    'lr': 0.0006,
    'infer': True,
    'val_with_vocoder': False,
    'gradient_clip_val': 3.0,
    'accumulate_grad_batches': 4,
    
    # ── Dictionaries ──
    'dictionaries': {},

    # ── SFC Breathness ──
    'sfc_breathy': {
        'enabled': True,
        'use_lf_glottal': True,
        'glottal_hidden_dim': 128,
        'glottal_warmup_steps': 5000,
        'breathy_warmup_full_step': 64000,
        'ddsp_gru_dropout': 0.2,
        'ddsp_output_dropout': 0.1,
        'aperiodic_vae_latent_dim': 32,
        'cvae_dropout': 0.2,
        'period_singer_loss_scale': 0.3,
        'perceptual_loss_scale': 0.3,
    },
    
    # ── Placeholders (filled dynamically) ──
    'exp_name': 'default',
    'work_dir': '',
}

def set_hparams(config_path: str = None):
    """Load YAML configuration into the global hparams dict.

    If config_path is not provided, it is inferred from ``sys.argv`` (matching
    the pattern ``--config <path>`` used by ``scripts/export.py``).
    """
    if config_path is None:
        if '--config' in sys.argv:
            idx = sys.argv.index('--config')
            if idx + 1 < len(sys.argv):
                config_path = sys.argv[idx + 1]
        elif '--exp_name' in sys.argv:
            idx = sys.argv.index('--exp_name')
            if idx + 1 < len(sys.argv):
                exp_name = sys.argv[idx + 1]
                exp_dir = ROOT_DIR / 'experiments' / exp_name
                candidates = [
                    exp_dir / 'config.yaml',
                    exp_dir / 'hparams.yaml',
                ]
                # Prefer saved hparams-*.yaml (full config incl. data section)
                if exp_dir.exists():
                    hparams_files = sorted(exp_dir.glob('hparams-*.yaml'))
                    if hparams_files:
                        candidates.insert(0, hparams_files[-1])  # newest FIRST
                for c in candidates:
                    if c.exists():
                        config_path = str(c)
                        print(f'| using config: {c.name}')
                        break

    # ── Start with comprehensive defaults ──
    result = dict(_DEFAULTS)

    # ── Try to load and merge actual config ──
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Deep-flatten v3 Pydantic config (model.hidden_size → hidden_size)
        if isinstance(config, dict):
            flat = _deep_flatten(config, max_depth=2)
            # Merge flattened keys (overrides defaults)
            for k, v in flat.items():
                if not k.startswith(('data_', 'training_', 'binarizer_')):
                    result[k] = v

            # Also merge known section prefixes directly
            for prefix in ('model', 'training', 'data', 'binarizer', 'inference'):
                if prefix in config and isinstance(config[prefix], dict):
                    section = _deep_flatten(config[prefix], max_depth=1)
                    for k, v in section.items():
                        # Strip prefix: model_hidden_size → hidden_size
                        short = k[len(prefix)+1:] if k.startswith(prefix + '_') else k
                        result[short] = v
                        # Also keep prefixed version for nested access
                        if prefix not in result:
                            result[prefix] = {}
                        if isinstance(result[prefix], dict):
                            result[prefix][short] = v

            # Top-level non-nested keys
            for k, v in config.items():
                if k not in ('model', 'training', 'data', 'binarizer', 'inference'):
                    result[k] = v

    # ── Dynamic keys ──
    if '--exp_name' in sys.argv:
        idx = sys.argv.index('--exp_name')
        if idx + 1 < len(sys.argv):
            exp_name = sys.argv[idx + 1]
            result['exp_name'] = exp_name
            result['work_dir'] = str(ROOT_DIR / 'experiments' / exp_name)

    if not result.get('work_dir'):
        result['work_dir'] = str(ROOT_DIR / 'experiments' / result.get('exp_name', 'default'))

    # ── Post-processing: ensure lists for spec_min/max ──
    for k in ('spec_min', 'spec_max'):
        if isinstance(result.get(k), (int, float)):
            result[k] = [float(result[k])]

    # ── Ensure augmentation_args sub-keys ──
    if 'augmentation_args' not in result or not isinstance(result['augmentation_args'], dict):
        result['augmentation_args'] = dict(_DEFAULTS['augmentation_args'])
    aug = result['augmentation_args']
    for sub in ('random_pitch_shifting', 'random_time_stretching'):
        if sub not in aug or not isinstance(aug.get(sub), dict):
            aug[sub] = dict(_DEFAULTS['augmentation_args'][sub])
        if 'range' not in aug[sub]:
            aug[sub]['range'] = _DEFAULTS['augmentation_args'][sub]['range']

    # ── Dictionaries fallback ──
    if not result.get('dictionaries'):
        dict_dir = ROOT_DIR / 'dictionaries'
        if dict_dir.exists():
            txt_files = list(dict_dir.glob('*.txt'))
            if txt_files:
                lang = txt_files[0].stem.split('-')[0] or 'zh'
                result['dictionaries'] = {lang: str(txt_files[0].relative_to(ROOT_DIR))}

    hparams.clear()
    hparams.update(result)

    print(f'| hparams loaded: {len(result)} keys (exp={result.get("exp_name")}, '
          f'audio_num_mel_bins={result.get("audio_num_mel_bins")}, '
          f'hidden_size={result.get("hidden_size")})')
