"""TDD test suite for GlottalConditionerONNX export.

RED → GREEN → REFACTOR. Each test must FAIL before implementation.
"""
import sys
import os
from pathlib import Path
import tempfile

import pytest
import torch
import numpy as np

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

# ── Test 1: GlottalConditionerONNX forward shape ──

def test_glottal_conditioner_forward_shapes():
    """GlottalConditionerONNX forward → output shape matches input condition."""
    from modules.sweet_breathy.lf_glottal_conditioner import (
        GlottalParamPredictor, GlottalToDiffSingerAdapter,
    )
    import torch.nn as nn

    hidden = 256
    B, T = 1, 10

    predictor = GlottalParamPredictor(d_cond=hidden)
    adapter = GlottalToDiffSingerAdapter()
    proj = nn.Linear(1, hidden)

    from deployment.modules.glottal_export import GlottalConditionerONNX

    model = GlottalConditionerONNX(predictor, adapter, proj).eval()

    condition = torch.randn(B, T, hidden)
    f0 = torch.full((B, T), 440.0, dtype=torch.float32)
    glottal_blend = torch.zeros(B, T, dtype=torch.float32)

    with torch.no_grad():
        cond_out = model(condition, f0, glottal_blend)

    assert cond_out.shape == (B, T, hidden), f"Expected shape {(B, T, hidden)}, got {cond_out.shape}"


# ── Test 2: glottal_blend=0 is identity ──

def test_glottal_blend_zero_is_identity():
    """When glottal_blend=0, condition_out == condition."""
    from modules.sweet_breathy.lf_glottal_conditioner import (
        GlottalParamPredictor, GlottalToDiffSingerAdapter,
    )
    import torch.nn as nn

    hidden = 256
    B, T = 1, 8

    predictor = GlottalParamPredictor(d_cond=hidden)
    adapter = GlottalToDiffSingerAdapter()
    proj = nn.Linear(1, hidden)

    from deployment.modules.glottal_export import GlottalConditionerONNX

    model = GlottalConditionerONNX(predictor, adapter, proj).eval()

    condition = torch.randn(B, T, hidden)
    f0 = torch.full((B, T), 220.0, dtype=torch.float32)
    glottal_blend = torch.zeros(B, T, dtype=torch.float32)

    with torch.no_grad():
        cond_out = model(condition, f0, glottal_blend)

    # With blend=0, the glottal_cond term is multiplied by 0 → cond_out == cond
    assert torch.allclose(cond_out, condition, atol=1e-6), \
        f"Expected identity, max diff: {(cond_out - condition).abs().max().item()}"


# ── Test 3: glottal_blend=+1 changes output ──

def test_glottal_blend_positive_changes_output():
    """When glottal_blend=+1, condition_out ≠ condition."""
    from modules.sweet_breathy.lf_glottal_conditioner import (
        GlottalParamPredictor, GlottalToDiffSingerAdapter,
    )
    import torch.nn as nn

    hidden = 256
    B, T = 1, 8

    predictor = GlottalParamPredictor(d_cond=hidden)
    adapter = GlottalToDiffSingerAdapter()
    proj = nn.Linear(1, hidden)

    from deployment.modules.glottal_export import GlottalConditionerONNX

    model = GlottalConditionerONNX(predictor, adapter, proj).eval()

    condition = torch.randn(B, T, hidden)
    f0 = torch.full((B, T), 330.0, dtype=torch.float32)
    glottal_blend = torch.ones(B, T, dtype=torch.float32)

    with torch.no_grad():
        cond_out = model(condition, f0, glottal_blend)

    # With blend=+1, glottal_cond is fully applied → output should differ
    assert not torch.allclose(cond_out, condition, atol=1e-5), \
        "Expected condition_out to differ from condition with blend=+1"


# ── Test 4: ONNX export succeeds ──

def test_onnx_export_glottal():
    """torch.onnx.export(glottal_model) completes and file exists."""
    from modules.sweet_breathy.lf_glottal_conditioner import (
        GlottalParamPredictor, GlottalToDiffSingerAdapter,
    )
    import torch.nn as nn

    hidden = 256
    B, T = 1, 10

    predictor = GlottalParamPredictor(d_cond=hidden)
    adapter = GlottalToDiffSingerAdapter()
    proj = nn.Linear(1, hidden)

    from deployment.modules.glottal_export import GlottalConditionerONNX

    model = GlottalConditionerONNX(predictor, adapter, proj).eval()

    condition = torch.randn(B, T, hidden)
    f0 = torch.full((B, T), 440.0, dtype=torch.float32)
    glottal_blend = torch.zeros(B, T, dtype=torch.float32)

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        tmp_path = f.name

    try:
        torch.onnx.export(
            model,
            (condition, f0, glottal_blend),
            tmp_path,
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
        assert Path(tmp_path).exists(), "ONNX file not created"
        assert Path(tmp_path).stat().st_size > 0, "ONNX file is empty"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Test 5: ONNX inference matches PyTorch ──

def test_onnx_inference_matches_pytorch():
    """ONNX Runtime inference produces same result as PyTorch forward."""
    import onnxruntime as ort

    from modules.sweet_breathy.lf_glottal_conditioner import (
        GlottalParamPredictor, GlottalToDiffSingerAdapter,
    )
    import torch.nn as nn

    hidden = 256
    B, T = 1, 8

    predictor = GlottalParamPredictor(d_cond=hidden)
    adapter = GlottalToDiffSingerAdapter()
    proj = nn.Linear(1, hidden)

    from deployment.modules.glottal_export import GlottalConditionerONNX

    model = GlottalConditionerONNX(predictor, adapter, proj).eval()

    condition = torch.randn(B, T, hidden)
    f0 = torch.full((B, T), 550.0, dtype=torch.float32)
    glottal_blend = torch.tensor(
        [[0.0, 0.3, -0.2, 0.8, -1.0, 0.5, 0.0, 0.7]], dtype=torch.float32
    )

    with torch.no_grad():
        pt_out = model(condition, f0, glottal_blend)

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        tmp_path = f.name

    try:
        torch.onnx.export(
            model,
            (condition, f0, glottal_blend),
            tmp_path,
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

        sess = ort.InferenceSession(tmp_path, providers=['CPUExecutionProvider'])
        ort_inputs = {
            'condition': condition.numpy().astype(np.float32),
            'f0': f0.numpy().astype(np.float32),
            'glottal_blend': glottal_blend.numpy().astype(np.float32),
        }
        ort_out = sess.run(None, ort_inputs)[0]

        assert ort_out.shape == (B, T, hidden), f"ORT shape: {ort_out.shape}"
        assert np.allclose(ort_out, pt_out.numpy(), atol=1e-5), \
            f"ORT vs PyTorch max diff: {np.abs(ort_out - pt_out.numpy()).max()}"

    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Test 6: Final ONNX has glottal_blend input ──
# Tests the full export flow (requires real checkpoint)

@pytest.mark.skip(reason="Needs trained checkpoint")
def test_final_onnx_has_glottal_blend_input():
    """Full export flow: merged ONNX must have glottal_blend as input."""
    import sys as _sys
    _sys.argv = [_sys.argv[0], '--exp_name', 'aco_testf', '--infer']
    from utils.hparams import set_hparams
    set_hparams()

    from deployment.exporters.acoustic_exporter import DiffSingerAcousticExporter

    exporter = DiffSingerAcousticExporter(
        device='cpu',
        cache_dir=Path('deployment/cache'),
        ckpt_steps=40000,
        freeze_gender=None,
        freeze_velocity=False,
        export_spk=None,
        freeze_spk=None,
    )
    assert exporter.use_glottal, "Test requires a model with glottal weights"

    import tempfile
    out = Path(tempfile.mkdtemp())
    try:
        exporter.export(out)
        onnx_path = out / 'aco_testf.onnx'
        assert onnx_path.exists(), "ONNX model not exported"

        import onnx
        m = onnx.load(str(onnx_path))
        inputs = {i.name for i in m.graph.input}

        assert 'glottal_blend' in inputs, \
            f"glottal_blend missing from ONNX inputs: {sorted(inputs)}"
    finally:
        import shutil
        shutil.rmtree(out, ignore_errors=True)


@pytest.mark.skip(reason="Needs trained checkpoint")
def test_spec_decoder_weights_loaded_to_diffusion_and_aux():
    """Full export: diffusion backbone and aux_decoder weights MUST be loaded
    from checkpoint (not random init).  Regression test for buzzing noise bug
    caused by missing v3→v2 key remapping (spec_decoder.* → diffusion/aux_decoder).
    """
    import numpy as np
    import onnx
    import sys as _sys
    _sys.argv = [_sys.argv[0], '--exp_name', 'aco_testf', '--infer']
    from utils.hparams import set_hparams
    set_hparams()

    from deployment.exporters.acoustic_exporter import DiffSingerAcousticExporter

    exporter = DiffSingerAcousticExporter(
        device='cpu',
        cache_dir=Path('deployment/cache'),
        ckpt_steps=40000,
        freeze_gender=None,
        freeze_velocity=False,
        export_spk=None,
        freeze_spk=None,
    )
    assert exporter.use_glottal, "Test requires a model with glottal weights"

    import tempfile
    out = Path(tempfile.mkdtemp())
    try:
        exporter.export(out)
        onnx_path = out / 'aco_testf.onnx'
        assert onnx_path.exists(), "ONNX model not exported"

        m = onnx.load(str(onnx_path))
        weight_map = {i.name: i for i in m.graph.initializer}

        # Must have diffusion backbone weights
        diff_keys = [k for k in weight_map if k.startswith('diffusion.')]
        assert len(diff_keys) >= 10, \
            f"Expected >=10 diffusion keys, got {len(diff_keys)}: {sorted(diff_keys)}"

        # output_projection MUST be non-zero (not random init)
        out_key = next((k for k in weight_map if 'output_projection.weight' in k), None)
        assert out_key is not None, "output_projection.weight not found in ONNX model"
        out_proj = onnx.numpy_helper.to_array(weight_map[out_key])
        assert np.abs(out_proj).sum() > 1e-6, \
            f"output_projection.weight is all zeros (not loaded from checkpoint)"

        # aux_decoder weights must exist
        aux_keys = [k for k in weight_map if k.startswith('aux_decoder.')]
        assert len(aux_keys) >= 5, \
            f"Expected >=5 aux_decoder keys, got {len(aux_keys)}: {sorted(aux_keys)}"

    finally:
        import shutil
        shutil.rmtree(out, ignore_errors=True)


# ── Test 7: dsconfig has use_glottal_blend ──

@pytest.mark.skip(reason="Needs trained checkpoint")
def test_dsconfig_has_use_glottal_blend():
    """Full export flow: dsconfig.yaml must contain use_glottal_blend: True (for OpenUtau renderer)."""
    import yaml
    import sys as _sys
    _sys.argv = [_sys.argv[0], '--exp_name', 'aco_testf', '--infer']
    from utils.hparams import set_hparams
    set_hparams()

    from deployment.exporters.acoustic_exporter import DiffSingerAcousticExporter

    exporter = DiffSingerAcousticExporter(
        device='cpu',
        cache_dir=Path('deployment/cache'),
        ckpt_steps=40000,
        freeze_gender=None,
        freeze_velocity=False,
        export_spk=None,
        freeze_spk=None,
    )
    assert exporter.use_glottal, "Test requires a model with glottal weights"

    import tempfile
    out = Path(tempfile.mkdtemp())
    try:
        exporter.export(out)
        dc_path = out / 'dsconfig.yaml'
        assert dc_path.exists(), "dsconfig.yaml not exported"

        with open(dc_path) as f:
            dc = yaml.safe_load(f)
        # glottal_blend is now an ONNX input fed by OpenUtau expression curve.
        # dsconfig key tells the renderer to show the slider and read the value.
        assert dc['use_glottal_blend'] is True, \
            f"use_glottal_blend={dc.get('use_glottal_blend')}, expected True"
    finally:
        import shutil
        shutil.rmtree(out, ignore_errors=True)
