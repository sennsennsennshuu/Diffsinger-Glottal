import torch.nn as nn
from .reflow import RectifiedFlow

# ---- Deployment-compatibility aliases ----
# The original DiffSinger v2 split diffusion classes (GaussianDiffusion,
# PitchDiffusion, etc.) were unified into RectifiedFlow in v3.
# The ONNX deployment code references the legacy names.
# We create dummy aliases so deployment modules can import without errors.

class _GaussianDiffusionDummy(nn.Module):
    """Dummy base class for legacy GaussianDiffusion import."""
    pass

GaussianDiffusion = _GaussianDiffusionDummy
PitchDiffusion = _GaussianDiffusionDummy
MultiVarianceDiffusion = _GaussianDiffusionDummy

class _RectifiedFlowDummy(RectifiedFlow):
    """Dummy subclass for legacy PitchRectifiedFlow/MultiVarianceRectifiedFlow."""
    pass

PitchRectifiedFlow = _RectifiedFlowDummy
MultiVarianceRectifiedFlow = _RectifiedFlowDummy
