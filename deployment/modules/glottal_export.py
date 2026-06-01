"""ONNX-exportable Glottal Conditioner for SFC Breathness.

PyTorch 1.13 cannot export TransformerEncoder (GlottalParamPredictor) inside the
same torch.onnx.export call as fs2. This module wraps the glottal pipeline as a
standalone nn.Module that can be exported independently and merged into the
final ONNX graph.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _GlottalPredictorTuple(nn.Module):
    """Wraps GlottalParamPredictor to return tuple instead of dict.

    PyTorch 1.13 ONNX export cannot trace dict returns from submodules.
    This wrapper changes dict → tuple for full trace compatibility.
    """

    def __init__(self, predictor: nn.Module):
        super().__init__()
        self.predictor = predictor

    def forward(self, cond: torch.Tensor, f0: torch.Tensor):
        out = self.predictor(cond, f0)  # returns dict
        return out['Rd'], out['OQ'], out['AQ'], out['noise_amp']


class GlottalConditionerONNX(nn.Module):
    """ONNX-exportable glottal conditioner.

    Inputs:  condition [B, T, hidden_size], f0 [B, T], glottal_blend [B, T]
    Outputs: condition_out [B, T, hidden_size]

    Pipeline:
      1. GlottalParamPredictor(condition, f0) → LF params (Rd, OQ, AQ, noise_amp)
      2. GlottalToDiffSingerAdapter(Rd, OQ, AQ, noise_amp)[0] → breathiness [B,T]
      3. Linear(breathiness) → glottal_cond [B,T,hidden_size]
      4. condition_out = condition + glottal_cond * glottal_blend
    """

    def __init__(
        self,
        glottal_predictor: nn.Module,
        glottal_adapter: nn.Module,
        glottal_proj: nn.Linear,
    ):
        super().__init__()
        self.predictor = _GlottalPredictorTuple(glottal_predictor)
        self.adapter = glottal_adapter
        self.proj = glottal_proj

    def forward(
        self,
        condition: torch.Tensor,
        f0: torch.Tensor,
        glottal_blend: torch.Tensor,
    ) -> torch.Tensor:
        Rd, OQ, AQ, noise_amp = self.predictor(condition, f0)
        breathiness = self.adapter(Rd, OQ, AQ, noise_amp)[0]
        glottal_cond = self.proj(breathiness.unsqueeze(-1))
        return condition + glottal_cond * glottal_blend.unsqueeze(-1)
