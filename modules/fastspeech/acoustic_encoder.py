"""FastSpeech2Acoustic stub for ONNX deployment compatibility.

This stub recreates the legacy FastSpeech2Acoustic class signature required by
``deployment/modules/fastspeech2.py`` (FastSpeech2AcousticONNX superclass).

In DiffSinger v3 the original split encoder was replaced by a unified
LinguisticEncoder/MelodyEncoder.  This stub exists purely so the ONNX export
pipeline can instantiate its deployment subclasses without importing the
removed v2 modules.
"""

import torch
import torch.nn as nn

from modules.commons.common_layers import NormalInitEmbedding as Embedding
from modules.commons.common_layers import XavierUniformInitLinear as Linear
from modules.commons.tts_modules import FastSpeech2Encoder
from utils.hparams import hparams


class FastSpeech2Acoustic(nn.Module):
    """Legacy FastSpeech2 acoustic encoder.

    Mirrors the original v2 class attributes that the ONNX subclass
    (FastSpeech2AcousticONNX) expects at init and during forward.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        hidden_size = hparams.get('hidden_size', 256)

        # ---- Token / duration / language embeddings ----
        self.txt_embed = Embedding(vocab_size, hidden_size, padding_idx=0)
        self.dur_embed = Linear(1, hidden_size)
        self.use_lang_id = hparams.get('use_lang_id', False)
        if self.use_lang_id:
            num_lang = hparams.get('num_lang', 3) + 1
            self.lang_embed = Embedding(num_lang, hidden_size, padding_idx=0)

        # ---- Encoder backbone (Transformer / Conformer exported by FS2Encoder) ----
        self.encoder = FastSpeech2Encoder(
            hidden_size=hidden_size,
            use_pos_embed=True,
            num_layers=4,
            num_heads=2,
            ffn_kernel_size=3,
            ffn_act='gelu',
            dropout=0.1,
            use_rope=True,
        )

        # ---- F0 embedding ----
        self.f0_embed_type = hparams.get('f0_embed_type', 'continuous')
        if self.f0_embed_type == 'discrete':
            self.pitch_embed = Embedding(300, hidden_size, 0)
        else:
            self.pitch_embed = Linear(1, hidden_size)

        # ---- Variance embeddings (breathiness, voicing, tension, energy) ----
        self.use_variance_embeds = hparams.get('use_variance_embeds', False)
        if self.use_variance_embeds:
            self.variance_embed_list = ['breathiness', 'voicing', 'tension']
            self.variance_embeds = nn.ModuleDict({
                name: Linear(1, hidden_size)
                for name in self.variance_embed_list
            })
        else:
            self.variance_embed_list = []
            self.variance_embeds = nn.ModuleDict()

        # ---- Key-shift (gender) / speed embeddings ----
        self.key_shift_embed = Linear(1, hidden_size)
        self.speed_embed = Linear(1, hidden_size)
