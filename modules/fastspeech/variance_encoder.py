"""FastSpeech2Variance stub for ONNX deployment compatibility.

This stub recreates the legacy FastSpeech2Variance class signature required by
``deployment/modules/fastspeech2.py`` (FastSpeech2VarianceONNX superclass).
"""

import torch
import torch.nn as nn

from modules.commons.common_layers import NormalInitEmbedding as Embedding
from modules.commons.common_layers import XavierUniformInitLinear as Linear
from modules.commons.tts_modules import FastSpeech2Encoder, DurationPredictor
from utils.hparams import hparams


class FastSpeech2Variance(nn.Module):
    """Legacy FastSpeech2 variance encoder.

    Mirrors the original v2 class attributes that the ONNX subclass
    (FastSpeech2VarianceONNX) expects.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        hidden_size = hparams.get('hidden_size', 256)

        # ---- Token / word-level / phoneme-level embeddings ----
        self.txt_embed = Embedding(vocab_size, hidden_size, padding_idx=0)
        self.onset_embed = Linear(1, hidden_size)
        self.word_dur_embed = Linear(1, hidden_size)
        self.ph_dur_embed = Linear(1, hidden_size)
        self.midi_embed = Linear(1, hidden_size)

        self.use_lang_id = hparams.get('use_lang_id', False)
        if self.use_lang_id:
            num_lang = hparams.get('num_lang', 3) + 1
            self.lang_embed = Embedding(num_lang, hidden_size, padding_idx=0)

        # ---- Encoder backbone ----
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

        # ---- Duration predictor ----
        self.predict_dur = True
        self.dur_predictor = DurationPredictor(
            hidden_size,
            n_chans=hparams.get('predictor_hidden', 256),
            kernel_size=hparams.get('dur_predictor_kernel', 5),
        )
