import abc
import json
import pathlib
from typing import Literal

import torch

from lib.config.schema import VocoderConfig
from .nsf_hifigan.env import AttrDict
from .nsf_hifigan.models import Generator


class Vocoder(abc.ABC):
    __vocoder_type__: None
    __registry__: dict[str, type] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        subclass_vocoder_type = cls.__vocoder_type__
        if subclass_vocoder_type is None:
            raise ValueError(f"Vocoder subclass {cls.__name__} must define __vocoder_type__")
        if subclass_vocoder_type in cls.__registry__:
            raise ValueError(
                f"Vocoder subclass {cls.__name__} with __vocoder_type__ '{subclass_vocoder_type}' already exists."
            )
        cls.__registry__[subclass_vocoder_type] = cls

    def __new__(cls, config: VocoderConfig):
        vocoder_type = config.vocoder_type
        if vocoder_type not in cls.__registry__:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")
        return super().__new__(cls.__registry__[vocoder_type])

    def __init__(self, config: VocoderConfig):
        super().__init__()
        self.setup(pathlib.Path(config.vocoder_path))
        self.validate(config)

    def validate(self, config: VocoderConfig):
        if config.spectrogram.type != "mel":
            raise ValueError(f"Vocoder spec_type mismatch: must be 'mel', got {config.spectrogram.type}")
        if self.sample_rate != config.audio_sample_rate:
            raise ValueError(f"Vocoder sample_rate mismatch: {self.sample_rate} != {config.audio_sample_rate}")
        if self.hop_size != config.hop_size:
            raise ValueError(f"Vocoder hop_size mismatch: {self.hop_size} != {config.hop_size}")
        if self.fft_size != config.fft_size:
            raise ValueError(f"Vocoder fft_size mismatch: {self.fft_size} != {config.fft_size}")
        if self.win_size != config.win_size:
            raise ValueError(f"Vocoder win_size mismatch: {self.win_size} != {config.win_size}")
        if abs(self.fmin - config.spectrogram.fmin) > 1e-6:
            raise ValueError(f"Vocoder fmin mismatch: {self.fmin} != {config.spectrogram.fmin}")
        if abs(self.fmax - config.spectrogram.fmax) > 1e-6:
            raise ValueError(f"Vocoder fmax mismatch: {self.fmax} != {config.spectrogram.fmax}")
        if self.num_bins != config.spectrogram.num_bins:
            raise ValueError(f"Vocoder num_bins mismatch: {self.num_bins} != {config.spectrogram.num_bins}")

    @abc.abstractmethod
    def setup(self, model_path: pathlib.Path):
        pass

    @property
    @abc.abstractmethod
    def spec_type(self) -> Literal["mel"]:
        pass

    @property
    @abc.abstractmethod
    def sample_rate(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def hop_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def fft_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def win_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def fmin(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def fmax(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def num_bins(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        pass

    @abc.abstractmethod
    def to(self, device: torch.device) -> "Vocoder":
        pass

    @abc.abstractmethod
    def run(self, spec: torch.Tensor, f0: torch.Tensor = None) -> torch.Tensor:
        pass


class NSFHiFiGAN(Vocoder):
    __vocoder_type__ = "nsf-hifigan"

    # noinspection PyAttributeOutsideInit
    def setup(self, model_path: pathlib.Path):
        config_file = model_path.with_name("config.json")
        with open(config_file) as f:
            config = json.load(f)
        self.config = AttrDict(config)
        generator = Generator(self.config)
        cp_dict = torch.load(model_path, map_location="cpu")
        generator.load_state_dict(cp_dict["generator"])
        generator.eval()
        generator.remove_weight_norm()
        self.generator = generator

    @property
    def spec_type(self) -> Literal["mel"]:
        return "mel"

    @property
    def sample_rate(self) -> int:
        return self.config.sampling_rate

    @property
    def hop_size(self) -> int:
        return self.config.hop_size

    @property
    def fft_size(self) -> int:
        return self.config.n_fft

    @property
    def win_size(self) -> int:
        return self.config.win_size

    @property
    def fmin(self) -> float:
        return self.config.fmin

    @property
    def fmax(self) -> float:
        return self.config.fmax

    @property
    def num_bins(self) -> int:
        return self.config.num_mels

    @property
    def device(self) -> torch.device:
        return next(self.generator.parameters()).device

    def to(self, device: torch.device) -> "NSFHiFiGAN":
        self.generator.to(device)
        return self

    @torch.no_grad()
    def run(self, spec: torch.Tensor, f0: torch.Tensor = None) -> torch.Tensor:
        return self.generator(spec.transpose(1, 2), f0)


class PCNSFHiFiGAN(NSFHiFiGAN):
    """Pitch-Controllable NSF-HiFiGAN — same architecture, different checkpoint."""
    __vocoder_type__ = "pc-nsf-hifigan"
