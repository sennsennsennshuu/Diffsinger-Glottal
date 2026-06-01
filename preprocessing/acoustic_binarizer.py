import collections
import copy
import csv
import random
from dataclasses import dataclass

import dask
import numpy

from lib.config.schema import DataSourceConfig
from lib.functional import resize_curve
from .binarizer_base import MetadataItem, BaseBinarizer, DataSample

ACOUSTIC_ITEM_ATTRIBUTES = [
    "spk_id",
    "languages",
    "tokens",
    "ph_dur",
    "mel",
    "f0",
    "energy",
    "breathiness",
    "voicing",
    "tension",
    "aperiodic_mel",
    "key_shift",
    "speed",
]


@dataclass
class AcousticMetadataItem(MetadataItem):
    pass


class AcousticBinarizer(BaseBinarizer):
    __data_attrs__ = ACOUSTIC_ITEM_ATTRIBUTES
    __augmentation__ = True

    def load_metadata(self, data_source_config: DataSourceConfig):
        metadata_dict = collections.OrderedDict()
        raw_data_dir = data_source_config.raw_data_dir_resolved
        with open(raw_data_dir / "transcriptions.csv", "r", encoding="utf8") as f:
            transcriptions = list(csv.DictReader(f))
        for transcription in transcriptions:
            item_name = transcription["name"]
            spk_name = data_source_config.speaker
            spk_id = data_source_config.spk_id
            succeeded, parse_results = self.parse_language_phoneme_sequences(
                transcription, language=data_source_config.language
            )
            if not succeeded:
                raise ValueError(
                    parse_results.format(raw_data_dir.as_posix(), item_name)
                )
            ph_text, lang_seq, ph_seq, ph_dur = parse_results
            wav_fn = raw_data_dir / "wavs" / f"{item_name}.wav"
            if not wav_fn.exists():
                raise ValueError(
                    f"Waveform file missing in raw dataset \'{raw_data_dir.as_posix()}\':\n"
                    f"item {item_name}, wav file \'{wav_fn.as_posix()}\'."
                )
            metadata_dict[item_name] = AcousticMetadataItem(
                item_name=item_name,
                estimated_duration=sum(ph_dur),
                spk_name=spk_name,
                spk_id=spk_id,
                ph_text=ph_text,
                lang_seq=lang_seq,
                ph_seq=ph_seq,
                ph_dur=ph_dur,
                wav_fn=wav_fn,
            )
        return metadata_dict

    def process_item(self, item: AcousticMetadataItem, augmentation=False) -> list[DataSample]:
        waveform = self.load_waveform(item.wav_fn)
        mel, length = self.get_mel(waveform)
        ph_dur_sec = numpy.array(item.ph_dur, dtype=numpy.float32)
        ph_dur = self.sec_dur_to_frame_dur(ph_dur_sec, length)
        f0, uv = self.get_f0(waveform, length)
        energy = self.get_energy(waveform, length, smooth_fn_name="energy")
        harmonic, noise = self.harmonic_noise_separation(waveform, f0)
        breathiness = self.get_energy(noise, length, smooth_fn_name="breathiness")
        voicing = self.get_energy(harmonic, length, smooth_fn_name="voicing")
        base_harmonic = self.get_kth_harmonic(harmonic, f0, k=0)
        tension = self.get_tension(harmonic, base_harmonic, length)
        aperiodic_mel = None
        if self.config.features.aperiodic_mel.enabled:
            aperiodic_mel, _ = self.get_mel(noise)

        data = {
            "spk_id": numpy.array(item.spk_id, dtype=numpy.int64),
            "languages": numpy.array(item.lang_seq, dtype=numpy.int64),
            "tokens": numpy.array(item.ph_seq, dtype=numpy.int64),
            "ph_dur": ph_dur,
            "mel": mel,
            "f0": f0,
            "key_shift": numpy.array(0., dtype=numpy.float32),
            "speed": numpy.array(1., dtype=numpy.float32),
        }
        if self.config.features.energy.enabled:
            data["energy"] = energy
        if self.config.features.breathiness.enabled:
            data["breathiness"] = breathiness
        if self.config.features.voicing.enabled:
            data["voicing"] = voicing
        if self.config.features.tension.enabled:
            data["tension"] = tension
        if aperiodic_mel is not None:
            data["aperiodic_mel"] = aperiodic_mel
        length, uv, data = dask.compute(length, uv, data)

        if uv.all():
            error = "empty gt f0"
        else:
            error = None
        sample = DataSample(
            name=item.item_name,
            spk_name=item.spk_name,
            spk_id=item.spk_id,
            ph_text=item.ph_text,
            length=length,
            augmented=False,
            data=data,
            error=error
        )

        samples = [sample]
        if error or not augmentation:
            return samples

        augmentation_params: list[tuple[float, float]] = []  # (shift, speed)
        shift_scale = self.config.augmentation.random_pitch_shifting.scale
        if self.config.augmentation.random_pitch_shifting.enabled:
            num_shifts = int(shift_scale) + (numpy.random.rand() < shift_scale % 1)
        else:
            num_shifts = 0
        key_shift_min, key_shift_max = self.config.augmentation.random_pitch_shifting.range
        for _ in range(num_shifts):
            rand = random.uniform(-1, 1)
            if rand < 0:
                shift = key_shift_min * abs(rand)
            else:
                shift = key_shift_max * rand
            augmentation_params.append((shift, 1))
        stretch_scale = self.config.augmentation.random_time_stretching.scale
        if self.config.augmentation.random_time_stretching.enabled:
            randoms = numpy.random.rand(1 + num_shifts)
            stretch_ids = list(numpy.where(randoms < stretch_scale % 1)[0])
            if stretch_scale > 1:
                stretch_ids.extend([0] * int(stretch_scale))
                stretch_ids.sort()
        else:
            stretch_ids = []
        speed_min, speed_max = self.config.augmentation.random_time_stretching.range
        for i in stretch_ids:
            # Uniform distribution in log domain
            speed = speed_min * (speed_max / speed_min) ** random.random()
            if i == 0:
                augmentation_params.append((0, speed))
            else:
                shift, _ = augmentation_params[i - 1]
                augmentation_params[i - 1] = (shift, speed)

        if not augmentation_params:
            return samples

        length_transforms = []
        data_transforms = []
        for shift, speed in augmentation_params:
            mel_transform, length_transform = self.get_mel(waveform, shift=shift, speed=speed)
            ph_dur_transform = self.sec_dur_to_frame_dur(ph_dur_sec / speed, length_transform)
            f0_transform = dask.delayed(resize_curve)(data["f0"] * 2 ** (shift / 12), length_transform)
            v_transform = {
                v_name: dask.delayed(resize_curve)(data[v_name], length_transform)
                for v_name in self.config.features.enabled_variance_names
            }
            data_transform = sample.data.copy()
            data_transform["ph_dur"] = ph_dur_transform
            data_transform["mel"] = mel_transform
            data_transform["f0"] = f0_transform
            data_transform["key_shift"] = numpy.array(shift, dtype=numpy.float32)
            data_transform["speed"] = (
                dask.delayed(
                    lambda x: numpy.array(sample.length / x, dtype=numpy.float32)  # real speed
                )(length_transform)
            )
            for v_name in self.config.features.enabled_variance_names:
                data_transform[v_name] = v_transform[v_name]
            length_transforms.append(length_transform)
            data_transforms.append(data_transform)

        length_transforms, data_transforms = dask.compute(length_transforms, data_transforms)
        for i in range(len(augmentation_params)):
            sample_transform = copy.copy(sample)
            sample_transform.length = length_transforms[i]
            sample_transform.data = data_transforms[i]
            sample_transform.augmented = True
            samples.append(sample_transform)

        return samples
