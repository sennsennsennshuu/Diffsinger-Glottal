"""Print total training data duration from binary .info.npz file.

Usage:
  python utils/data_hours.py --binary data_dir
  python utils/data_hours.py --exp aco_test4s

This helps determine appropriate loss weights and LR scheduling for
SFC breathy training. Output includes recommended scheduler parameters.
"""
import argparse
import pathlib

import numpy as np

from utils.hparams import hparams, set_hparams


def compute_data_hours(info_path: pathlib.Path) -> dict:
    """Read .info.npz and compute total audio duration in hours."""
    if not info_path.exists():
        raise FileNotFoundError(f"Info file not found: {info_path}")

    info = dict(np.load(info_path, allow_pickle=True))
    lengths = info.get("lengths")
    if lengths is None:
        raise KeyError(f"'lengths' not found in {info_path}. Keys: {list(info.keys())}")

    total_frames = int(lengths.sum())
    hop_size = hparams.get("hop_size", 512)
    sample_rate = hparams.get("audio_sample_rate", 44100)

    total_seconds = total_frames * hop_size / sample_rate
    total_minutes = total_seconds / 60
    total_hours = total_seconds / 3600

    return {
        "total_frames": total_frames,
        "total_hours": round(total_hours, 2),
        "total_minutes": round(total_minutes, 1),
        "total_seconds": round(total_seconds, 1),
        "num_samples": len(lengths),
        "hop_size": hop_size,
        "sample_rate": sample_rate,
    }


def recommend_params(data_hours: float, max_steps: int) -> dict:
    """Recommend LR scheduler and SFC warmup parameters based on data size.

    Smaller datasets need slower LR decay and longer phase 1 (spec-only)
    to prevent SFC losses from overwhelming spec optimization.
    """
    if data_hours < 5:
        hr_category = "small (<5h)"
        lr_decay = max_steps // 6
        gamma = 0.85
    elif data_hours < 20:
        hr_category = "medium (5-20h)"
        lr_decay = max_steps // 7
        gamma = 0.82
    else:
        hr_category = "large (>20h)"
        lr_decay = max_steps // 8
        gamma = 0.80

    return {
        "data_category": hr_category,
        "recommended_step_size": lr_decay,
        "recommended_gamma": gamma,
        "recommended_p1_end_pct": 0.25,
        "recommended_p2_end_pct": 0.50,
        "recommended_p3_end_pct": 0.75,
        "recommended_sfc_interval": max(5000, lr_decay // 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Print training data duration from binary dataset"
    )
    parser.add_argument("--binary", type=pathlib.Path, help="Path to binary data directory")
    parser.add_argument("--exp", type=str, help="Experiment name (reads from hparams)")
    args = parser.parse_args()

    if args.exp:
        set_hparams()
        work_dir = pathlib.Path(hparams["work_dir"])
        # Try to find binary_dir from hparams
        binary_dir = hparams.get("binary_data_dir")
        if not binary_dir:
            # Fallback: look in data/
            binary_dir = pathlib.Path("data") / args.exp.replace("aco_", "").replace("var_", "")
        binary_dir = pathlib.Path(binary_dir)
    elif args.binary:
        set_hparams()
        binary_dir = args.binary
    else:
        parser.error("Either --binary or --exp is required")

    # Find train.info.npz
    info_path = binary_dir / "binary" / "acoustic" / "train.info.npz"
    if not info_path.exists():
        info_path = binary_dir / "train.info.npz"
    if not info_path.exists():
        # Try binary/acoustic subdirectory
        candidates = list(binary_dir.glob("**/train.info.npz"))
        if candidates:
            info_path = candidates[0]
        else:
            print(f"ERROR: Cannot find train.info.npz under {binary_dir}")
            print(f"  Checked: {info_path}")
            return 1

    stats = compute_data_hours(info_path)
    rec = recommend_params(stats["total_hours"], hparams.get("trainer_max_steps", 120000) if args.exp else 120000)

    print("=" * 60)
    print(f"  Binary dir:  {binary_dir}")
    print(f"  Info file:   {info_path}")
    print(f"  Total frames: {stats['total_frames']:,}")
    print(f"  Data duration: {stats['total_hours']}h ({stats['total_minutes']}min)")
    print(f"  Num samples:   {stats['num_samples']}")
    print(f"  Hop size:      {stats['hop_size']} samples")
    print(f"  Sample rate:   {stats['sample_rate']} Hz")
    print("=" * 60)
    print(f"  Data category:  {rec['data_category']}")
    print(f"  Recommended LR step_size: {rec['recommended_step_size']}")
    print(f"  Recommended LR gamma:     {rec['recommended_gamma']}")
    print(f"  Phase pct (P1/P2/P3):    {rec['recommended_p1_end_pct']:.2f} / "
          f"{rec['recommended_p2_end_pct']:.2f} / {rec['recommended_p3_end_pct']:.2f}")
    print(f"  SFC unfreeze interval:    {rec['recommended_sfc_interval']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
