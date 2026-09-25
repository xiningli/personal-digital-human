"""Download a stratified BEAT2 English subset and window it for LoRA mixing.

Anti-overfitting regularizer for the profile-only fine-tune: ~300 clips spread across all
25 English speakers, weighted toward the six 130-clip speakers but with every small
speaker present. Deterministic (seed 222). Downloads per-file from the HF dataset repo
into data/beat2/{smplxflame_30,wave16k,footcontact}/ — the `smplxflame_30` segment is
load-bearing, BEAT2DatasetEamgeFootContact resolves foot contacts by string-replacing it.

Differences from prepare_dataset.py (profiles): trans is still zeroed (same train/infer
convention) but there is NO balance screen — BEAT2 is real mocap, physically real by
construction. balance_stats is still computed per clip and logged for the record.

    ../.venv/bin/python prepare_beat2.py
"""
from __future__ import annotations

import json
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

FINETUNE = Path(__file__).resolve().parent
MOTION = FINETUNE.parent
sys.path.insert(0, str(MOTION))
sys.path.insert(0, str(MOTION / "pantomatrix"))

import metrics  # noqa: E402
from balance import balance_stats  # noqa: E402
from emage_utils.motion_io import beat_format_load  # noqa: E402
from finetune.prepare_dataset import JOINT_MAP, foot_contact  # noqa: E402

HF_RESOLVE = "https://huggingface.co/datasets/H-Liu1997/BEAT2/resolve/main/"
NPZ_LIST = Path("/tmp/beat2_eng_npz.txt")
WAV_LIST = Path("/tmp/beat2_eng_wav.txt")
OUT = FINETUNE / "data" / "beat2"
META = FINETUNE / "data" / "beat2_s20_l64.json"
SEED = 222
FPS, WIN, STRIDE = 30, 64, 20
# speaker -> clips to take; sums to 300
QUOTA = {
    "lawrence": 40, "nidal": 40, "scott": 40, "solomon": 40, "sophie": 40, "wayne": 40,
    "ayana": 12, "miranda": 12, "carla": 10,
    "carlos": 2, "daiki": 2, "goto": 2, "hailing": 2, "itoi": 2, "jorge": 2,
    "katya": 2, "kexin": 2, "kieks": 2, "li": 2, "lu": 2, "luqi": 2,
    "stewart": 2, "tiffnay": 2, "yingqing": 2, "zhao": 2,
}


def speaker_of(path: str) -> str:
    return Path(path).name.split("_")[1]


def pick_clips() -> list[str]:
    npz_paths = [l.strip() for l in NPZ_LIST.read_text().splitlines() if l.strip()]
    wavs = {Path(l.strip()).stem for l in WAV_LIST.read_text().splitlines() if l.strip()}
    by_speaker: dict[str, list[str]] = {}
    for p in npz_paths:
        if Path(p).stem in wavs:  # skip clips whose wav is missing
            by_speaker.setdefault(speaker_of(p), []).append(p)
    rng = random.Random(SEED)
    chosen = []
    for speaker, n in QUOTA.items():
        pool = sorted(by_speaker[speaker])
        chosen.extend(rng.sample(pool, n))
    return sorted(chosen)


def download(url_path: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists():
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(HF_RESOLVE + url_path, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
            return True
        except Exception as e:
            print(f"  retry {attempt + 1}/{retries} {url_path}: {e}", flush=True)
            time.sleep(2 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    return False


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for d in ("smplxflame_30", "wave16k", "footcontact"):
        (OUT / d).mkdir(parents=True, exist_ok=True)

    clips = pick_clips()
    print(f"{len(clips)} clips across {len(QUOTA)} speakers", flush=True)

    meta = []
    report = {}
    skipped = []
    for i, rel in enumerate(clips):
        stem = Path(rel).stem
        npz_dest = OUT / "smplxflame_30" / f"{stem}.npz"
        wav_dest = OUT / "wave16k" / f"{stem}.wav"
        wav_rel = rel.replace("smplxflame_30", "wave16k").replace(".npz", ".wav")
        if not download(rel, npz_dest) or not download(wav_rel, wav_dest):
            skipped.append(stem)
            continue

        try:
            data = beat_format_load(str(npz_dest))
        except Exception as e:
            print(f"  corrupt npz {stem}: {e}", flush=True)
            npz_dest.unlink(missing_ok=True)
            skipped.append(stem)
            continue
        poses = np.asarray(data["poses"], dtype=np.float32)
        expressions = np.asarray(data["expressions"], dtype=np.float32)
        trans_orig = np.asarray(data["trans"], dtype=np.float32)
        n = poses.shape[0]
        if n < WIN:
            skipped.append(stem)
            continue

        # zero trans like the profile data, keep everything else as downloaded
        np.savez(npz_dest, betas=data["betas"], poses=poses, expressions=expressions,
                 trans=np.zeros_like(trans_orig), model="smplx2020", gender="neutral",
                 mocap_frame_rate=30)

        fc_path = OUT / "footcontact" / f"{stem}.npy"
        if not fc_path.exists():
            np.save(fc_path, foot_contact(poses, np.zeros_like(trans_orig), device))

        # balance stats for the record only (real mocap — never screened)
        joints = np.concatenate([
            metrics.joint_positions(poses[s:s + 256], betas=np.asarray(data["betas"], dtype=np.float32), device=device)
            for s in range(0, n, 256)
        ])
        pos = {name: joints[:, idx, :] + trans_orig for name, idx in JOINT_MAP.items()}
        stats = balance_stats(pos, FPS)
        exc = stats["excursion"]
        report[stem] = {
            "speaker": speaker_of(rel), "frames": n,
            "fall_risk_runs": len(stats["runs"]),
            "pct_frames_cp_outside": round(stats["pct_outside"], 2),
            "excursion_max_m": round(float(exc[np.isfinite(exc)].max()), 3) if np.isfinite(exc).any() else None,
        }

        for s in range(0, n - WIN + 1, STRIDE):
            meta.append({
                "video_id": stem,
                "motion_path": str(npz_dest),
                "audio_path": str(wav_dest),
                "mode": "train",
                "start_idx": s,
                "end_idx": s + WIN,
            })
        if (i + 1) % 25 == 0:
            print(f"[{i + 1}/{len(clips)}] windows so far: {len(meta)}", flush=True)

    META.write_text(json.dumps(meta, indent=1))
    (OUT / "prepare_report.json").write_text(json.dumps(report, indent=1))
    runs = sum(r["fall_risk_runs"] for r in report.values())
    print(f"\nwrote {META} — {len(meta)} train windows from {len(report)} clips, {len(skipped)} skipped")
    if skipped:
        print(f"skipped: {skipped}")
    print(f"balance record: {runs} fall-risk runs across all BEAT2 clips (not screened)")


if __name__ == "__main__":
    main()
