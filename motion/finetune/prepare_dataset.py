"""Build the EMAGE fine-tune dataset from the speaker profiles in data/profiles/.

For each profile this:
  1. copies motion.npz -> data/smplxflame_30/<id>.npz with `trans` zeroed (generate.py
     passes zero translation at inference, so training sees the same convention),
  2. copies audio.wav -> data/wave16k/<id>.wav (profiles are already 16k mono),
  3. computes footcontact/<id>.npy: SMPL-X FK on the (zero-trans) poses, per-frame speed
     of joints 7/8/10/11 (ankles + toes) < 0.01 m/frame, same as
     pantomatrix/datasets/foot_contact.py. The directory names are load-bearing:
     BEAT2DatasetEamgeFootContact resolves the npy by string-replacing
     "smplxflame_30" -> "footcontact" in the motion path,
  4. runs the LIPM balance audit (motion/balance.py) on the ORIGINAL world-space track
     and drops every training window that overlaps a fall-risk run — the TED profile
     still contains tip-over segments that correct_balance.py only fixed on the avatar
     track,
  5. emits data/profiles_s20_l64.json: 64-frame windows, stride 20, last ~15s of each
     profile held out as mode "test".

    .venv/bin/python finetune/prepare_dataset.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

MOTION = Path(__file__).resolve().parent.parent          # motion/
ROOT = MOTION.parent                                     # repo root
sys.path.insert(0, str(MOTION))

import metrics  # noqa: E402
from balance import balance_stats  # noqa: E402

PROFILES = ROOT / "data" / "profiles"
DATA = Path(__file__).resolve().parent / "data"
SMPLX_DIR = DATA / "smplxflame_30"
WAV_DIR = DATA / "wave16k"
FC_DIR = DATA / "footcontact"
META = DATA / "profiles_s20_l64.json"

FPS = 30
WIN = 64
STRIDE = 20
HOLDOUT_S = 15
# SMPL-X joint index -> the Mixamo-style bone names balance.py expects.
JOINT_MAP = {
    "Hips": 0, "LeftUpLeg": 1, "RightUpLeg": 2, "Spine": 3,
    "LeftLeg": 4, "RightLeg": 5, "Spine1": 6,
    "LeftFoot": 7, "RightFoot": 8, "Spine2": 9,
    "LeftToeBase": 10, "RightToeBase": 11, "Neck": 12,
    "Head": 15, "LeftArm": 16, "RightArm": 17,
    "LeftForeArm": 18, "RightForeArm": 19, "LeftHand": 20, "RightHand": 21,
}
FOOT_JOINTS = (7, 8, 10, 11)


def foot_contact(poses: np.ndarray, trans: np.ndarray, device: torch.device) -> np.ndarray:
    """(T,4) float contact flags, ported from pantomatrix/datasets/foot_contact.py."""
    import smplx

    model = smplx.create(
        str(MOTION / "emage_evaltools" / "smplx_models"), model_type="smplx",
        gender="NEUTRAL_2020", use_face_contour=False, num_betas=300,
        num_expression_coeffs=100, ext="npz", use_pca=False,
    ).eval().to(device)
    n = poses.shape[0]
    p = torch.from_numpy(poses).float().to(device)
    t = torch.from_numpy(trans).float().to(device)
    zeros_b = torch.zeros(n, 300, device=device)
    zeros_e = torch.zeros(n, 100, device=device)
    chunks = []
    for s in range(0, n, 128):
        e = min(s + 128, n)
        with torch.no_grad():
            joints = model(
                betas=zeros_b[s:e], transl=t[s:e], expression=zeros_e[s:e],
                jaw_pose=p[s:e, 66:69], global_orient=p[s:e, :3],
                body_pose=p[s:e, 3:66], left_hand_pose=p[s:e, 75:120],
                right_hand_pose=p[s:e, 120:165], leye_pose=p[s:e, 69:72],
                reye_pose=p[s:e, 72:75], return_joints=True,
            )["joints"][:, FOOT_JOINTS, :].reshape(e - s, 4, 3).cpu()
        chunks.append(joints)
    joints = torch.cat(chunks, dim=0).permute(1, 0, 2)          # 4, T, 3
    speed = torch.zeros(4, n)
    speed[:, :-1] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
    return (speed < 0.01).numpy().astype(float).T               # T, 4


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SMPLX_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    FC_DIR.mkdir(parents=True, exist_ok=True)

    meta = []
    report = {}
    for profile in sorted(p for p in PROFILES.iterdir() if (p / "motion.npz").exists()):
        pid = profile.name
        src = np.load(profile / "motion.npz", allow_pickle=True)
        poses = src["poses"].astype(np.float32)
        trans_orig = src["trans"].astype(np.float32)
        expressions = src["expressions"].astype(np.float32)
        n = poses.shape[0]

        out_npz = SMPLX_DIR / f"{pid}.npz"
        np.savez(
            out_npz,
            betas=src["betas"], poses=poses, expressions=expressions,
            trans=np.zeros_like(trans_orig), model="smplx2020", gender="neutral",
            mocap_frame_rate=30,
        )
        shutil.copy(profile / "audio.wav", WAV_DIR / f"{pid}.wav")

        fc = foot_contact(poses, np.zeros_like(trans_orig), device)
        np.save(FC_DIR / f"{pid}.npy", fc)

        # Balance audit on the original world-space track.
        joints = metrics.joint_positions(poses, betas=src["betas"], device=device)
        pos = {name: joints[:, idx, :] + trans_orig for name, idx in JOINT_MAP.items()}
        stats = balance_stats(pos, FPS)
        runs = stats["runs"]

        def overlaps_run(s: int, e: int) -> bool:
            return any(s < r["end"] and e > r["start"] for r in runs)

        holdout_start = max(0, n - HOLDOUT_S * FPS)
        kept = dropped = 0
        for s in range(0, n - WIN + 1, STRIDE):
            e = s + WIN
            if e <= holdout_start:
                mode = "train"
            elif s >= holdout_start:
                mode = "test"
            else:
                continue  # straddles the split; skip
            if mode == "train" and overlaps_run(s, e):
                dropped += 1
                continue
            kept += 1
            meta.append({
                "video_id": pid,
                "motion_path": str(out_npz),
                "audio_path": str(WAV_DIR / f"{pid}.wav"),
                "mode": mode,
                "start_idx": s,
                "end_idx": e,
            })

        report[pid] = {
            "frames": n,
            "seconds": round(n / FPS, 1),
            "fall_risk_runs": [
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
                for r in runs
            ],
            "pct_frames_cp_outside": round(stats["pct_outside"], 2),
            "excursion_max_m": round(float(stats["excursion"][np.isfinite(stats["excursion"])].max()), 3),
            "windows_kept": kept,
            "windows_dropped_balance": dropped,
        }
        print(f"{pid}: {report[pid]}", flush=True)

    META.write_text(json.dumps(meta, indent=1))
    n_train = sum(1 for m in meta if m["mode"] == "train")
    n_test = sum(1 for m in meta if m["mode"] == "test")
    print(f"\nwrote {META} — {n_train} train / {n_test} test windows")
    (DATA / "prepare_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
