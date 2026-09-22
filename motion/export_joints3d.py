"""Export SMPL-X joint positions from a profile's motion.npz for the eval page's skeleton
column (docs/protocol.md §5). The eval page draws the extracted motion as a bare stick
figure between the source video and the retargeted avatar, so a low likeness score can be
blamed on the right stage: video vs skeleton judges the extraction, skeleton vs avatar
judges the retarget/presentation.

    .venv/bin/python export_joints3d.py <motion.npz> <out.json>

Output: {"fps", "frames", "names": [55 SMPL-X joint names], "joints": flat
[frames * 55 * 3] xyz}. The npz's world frame is already yaw-fixed to the three.js stage
(extract.py's WORLD_YAW_FIX) and SMPL-X is y-up, so the positions render as-is; only the
floor is normalized (clip-wide lowest toe sits at y=0), matching how the player grounds
the avatar. Root translation is dropped — the retargeted track is pure rotation with
locked hips translation, so an in-place skeleton is what compares honestly with the
avatar column. Pure CPU; the model is small.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "pantomatrix"))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from metrics import joint_positions  # noqa: E402

# SMPL-X joint order, the first 55 of the model's kinematic tree (the slice
# metrics.joint_positions returns). Indices 22-24 are jaw/eyes, 25-54 the fingers.
JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw",
    "left_eye", "right_eye",
    "left_index1", "left_index2", "left_index3",
    "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
    "right_index1", "right_index2", "right_index3",
    "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3",
    "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3",
]

LEFT_TOE, RIGHT_TOE = 10, 11


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    npz, out = Path(sys.argv[1]), Path(sys.argv[2])
    data = np.load(npz, allow_pickle=True)
    poses, betas = data["poses"], data["betas"]
    fps = float(data["mocap_frame_rate"]) if "mocap_frame_rate" in data.files else 30.0

    joints = joint_positions(poses, betas, torch.device("cpu"))  # (frames, 55, 3), metres
    floor = joints[:, [LEFT_TOE, RIGHT_TOE], 1].min()
    joints[:, :, 1] -= floor

    payload = {
        "fps": fps,
        "frames": int(joints.shape[0]),
        "names": JOINT_NAMES,
        "joints": [round(v, 4) for v in joints.reshape(-1).tolist()],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"{out.name}: {payload['frames']} frames x 55 joints at {fps:g} fps "
          f"({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
