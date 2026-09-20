"""Check a retargeted track against the motion it came from.

    .venv/bin/python verify_retarget.py <motion.npz> <track.json> [--check]

Retargeting is easy to get subtly wrong and hard to see, so it is measured rather than
eyeballed: run forward kinematics on the rig with the retargeted rotations, run SMPL-X on the
source poses, and compare the direction of each limb. If the transfer is right the angles are
zero; a constant non-zero angle per limb means the rest poses were not aligned, and an angle
that varies over time means the motion itself is being mangled.

That distinction caught both bugs this file exists because of. Aligning to the rig's bind pose
gave a constant 15.9 degrees average (SMPL-X rests in an A shape, the rig binds in a T), and
deriving the head's rest direction from SMPL-X's first child joint gave a large constant error
on the neck, because that child is the jaw and it points forward while HeadTop_End points up.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "pantomatrix"))

from retarget import read_glb  # noqa: E402

AVATAR = HERE.parent / "public" / "assets" / "model-clips.glb"
# (rig parent, rig child, smplx parent joint, smplx child joint)
LIMBS = [
    ("LeftArm", "LeftForeArm", 16, 18), ("LeftForeArm", "LeftHand", 18, 20),
    ("RightArm", "RightForeArm", 17, 19), ("RightForeArm", "RightHand", 19, 21),
    ("LeftUpLeg", "LeftLeg", 1, 4), ("RightUpLeg", "RightLeg", 2, 5),
    ("Spine1", "Spine2", 6, 9), ("Neck", "Head", 12, 15),
]
TOLERANCE_DEG = 1.0


def rig_positions(track: dict, gltf: dict, frames: list[int]) -> dict[int, dict[str, np.ndarray]]:
    nodes = gltf["nodes"]
    by = {n.get("name"): i for i, n in enumerate(nodes) if n.get("name")}
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i
    bones = track["bones"]
    index = {b: k for k, b in enumerate(bones)}
    quats = np.asarray(track["quats"], dtype=np.float32).reshape(track["frames"], len(bones), 4)
    out = {}
    for f in frames:
        glob: dict[str, np.ndarray] = {}
        for b in bones:
            i = by[b]
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(quats[f, index[b]]).as_matrix()
            T[:3, 3] = nodes[i].get("translation", [0, 0, 0])
            p = parent.get(i)
            pn = nodes[p].get("name") if p is not None else None
            glob[b] = T if pn not in glob else glob[pn] @ T
        out[f] = {b: m[:3, 3] for b, m in glob.items()}
    return out


def source_positions(poses: np.ndarray, betas: np.ndarray) -> np.ndarray:
    import smplx

    frames = poses.shape[0]
    model = smplx.create(
        str(HERE / "emage_evaltools" / "smplx_models"), model_type="smplx", gender="NEUTRAL_2020",
        use_face_contour=False, num_betas=300, num_expression_coeffs=100, ext="npz", use_pca=False,
    ).eval()
    p = torch.from_numpy(poses).float()
    with torch.no_grad():
        out = model(
            betas=torch.from_numpy(betas[:300]).float().unsqueeze(0).repeat(frames, 1),
            transl=torch.zeros(frames, 3), global_orient=p[:, 0:3], body_pose=p[:, 3:66],
            jaw_pose=p[:, 66:69], leye_pose=p[:, 69:72], reye_pose=p[:, 72:75],
            left_hand_pose=p[:, 75:120], right_hand_pose=p[:, 120:165],
            expression=torch.zeros(frames, 100), return_verts=False,
        )
    return out.joints[:, :55].numpy()


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    check = "--check" in sys.argv
    npz, track_path = Path(sys.argv[1]), Path(sys.argv[2])
    data = np.load(npz, allow_pickle=True)
    track = json.loads(track_path.read_text())
    gltf, _ = read_glb(AVATAR)

    frames = list(range(0, track["frames"], max(1, track["frames"] // 30)))
    rig = rig_positions(track, gltf, frames)
    src = source_positions(data["poses"], data["betas"])

    print(f"{'limb':<26} {'mean':>7} {'max':>7} {'drift':>8}")
    worst = 0.0
    for a, b, ja, jb in LIMBS:
        angles = []
        for f in frames:
            v1 = rig[f][b] - rig[f][a]
            v2 = src[f, jb] - src[f, ja]
            v1 = v1 / (np.linalg.norm(v1) + 1e-9)
            v2 = v2 / (np.linalg.norm(v2) + 1e-9)
            angles.append(np.degrees(np.arccos(np.clip(v1 @ v2, -1, 1))))
        angles = np.asarray(angles)
        # a constant offset is a rest-pose bug; a varying one is a motion bug
        print(f"{a + '->' + b:<26} {angles.mean():6.2f}° {angles.max():6.2f}° {angles.std():7.2f}°")
        worst = max(worst, angles.max())

    print(f"\n  worst limb-direction error: {worst:.2f}°")
    if not check:
        return
    if worst > TOLERANCE_DEG:
        print(f"  FAIL  above the {TOLERANCE_DEG}° tolerance")
        raise SystemExit(1)
    print(f"  PASS  within {TOLERANCE_DEG}°")


if __name__ == "__main__":
    main()
