"""Retarget generated SMPL-X motion onto the avatar's Mixamo skeleton.

    .venv/bin/python retarget.py <motion.npz> <avatar.glb> [out.json]

Copying local rotations bone to bone does not work: the two skeletons have different rest
poses, so the same local rotation means different things. What transfers is the *global*
orientation of each joint, re-expressed in the target's rest frame.

SMPL-X pose parameters are rotations away from the canonical zero pose, and in that pose
every joint frame is identity, so a joint's composed global rotation *is* the delta it has
moved through. Applying that same delta to the target gives

    Bglobal_i = Aglobal_i @ N_i
    Blocal_i  = inv(Bglobal_parent(i)) @ Bglobal_i

computed down the target's tree so each bone sees its parent already posed.

`N_i` is the target's neutral orientation, and it is not the bind pose. Using the bind pose
directly was measured wrong: the limb-direction error against the source came out constant
per bone (19.4 degrees mean against 19.5 max on the upper arm), which is the signature of a
rest-pose mismatch rather than a motion error. SMPL-X rests in an A shape with the arms
hanging at roughly 45 degrees; this rig binds in a T shape with the arms straight out, so
treating the two as the same neutral leaves the avatar standing in a T-pose with small
gestures around it. `N_i` therefore carries a per-bone correction that rotates the rig's rest
bone direction onto the source's rest bone direction, so "the model is at rest" means "the
avatar stands the way the model stands at rest".

Bones the source has no opinion about (the twist bones LeftForeArm1/2, Neck1/2, the eyes,
the bone tips) keep their bind rotation and simply inherit their parent's motion.

Root translation is deliberately dropped. EMAGE emits a global trajectory, but a digital
human answering a question should stay where it is, and the player grounds the feet anyway.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# SMPL-X joint index -> Mixamo bone. The body is a direct correspondence; the fingers are
# one to one once SMPL-X's order (index, middle, pinky, ring, thumb) is respected.
BODY = {
    0: "Hips", 1: "LeftUpLeg", 2: "RightUpLeg", 3: "Spine", 4: "LeftLeg", 5: "RightLeg",
    6: "Spine1", 7: "LeftFoot", 8: "RightFoot", 9: "Spine2", 10: "LeftToeBase",
    11: "RightToeBase", 12: "Neck", 13: "LeftShoulder", 14: "RightShoulder", 15: "Head",
    16: "LeftArm", 17: "RightArm", 18: "LeftForeArm", 19: "RightForeArm",
    20: "LeftHand", 21: "RightHand",
}
FINGERS = ["Index", "Middle", "Pinky", "Ring", "Thumb"]

# The child each bone points at, for measuring its rest direction. A bone with several
# children has no direction of its own, so one is named; tips borrow the rig's own end bone.
POINTS_AT = {
    "Hips": "Spine", "Spine": "Spine1", "Spine1": "Spine2", "Spine2": "Neck",
    "Neck": "Head", "Head": "HeadTop_End",
    "LeftShoulder": "LeftArm", "LeftArm": "LeftForeArm", "LeftForeArm": "LeftHand",
    "LeftHand": "LeftHandMiddle1",
    "RightShoulder": "RightArm", "RightArm": "RightForeArm", "RightForeArm": "RightHand",
    "RightHand": "RightHandMiddle1",
    "LeftUpLeg": "LeftLeg", "LeftLeg": "LeftFoot", "LeftFoot": "LeftToeBase",
    "LeftToeBase": "LeftToe_End",
    "RightUpLeg": "RightLeg", "RightLeg": "RightFoot", "RightFoot": "RightToeBase",
    "RightToeBase": "RightToe_End",
}
for _side in ("Left", "Right"):
    for _f in FINGERS:
        for _s in (1, 2, 3):
            POINTS_AT[f"{_side}Hand{_f}{_s}"] = f"{_side}Hand{_f}{_s + 1}"


def align(from_dir: np.ndarray, to_dir: np.ndarray) -> np.ndarray:
    """Minimal rotation taking one unit direction onto another."""
    a = from_dir / (np.linalg.norm(from_dir) + 1e-12)
    b = to_dir / (np.linalg.norm(to_dir) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def joint_map() -> dict[int, str]:
    out = dict(BODY)
    for side, base in (("Left", 25), ("Right", 40)):
        for f, finger in enumerate(FINGERS):
            for seg in range(3):
                out[base + f * 3 + seg] = f"{side}Hand{finger}{seg + 1}"
    return out


def read_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    length = struct.unpack("<I", raw[12:16])[0]
    return json.loads(raw[20:20 + length]), raw[20 + length + 8:]


def trs(node: dict) -> np.ndarray:
    m = np.eye(4)
    if "rotation" in node:
        m[:3, :3] = Rotation.from_quat(node["rotation"]).as_matrix()
    if "scale" in node:
        m[:3, :3] = m[:3, :3] @ np.diag(node["scale"])
    if "translation" in node:
        m[:3, 3] = node["translation"]
    return m


def bind_pose(gltf: dict) -> tuple[dict[str, np.ndarray], dict[str, str | None], list[str]]:
    """Global rest rotation per bone, the parent of each bone, and a top-down bone order."""
    nodes = gltf["nodes"]
    parent: dict[int, int] = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i
    order: list[int] = []
    seen: set[int] = set()

    def visit(i: int) -> None:
        if i in seen:
            return
        p = parent.get(i)
        if p is not None:
            visit(p)
        seen.add(i)
        order.append(i)

    for i in range(len(nodes)):
        visit(i)

    glob: dict[int, np.ndarray] = {}
    for i in order:
        local = trs(nodes[i])
        p = parent.get(i)
        glob[i] = local if p is None else glob[p] @ local

    named = {i: nodes[i].get("name") for i in order if nodes[i].get("name")}
    rest = {name: glob[i][:3, :3].copy() for i, name in named.items()}
    pos = {name: glob[i][:3, 3].copy() for i, name in named.items()}
    parents = {name: nodes[parent[i]].get("name") if i in parent else None for i, name in named.items()}
    return rest, pos, parents, [named[i] for i in order if i in named]


def source_rest_positions(model) -> np.ndarray:
    """SMPL-X joint positions in the zero pose: the shape its rest directions are measured from."""
    import torch

    with torch.no_grad():
        out = model(
            betas=torch.zeros(1, 300), transl=torch.zeros(1, 3),
            global_orient=torch.zeros(1, 3), body_pose=torch.zeros(1, 63),
            jaw_pose=torch.zeros(1, 3), leye_pose=torch.zeros(1, 3), reye_pose=torch.zeros(1, 3),
            left_hand_pose=torch.zeros(1, 45), right_hand_pose=torch.zeros(1, 45),
            expression=torch.zeros(1, 100), return_verts=False,
        )
    return out.joints[0, :55].cpu().numpy()


def neutral_orientations(rest, rest_pos, mapping, src_rest_pos, src_parents) -> dict[str, np.ndarray]:
    """Per-bone neutral: the rig's bind rotation, turned so its rest direction matches the source's."""
    to_joint = {v: k for k, v in mapping.items()}
    out = {}
    for bone, bind in rest.items():
        tip = POINTS_AT.get(bone)
        j = to_joint.get(bone)
        if tip is None or tip not in rest_pos or j is None:
            out[bone] = bind
            continue
        # Only align when the tip is itself a mapped bone, so both skeletons are measuring the
        # same anatomical segment. Falling back to "the source joint's first child" was wrong
        # for the head: its first child in SMPL-X is the jaw, which points forward and down,
        # while HeadTop_End points up, so the correction threw the head back to face the
        # ceiling. Where there is no shared segment the bind orientation is already right.
        tip_joint = to_joint.get(tip)
        if tip_joint is None or tip_joint >= len(src_rest_pos):
            out[bone] = bind
            continue
        d_rig = rest_pos[tip] - rest_pos[bone]
        d_src = src_rest_pos[tip_joint] - src_rest_pos[j]
        if np.linalg.norm(d_rig) < 1e-6 or np.linalg.norm(d_src) < 1e-6:
            out[bone] = bind
            continue
        out[bone] = align(d_rig, d_src) @ bind
    return out


def retarget(poses: np.ndarray, gltf: dict) -> tuple[list[str], np.ndarray]:
    """(bone names, quaternions per frame) shaped (frames, bones, 4) as xyzw."""
    import smplx

    model = smplx.create(
        "emage_evaltools/smplx_models", model_type="smplx", gender="NEUTRAL_2020",
        use_face_contour=False, num_betas=300, num_expression_coeffs=100, ext="npz", use_pca=False,
    )
    src_parents = model.parents.cpu().numpy()
    mapping = joint_map()
    rest, rest_pos, parents, order = bind_pose(gltf)
    src_rest_pos = source_rest_positions(model)
    neutral = neutral_orientations(rest, rest_pos, mapping, src_rest_pos, src_parents)
    mapped = {v: k for k, v in mapping.items() if v in rest}
    missing = [v for v in mapping.values() if v not in rest]
    if missing:
        print(f"  note: {len(missing)} mapped bones absent from the rig: {missing[:4]}")

    frames = poses.shape[0]
    aa = poses.reshape(frames, -1, 3)[:, :len(src_parents)]
    local = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(frames, -1, 3, 3)

    # source global rotations, down the SMPL-X tree
    src_glob = np.zeros_like(local)
    for j, p in enumerate(src_parents):
        src_glob[:, j] = local[:, j] if p < 0 else src_glob[:, p] @ local[:, j]

    bones = [b for b in order if b in rest]
    out_glob: dict[str, np.ndarray] = {}
    out_local: dict[str, np.ndarray] = {}
    for bone in bones:
        parent_name = parents.get(bone)
        pg = out_glob.get(parent_name) if parent_name else None
        if bone in mapped:
            bg = src_glob[:, mapped[bone]] @ neutral[bone][None]
        else:
            # no source opinion: keep the bind local rotation and inherit the parent
            bind_local = rest[bone] if pg is None else np.linalg.inv(rest[parent_name]) @ rest[bone]
            bg = (pg @ bind_local[None]) if pg is not None else np.broadcast_to(neutral.get(bone, rest[bone]), (frames, 3, 3))
        out_glob[bone] = bg
        out_local[bone] = bg if pg is None else np.linalg.inv(pg) @ bg

    keep = [b for b in bones if b in mapped or b in out_local]
    quats = np.stack([Rotation.from_matrix(out_local[b]).as_quat() for b in keep], axis=1)
    return keep, quats


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    npz, glb = Path(sys.argv[1]), Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else npz.with_suffix(".track.json")
    data = np.load(npz, allow_pickle=True)
    poses = data["poses"]
    fps = int(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else 30
    gltf, _ = read_glb(glb)
    bones, quats = retarget(poses, gltf)
    out.write_text(json.dumps({
        "fps": fps,
        "frames": int(quats.shape[0]),
        "bones": bones,
        # xyzw per bone per frame, rounded: 4 decimals is under a tenth of a degree
        "quats": np.round(quats, 4).reshape(-1).tolist(),
        "source": npz.name,
    }))
    print(f"  {quats.shape[0]} frames x {len(bones)} bones -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
