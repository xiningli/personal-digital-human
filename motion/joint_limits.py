"""Anatomical joint-limit audit: does any elbow or knee bend backwards or sideways?

    .venv/bin/python joint_limits.py <motion.npz | track.json | clip:<name>> [...] [--glb model-clips.glb]

check_track.py and physics_audit.py check contact, balance and speed; none of them asks
whether a pose is one a human arm or leg can take. This does, for the two hinge joints a
viewer reads instantly (elbow, knee) and, on SMPL-X input, the wrist.

**Method (rig-independent, purely geometric).** For a hinge joint with proximal bone u
(upper arm / thigh) and distal bone f (forearm / shank), n = u x f is perpendicular to the
plane they bend in. Expressed in the proximal bone's own frame, a real elbow or knee bends
about one fixed axis h, so n points along +h when it flexes and along -h when it
hyperextends. Per frame:

- flexion  = signed angle from straight, positive when n . h > 0 (degrees);
- off-hinge = angle between n and the h axis, i.e. how far the joint bends sideways
  (meaningful only when bent more than BEND_MIN degrees).

h comes from somewhere independent of the clip under test: on SMPL-X the rest skeleton
(left elbow flexes about -y of the humerus frame, right about +y, knees about +x); on the
MetaPerson rig it is the bend-weighted mean direction of n over the rig's own Mixamo
clips (studio mocap), printed with the share of their frames that agree.

**Limits** (normal adult range of motion, AAOS / Soucie et al. 2011, loosened a little so
a flag means "a viewer would see it"):

- elbow: hyperextension beyond 10 deg, flexion beyond 150 deg, off-hinge beyond 40 deg;
- knee:  hyperextension beyond 15 deg, flexion beyond 145 deg, off-hinge beyond 15 deg;

The off-hinge limits are set from studio mocap, not anatomy tables: SMPL-X (like most rigs)
lets the elbow joint carry part of the upper arm's axial roll, so even BEAT2 mocap shows the
forearm leaving the ideal hinge plane by up to ~37 deg (p99 23-35 deg over 8 clips). A real
sideways break shows up far beyond that. Locked knees in BEAT2 reach 12 deg of
hyperextension, hence 15.
- wrist (SMPL-X only, swing about the forearm frame): flexion/extension beyond 90 deg,
  radial/ulnar deviation beyond 40 deg.

Exit code 1 if any input exceeds a limit on more than FAIL_SHARE of its frames.

Known blind spot: knee off-hinge on the MetaPerson rig is reported but does not fail. The rig's
Mixamo clips barely bend their knees, so its knee axis is poorly calibrated: SMPL-X's rest
pose retargeted onto the rig already reads 7-13 deg "sideways" while the rendered legs are
straight, and ground_track.py solves the legs as exact hinges anyway (knee over the toes).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BEND_MIN = 15.0
FAIL_SHARE = 0.005
LIMITS = {
    "elbow": {"hyper": 10.0, "flex": 150.0, "side": 40.0},
    "knee": {"hyper": 15.0, "flex": 145.0, "side": 15.0},
}
WRIST = {"flex": 90.0, "dev": 40.0}

# SMPL-X body joint indices: (proximal joint, hinge joint, distal joint)
SMPLX_HINGES = {
    "L elbow": (16, 18, 20), "R elbow": (17, 19, 21),
    "L knee": (1, 4, 7), "R knee": (2, 5, 8),
}
RIG_HINGES = {
    "L elbow": ("LeftArm", "LeftForeArm", "LeftHand"), "R elbow": ("RightArm", "RightForeArm", "RightHand"),
    "L knee": ("LeftUpLeg", "LeftLeg", "LeftFoot"), "R knee": ("RightUpLeg", "RightLeg", "RightFoot"),
}


def distal_local(p0, p1, p2, frame_rot):
    """Unit direction of the distal bone in the proximal bone's frame, (F,3)."""
    f = p2 - p1
    f = np.einsum("fji,fj->fi", frame_rot, f)                       # R^T f
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def fit_axis(d: np.ndarray, d0: np.ndarray) -> np.ndarray:
    """Hinge axis up to sign: principal direction of the rotation axes d0 x d, weighted by angle."""
    a = np.cross(d0[None], d)
    big = np.linalg.norm(a, axis=1) > np.sin(np.radians(BEND_MIN))
    _, _, vt = np.linalg.svd(a[big], full_matrices=False)
    return vt[0]


def hinge_angles(d, d0, h, flex_sign):
    """d: (F,3) distal direction in the proximal frame; d0: its rest direction; h: hinge axis.
    flexion = signed rotation of d about h away from d0 (positive = anatomical flexion);
    off-hinge = how far d has left the plane the hinge sweeps, relative to rest."""
    h = h / np.linalg.norm(h)
    def plane(v):
        w = v - np.outer(v @ h, h) if v.ndim == 2 else v - (v @ h) * h
        return w / np.linalg.norm(w, axis=-1, keepdims=True)
    pd, p0 = plane(d), plane(d0)
    ang = np.degrees(np.arctan2(np.cross(p0[None], pd) @ h, pd @ p0))
    flex = flex_sign * ang
    side = np.abs(np.degrees(np.arcsin(np.clip(d @ h, -1, 1)) - np.arcsin(np.clip(d0 @ h, -1, 1))))
    return flex, side


def flexion_sign(h, d0, rest_rot, forward, toward_front: bool) -> float:
    """+1 if rotating the rest distal bone about +h moves it toward the body's front (elbow) or
    back (knee): the anatomical direction of flexion."""
    step = Rotation.from_rotvec(np.radians(5) * h / np.linalg.norm(h)).apply(d0) - d0
    s = float(np.sign((rest_rot @ step) @ forward))
    return s if toward_front else -s


# ---------------------------------------------------------------------- SMPL-X input
_SMPLX = None


def smplx_fk(poses: np.ndarray, betas: np.ndarray):
    """Joint positions (F,55,3), world (F,55,3,3) and local (F,55,3,3) rotations."""
    global _SMPLX
    import smplx
    import torch

    if _SMPLX is None:
        _SMPLX = smplx.create(
            str(HERE / "emage_evaltools" / "smplx_models"), model_type="smplx", gender="NEUTRAL_2020",
            use_face_contour=False, num_betas=300, num_expression_coeffs=100, ext="npz", use_pca=False,
        ).eval()
    model = _SMPLX
    n = poses.shape[0]
    p = torch.from_numpy(poses).float()
    with torch.no_grad():
        out = model(betas=torch.from_numpy(betas[:300]).float().unsqueeze(0).repeat(n, 1),
                    transl=torch.zeros(n, 3), global_orient=p[:, 0:3], body_pose=p[:, 3:66],
                    jaw_pose=p[:, 66:69], leye_pose=p[:, 69:72], reye_pose=p[:, 72:75],
                    left_hand_pose=p[:, 75:120], right_hand_pose=p[:, 120:165],
                    expression=torch.zeros(n, 100), return_verts=False)
    joints = out.joints[:, :55].numpy().astype(np.float64)
    parents = model.parents.numpy()
    local = Rotation.from_rotvec(poses[:, :165].reshape(n * 55, 3)).as_matrix().reshape(n, 55, 3, 3)
    world = np.zeros_like(local)
    for j in range(55):
        world[:, j] = local[:, j] if parents[j] < 0 else world[:, parents[j]] @ local[:, j]
    return joints, world, local


def wrist_rom(local: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Swing of the wrist about the forearm frame, bone axis +-x: rotation about z is
    flexion/extension (palm-down T-pose: hand up/down), about y radial/ulnar deviation."""
    j = 20 if side == "L" else 21
    q = Rotation.from_matrix(local[:, j]).as_quat()                    # xyzw
    tw = np.zeros_like(q)
    tw[:, 0], tw[:, 3] = q[:, 0], q[:, 3]
    tw /= np.linalg.norm(tw, axis=1, keepdims=True)
    swing = (Rotation.from_quat(q) * Rotation.from_quat(tw).inv()).as_rotvec()
    return np.degrees(swing[:, 2]), np.degrees(swing[:, 1])


def smplx_load(path: Path):
    d = np.load(path, allow_pickle=True)
    poses = d["poses"].astype(np.float64)
    joints, world, local = smplx_fk(poses, d["betas"])
    dirs = {k: distal_local(joints[:, a], joints[:, b], joints[:, c], world[:, a])
            for k, (a, b, c) in SMPLX_HINGES.items()}
    return dirs, local, poses.shape[0], float(d["mocap_frame_rate"])


def smplx_rest():
    joints, world, _ = smplx_fk(np.zeros((1, 165)), np.zeros(300))
    d0 = {k: distal_local(joints[:, a], joints[:, b], joints[:, c], world[:, a])[0]
          for k, (a, b, c) in SMPLX_HINGES.items()}
    rest_rot = {k: world[0, a] for k, (a, _, _) in SMPLX_HINGES.items()}
    return d0, rest_rot, np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------- MetaPerson rig input
def rig_world(glb: Path, spec: str):
    from physics_audit import load_track
    from refine_track import fk_world

    skel, rot, tr, n, fps = load_track(glb, spec)
    pos, wrot = fk_world(skel, rot, tr, n, want_rot=True)
    return pos, wrot, n, fps


def rig_load(glb: Path, spec: str):
    pos, wrot, n, fps = rig_world(glb, spec)
    return {k: distal_local(pos[a], pos[b], pos[c], wrot[a]) for k, (a, b, c) in RIG_HINGES.items()}, n, fps


def rig_rest(glb: Path):
    from retarget import read_glb
    from refine_track import Skeleton, fk_world

    gltf, _ = read_glb(glb)
    pos, wrot = fk_world(Skeleton(gltf), {}, {}, 1, want_rot=True)
    d0 = {k: distal_local(pos[a], pos[b], pos[c], wrot[a])[0] for k, (a, b, c) in RIG_HINGES.items()}
    rest_rot = {k: wrot[a][0] for k, (a, _, _) in RIG_HINGES.items()}
    fwd = sum(pos[f"{s}ToeBase"][0] - pos[f"{s}Foot"][0] for s in ("Left", "Right"))
    fwd[1] = 0
    return d0, rest_rot, fwd / np.linalg.norm(fwd), [c["name"] for c in gltf["animations"]]


# ---------------------------------------------------------------------- calibration
def calibrate(reference_dirs: list[dict], d0, rest_rot, forward) -> dict:
    """Hinge axis per joint from reference mocap, its flexion sign from anatomy (elbow flexion
    brings the forearm forward from the T-pose, knee flexion takes the shank back)."""
    axes = {}
    for k in d0:
        d = np.concatenate([r[k] for r in reference_dirs])
        h = fit_axis(d, d0[k])
        sign = flexion_sign(h, d0[k], rest_rot[k], forward, toward_front="elbow" in k)
        flex, side = hinge_angles(d, d0[k], h, sign)
        axes[k] = {"h": h, "sign": sign, "ref_flex_p1": float(np.percentile(flex, 1)),
                   "ref_side_p99": float(np.percentile(side, 99))}
    return axes


# ---------------------------------------------------------------------- report
def summarise(label: str, r: dict) -> list[str]:
    fails = []
    n = r["frames"]
    print(f"\n{label}  ({n} frames, {n / r['fps']:.1f} s)")
    for k, v in r["joints"].items():
        if "wrist" in k:
            fl, dv = v
            bad_f, bad_d = np.abs(fl) > WRIST["flex"], np.abs(dv) > WRIST["dev"]
            print(f"  {k:8s} flex/ext p1..p99 {np.percentile(fl, 1):6.1f}..{np.percentile(fl, 99):6.1f}   "
                  f"deviation p1..p99 {np.percentile(dv, 1):6.1f}..{np.percentile(dv, 99):6.1f}   "
                  f"over limit: flex {bad_f.mean():.1%}  dev {bad_d.mean():.1%}")
            for what, bad in (("flex", bad_f), ("deviation", bad_d)):
                if bad.mean() > FAIL_SHARE:
                    fails.append(f"{label}: {k} {what} beyond limit on {bad.mean():.1%} of frames")
            continue
        flex, side = v
        lim = LIMITS[k.split()[1]]
        hyper, over, sideways = -flex > lim["hyper"], flex > lim["flex"], side > lim["side"]
        print(f"  {k:8s} flex min/p1/p50/p99/max {flex.min():6.1f} {np.percentile(flex, 1):6.1f} "
              f"{np.percentile(flex, 50):6.1f} {np.percentile(flex, 99):6.1f} {flex.max():6.1f}   "
              f"off-hinge p99/max {np.percentile(side, 99):5.1f} {side.max():5.1f}   "
              f"hyperext {hyper.mean():5.1%}  overflex {over.mean():5.1%}  sideways {sideways.mean():5.1%}")
        for what, bad in (("hyperextension", hyper), ("overflexion", over), ("sideways bend", sideways)):
            if what == "sideways bend" and "knee" in k and r.get("rig"):
                continue
            if bad.mean() > FAIL_SHARE:
                fails.append(f"{label}: {k} {what} on {bad.mean():.1%} of frames "
                             f"(worst {(-flex).max() if what[0] == 'h' else (flex.max() if what[0] == 'o' else side.max()):.0f} deg)")
    return fails


def main() -> None:
    args = sys.argv[1:]
    glb = HERE.parent / "public" / "assets" / "model-clips.glb"
    ref = sorted((HERE / "reference" / "beat2").glob("*.npz"))[:8]
    if "--glb" in args:
        i = args.index("--glb")
        glb = Path(args[i + 1])
        del args[i:i + 2]
    if not args:
        print(__doc__)
        sys.exit(2)
    fails: list[str] = []
    smplx_axes = rig_axes = None
    for spec in args:
        label = spec if spec.startswith("clip:") else Path(spec).name
        if spec.endswith(".npz"):
            if smplx_axes is None:
                d0, rr, fwd = smplx_rest()
                smplx_axes = (d0, calibrate([smplx_load(p)[0] for p in ref], d0, rr, fwd))
                print_axes(f"SMPL-X, from {len(ref)} BEAT2 mocap clips", smplx_axes[1])
            dirs, local, n, fps = smplx_load(Path(spec))
            d0, axes = smplx_axes
            joints = {k: hinge_angles(dirs[k], d0[k], axes[k]["h"], axes[k]["sign"]) for k in SMPLX_HINGES}
            joints.update({f"{s} wrist": wrist_rom(local, s) for s in "LR"})
        else:
            if rig_axes is None:
                d0, rr, fwd, clips = rig_rest(glb)
                rig_axes = (d0, calibrate([rig_load(glb, "clip:" + c)[0] for c in clips], d0, rr, fwd))
                print_axes(f"MetaPerson rig, from its {len(clips)} Mixamo clips", rig_axes[1])
            dirs, n, fps = rig_load(glb, spec)
            d0, axes = rig_axes
            joints = {k: hinge_angles(dirs[k], d0[k], axes[k]["h"], axes[k]["sign"]) for k in RIG_HINGES}
        fails += summarise(label, {"frames": n, "fps": fps, "joints": joints, "rig": not spec.endswith(".npz")})
    print()
    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("PASS: no elbow, knee or wrist beyond its anatomical range")


def print_axes(what: str, axes: dict) -> None:
    print(f"hinge axes ({what}; reference flexion p1 and off-hinge p99 in deg):")
    for k, a in axes.items():
        h = a["h"]
        print(f"  {k:8s} h = [{h[0]:+.2f} {h[1]:+.2f} {h[2]:+.2f}] sign {a['sign']:+.0f}   "
              f"ref flex p1 {a['ref_flex_p1']:6.1f}   ref off-hinge p99 {a['ref_side_p99']:5.1f}")


if __name__ == "__main__":
    main()
