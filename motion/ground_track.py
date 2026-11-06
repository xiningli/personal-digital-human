"""Root the body at its feet: planted feet stay put, the pelvis moves, the legs reach down.

    .venv/bin/python ground_track.py <track.json | clip:<name>> [--avatar model-clips.glb]
        [--sway 0.6] [--tilt 0.5] [--once] [--force] [--out out.json]

Why (measured 2026-09-28 with physics_audit.py on the TED stage track personal-site shipped):
retarget.py drops the root translation, so the player pinned the pelvis in space and FK'd
everything else from it. That puts the kinematic chain upside down. In a standing person
friction holds the loaded foot still and the pelvis is wherever the legs put it; here the
pelvis was still and the feet went wherever the leg rotations put them. Every weight shift,
knee bend or step of the source showed up as the *feet* moving: a planted foot travelled up
to 31 cm inside a single stance (2.5 m of skating per minute), the source's steps became
steps in place (8 a minute), and the runtime's vertical ground shift (player/ground.ts, lower
sole pressed to the floor) bobbed the whole body 6.6 cm to keep up. That is the "balloon
without gravity" the owner kept reporting. The per-frame gates missed it because the drift is
slow: 0.14 cm/frame is under the skate gate, and 200 frames of it is 28 cm (check_track.py
now gates the per-stance travel). correct_balance.py then fixed the capture point by tilting
the pelvis up to 30 deg, which is where the sideways lean came from; this pass replaces it.

The asset's own capture clips had the other half of the problem: all nine stand crouched and
wide (15-34 deg of knee flexion, ankles 34-43 cm apart, Breathing Idle's toes turned out ~40
deg), a retargeting leftover and the "gorilla" the owner kept rejecting — and a stance the
speaking track did not share, so every idle/speaking crossfade dragged the feet somewhere new.
scripts/ground-avatar-clips.mjs runs this same pass over every clip (clip:<name>).

Per frame, from the ground up:

1. **Feet.** Fixed at one natural stance for every motion source: ankle centres
   STANCE_HALF_WIDTH either side of the midline (about under the hip joints), toes out
   TOE_OUT_DEG, soles flat (the Idle clip's foot pitch), and the polygon's centre placed
   under the body's median CoM. A presenter answering a question does not walk; the source's
   steps are not reproduced.
2. **Pelvis tilt.** Yaw kept whole (turning to the audience), roll/pitch scaled by --tilt:
   with the legs solved afterwards this only straightens the body.
3. **Pelvis, horizontally.** Where the source's pelvis sits over its own planted ankles is the
   weight-shift signal: low-passed at SWAY_CUTOFF_HZ (quiet-standing sway and talking weight
   shifts live well below 1 Hz, and a slow CoM keeps the zero-moment point inside the feet),
   scaled by --sway, soft-limited to SWAY_LIMIT. Then iterated: the whole-body CoM (Dempster
   segments, balance.py) is pushed MARGIN inside the sole polygon (heel to toe tip), and the
   LIPM capture point CP_MARGIN inside the ankle+toe polygon that check_track.py's balance gate
   audits, each push low-passed before it moves the pelvis.
4. **Pelvis, vertically.** The height at which the more-stretched leg is flexed REST_KNEE_DEG,
   lowered by the source's own knee dip (bounded by DIP_LIMIT), capped at MIN_KNEE_DEG. The
   cap goes through a moving minimum before the Hann smoothing, so the smoothed height
   provably never locks a knee.
5. **Legs.** Analytic two-bone IK from the hip joints to the fixed ankles: the knee angle from
   the law of cosines, the smallest hip rotation that lands the ankle, then a twist about the
   hip-ankle axis that puts the knee over the toes (the knee is a hinge and the tibia barely
   twists). Foot and toe get the stance's rotations, so the sole is flat.
6. **Hands** (tracks only). The legs moved under arms that did not, so a hand can end up inside
   a thigh; clear_hands() abducts the arm the least that clears it.

The rest of the upper body is the input's. For a looping source the last LOOP_BLEND_S of every
rotation is slerped onto frame 0 and every filter wraps, so LoopRepeat does not jump. The
output gets a "hips" array (F x 3, the Hips translation; AvatarRoot is identity, so world),
which player/avatar.ts trackToClip plays as Hips.position.

Idempotent: a second run refuses unless --force. Verify with check_track.py (every gate,
including per-stance travel and balance) and physics_audit.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from refine_track import (  # noqa: E402
    CAPSULES, DEFAULT_AVATAR, HAND_POINTS, Skeleton, calibrated_surfaces, clip_locals, fk_world,
    hann_smooth, penetration_counts, read_glb, track_rot,
)
from balance import SEGMENTS, capture_point  # noqa: E402

VERSION = 1
SIDES = ("Left", "Right")
REST_KNEE_DEG = 7.0      # knee flexion of relaxed standing: slightly soft, never locked
MIN_KNEE_DEG = 2.0       # the knee never straightens past this, dip or not
STANCE_HALF_WIDTH = 0.105  # ankle centre from the midline (m): feet about under the hip joints
TOE_OUT_DEG = 10.0       # each foot turned out this far from straight ahead
SWAY_LIMIT = (0.04, 0.025)  # soft limit on the pelvis weight shift, lateral / fore-aft (m)
DIP_LIMIT = 0.02         # the pelvis lowers at most this far below its rest height (knee flex)
MARGIN = 0.04            # CoM kept this far inside the support polygon (m)
CP_MARGIN = 0.015        # capture point kept this far inside the ankle+toe polygon (m)
SWAY_CUTOFF_HZ = 0.6     # low-pass for the pelvis's horizontal weight-shift signal
HEIGHT_CUTOFF_HZ = 1.0   # and for its height (knee flex)
LOOP_BLEND_S = 1.0       # the loop seam is closed over this long
FOOT_LEN_BACK = 0.06     # heel behind the ankle, m (sole outline for the support polygon)
FOOT_LEN_FRONT = 0.07    # toe tip beyond the toe joint, m
FOOT_HALF_WIDTH = 0.045


# -- small rotation helpers on (F,3,3) matrices -----------------------------------------
def rodrigues(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(axis * angle[:, None]).as_matrix()


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def swing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(F,3,3): the smallest rotation taking direction a onto direction b."""
    a, b = unit(a), unit(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis, axis=1)
    c = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    ang = np.arctan2(s, c)
    return rodrigues(unit(axis), np.where(s > 1e-9, ang, 0.0))


def mats(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(q).as_matrix()


def quats(m: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(m).as_quat()


# -- periodic filters ----------------------------------------------------------------------
def _pad(x: np.ndarray, radius: int, wrap: bool) -> np.ndarray:
    if wrap and len(x) > 2 * radius:
        return np.concatenate([x[-radius:], x, x[:radius]])
    return np.pad(x, [(radius, radius)] + [(0, 0)] * (x.ndim - 1), mode="edge")


def hann_lowpass(x: np.ndarray, cutoff_hz: float, fps: float, wrap: bool = True) -> np.ndarray:
    """Hann-window FIR, wrapped when the motion loops. Half-power near cutoff_hz."""
    radius = max(1, int(round(fps / cutoff_hz / 2)))
    w = np.hanning(2 * radius + 3)[1:-1]
    w /= w.sum()
    pad = _pad(x, radius, wrap)
    if x.ndim == 1:
        return np.convolve(pad, w, mode="valid")
    return np.stack([np.convolve(pad[:, k], w, mode="valid") for k in range(x.shape[1])], axis=1)


def capped_lowpass(x: np.ndarray, cap: np.ndarray, cutoff_hz: float, fps: float, wrap: bool = True) -> np.ndarray:
    """Low-pass min(x, cap) such that the result never exceeds cap: erode by the window radius
    first (a moving minimum), then Hann with the same radius. Each output averages eroded
    values whose windows all contain the frame, so every term is <= cap at that frame."""
    radius = max(1, int(round(fps / cutoff_hz / 2)))
    y = np.minimum(x, cap)
    pad = _pad(y, radius, wrap)
    eroded = np.lib.stride_tricks.sliding_window_view(pad, 2 * radius + 1).min(axis=1)
    return np.minimum(hann_lowpass(eroded, cutoff_hz, fps, wrap), cap)


# -- geometry ------------------------------------------------------------------------------
def sole(ankle: np.ndarray, toe: np.ndarray) -> np.ndarray:
    """(4,2) xz corners of a foot's sole from its ankle and toe joints (one frame)."""
    a, t = ankle[[0, 2]], toe[[0, 2]]
    d = t - a
    d /= max(np.linalg.norm(d), 1e-9)
    n = np.array([-d[1], d[0]])
    heel, tip = a - d * FOOT_LEN_BACK, t + d * FOOT_LEN_FRONT
    return np.array([heel + n * FOOT_HALF_WIDTH, heel - n * FOOT_HALF_WIDTH,
                     tip - n * FOOT_HALF_WIDTH, tip + n * FOOT_HALF_WIDTH])


def hull(pts: np.ndarray) -> np.ndarray:
    p = sorted(map(tuple, pts))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo: list = []
    for pt in p:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], pt) <= 0:
            lo.pop()
        lo.append(pt)
    up: list = []
    for pt in reversed(p):
        while len(up) >= 2 and cross(up[-2], up[-1], pt) <= 0:
            up.pop()
        up.append(pt)
    return np.array(lo[:-1] + up[:-1])


def push_inside(pts: np.ndarray, poly: np.ndarray, margin: float) -> np.ndarray:
    """(F,2) displacement that moves each point at least `margin` inside the CCW convex polygon."""
    out = np.zeros_like(pts)
    n = len(poly)
    for _ in range(3):                       # edges interact at corners; a few sweeps settle it
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            e = b - a
            inward = np.array([-e[1], e[0]]) / np.linalg.norm(e)   # CCW: left of the edge is inside
            depth = (pts + out - a) @ inward                        # signed distance inside
            need = np.maximum(margin - depth, 0.0)
            out += need[:, None] * inward[None, :]
    return out


def com(pos: dict[str, np.ndarray]) -> np.ndarray:
    total = sum(m for m, _ in SEGMENTS)
    return sum((m / total) * np.mean([pos[b] for b in bs], axis=0) for m, bs in SEGMENTS)


# -- the pass ------------------------------------------------------------------------------
def close_seam(q: np.ndarray, fps: float) -> np.ndarray:
    """Slerp the last LOOP_BLEND_S of every bone onto frame 0, smoothstep-weighted."""
    n = min(len(q) // 4, int(round(LOOP_BLEND_S * fps)))
    if n < 2:
        return q
    out = q.copy()
    first = q[0]
    for k in range(n):
        f = len(q) - n + k
        t = (k + 1) / n
        w = t * t * (3 - 2 * t)
        a, b = q[f], first.copy()
        b[np.sum(a * b, axis=1) < 0] *= -1
        mix = a * (1 - w) + b * w                     # nlerp: fine at these small angles
        out[f] = mix / np.linalg.norm(mix, axis=1, keepdims=True)
    return out


def leg_ik(H, R_hips, R_ul, R_l, t_leg, t_foot, target, toe_dir):
    """Two-bone IK. H hip joint (F,3); R_ul/R_l the thigh/shank world rotations to start from;
    t_leg/t_foot the knee/ankle offsets in their parents' frames; target ankle (F,3);
    toe_dir (3,) horizontal foot direction the knee should point over. Returns new world
    rotations for thigh and shank."""
    a, b = np.linalg.norm(t_leg), np.linalg.norm(t_foot)
    K = H + np.einsum("fij,j->fi", R_ul, t_leg)
    A = K + np.einsum("fij,j->fi", R_l, t_foot)
    c = np.clip(np.linalg.norm(target - H, axis=1), abs(a - b) + 1e-4, (a + b) * 0.99995)
    want = np.arccos(np.clip((a * a + b * b - c * c) / (2 * a * b), -1, 1))
    u, v = H - K, A - K
    now = np.arccos(np.clip(np.sum(unit(u) * unit(v), axis=1), -1, 1))
    n = np.cross(u, v)
    fallback = np.einsum("fij,j->fi", R_hips, np.array([1.0, 0.0, 0.0]))    # pelvis lateral
    n = np.where((np.linalg.norm(n, axis=1) > 1e-6)[:, None], unit(n), fallback)
    R_l = rodrigues(n, want - now) @ R_l                     # open/close the knee (about K)
    A = K + np.einsum("fij,j->fi", R_l, t_foot)
    Q = swing(A - H, target - H)                              # land the ankle (about H)
    R_ul, R_l = Q @ R_ul, Q @ R_l
    # knee over the toes: twist the whole leg about the hip-ankle axis
    axis = unit(target - H)
    K = H + np.einsum("fij,j->fi", R_ul, t_leg)
    knee = K - H - np.sum((K - H) * axis, axis=1, keepdims=True) * axis
    pole = toe_dir[None, :] - np.sum(toe_dir[None, :] * axis, axis=1, keepdims=True) * axis
    cosang = np.sum(unit(knee) * unit(pole), axis=1)
    sinang = np.sum(np.cross(unit(knee), unit(pole)) * axis, axis=1)
    ok = (np.linalg.norm(knee, axis=1) > 1e-3)
    T = rodrigues(axis, np.where(ok, np.arctan2(sinang, cosang), 0.0))
    return T @ R_ul, T @ R_l


def clear_hands(skel: Skeleton, surfaces: dict[str, float], bones: list[str], q: np.ndarray,
                idx: dict[str, int], p: np.ndarray, radius: int = 9) -> int:
    """A hand cannot pass through a thigh. Solving the legs moves the thighs and shins under
    arms that did not move with them (arms and torso both hang off the pelvis, so only the
    leg capsules are new), and a hand hanging in front of a thigh can end up inside it.
    Where any hand joint is closer to a leg capsule than any capture clip ever gets
    (refine_track.py's calibrated surfaces, x1.08), the whole arm abducts about the
    shoulder by the angle that clears it: depth / shoulder-to-joint lever, through a moving
    maximum then a Hann window of the same radius (so the smoothed angle is never below
    what a frame needs). Iterated: each pass re-FKs. Returns the frames touched. q is
    modified in place."""
    F = len(q)
    legs = {k: v for k, v in CAPSULES.items() if k.startswith(("thigh", "shin"))}
    touched = np.zeros(F, dtype=bool)
    for _ in range(4):
        pos, rots = fk_world(skel, track_rot(bones, q), {"Hips": p}, F, want_rot=True)
        moved = False
        for side in SIDES:
            depth = np.zeros(F)
            esc = np.zeros((F, 3))
            lever = np.full(F, 1.0)
            for hp in HAND_POINTS:
                if not hp.startswith(side) or hp not in pos:
                    continue
                for cap, (a, b, _r) in legs.items():
                    ab = pos[b] - pos[a]
                    t = np.clip(np.sum((pos[hp] - pos[a]) * ab, axis=1, keepdims=True)
                                / np.maximum(np.sum(ab * ab, axis=1, keepdims=True), 1e-12), 0, 1)
                    c = pos[a] + t * ab
                    need = surfaces[cap] * 1.08 - np.linalg.norm(pos[hp] - c, axis=1)
                    better = need > depth
                    depth = np.where(better, need, depth)
                    esc = np.where(better[:, None], pos[hp] - c, esc)
                    lever = np.where(better, np.linalg.norm(pos[hp] - pos[f"{side}Arm"], axis=1), lever)
            if not (depth > 1e-4).any():
                continue
            moved = True
            touched |= depth > 1e-4
            theta = np.clip(depth / np.maximum(lever, 0.2), 0, 0.3)
            pad = np.concatenate([np.zeros(radius), theta, np.zeros(radius)])
            theta = hann_smooth(np.lib.stride_tricks.sliding_window_view(pad, 2 * radius + 1).max(axis=1), radius)
            v = pos[f"{side}Hand"] - pos[f"{side}Arm"]
            axis = np.cross(v, esc)
            axis = np.stack([hann_smooth(axis[:, k] * (depth > 1e-4), radius) for k in range(3)], 1)  # steady through a contact
            live = (theta > 1e-5) & (np.linalg.norm(axis, axis=1) > 1e-9)
            axis = unit(np.where(live[:, None], axis, np.array([0.0, 0.0, 1.0])))
            Rw = rodrigues(axis, np.where(live, theta, 0.0))
            gp = rots[f"{side}Shoulder"]
            Rl = np.transpose(gp, (0, 2, 1)) @ Rw @ gp @ mats(q[:, idx[f"{side}Arm"]])
            q[:, idx[f"{side}Arm"]] = quats(Rl)
        if not moved:
            break
    return int(touched.sum())


def natural_stance(skel: Skeleton, gltf: dict, bin_: bytes, com_over_ankles_z: float) -> dict:
    """The stance the body is solved onto. Ankles STANCE_HALF_WIDTH either side of the
    midline, toes out TOE_OUT_DEG, the sole pitched as the asset's own idle foot (so the
    toe joint sits SOLE_BELOW_TOE up and the sole is flat), and the ankles placed so the
    support polygon's centre lies under the body's mean CoM (com_over_ankles_z: how far in
    front of the ankles that CoM falls, measured by the caller)."""
    clip = next(c for c in gltf["animations"] if c["name"] == "Idle")
    irot, itr, _ = clip_locals(gltf, bin_, clip)
    ipos, iR = fk_world(skel, {k: v[:1] for k, v in irot.items()}, {k: v[:1] for k, v in itr.items()}, 1, want_rot=True)
    out = {}
    for s, sign in (("Left", 1.0), ("Right", -1.0)):
        R_idle = iR[f"{s}Foot"][0]
        d = ipos[f"{s}ToeBase"][0] - ipos[f"{s}Foot"][0]
        yaw_now = np.arctan2(d[0], d[2])                       # 0 = straight ahead (+z)
        yaw_want = sign * np.radians(TOE_OUT_DEG)
        R_foot = Rotation.from_rotvec([0, yaw_want - yaw_now, 0]).as_matrix() @ R_idle
        ankle = np.array([sign * STANCE_HALF_WIDTH, ipos[f"{s}Foot"][0][1], 0.0])
        toe = ankle + R_foot @ skel.bind_tr[skel.name2idx[f"{s}ToeBase"]]
        out[s] = {"ankle": ankle, "toe": toe, "R_foot": R_foot,
                  "toe_local": irot.get(f"{s}ToeBase", np.array([[0, 0, 0, 1.0]]))[0]}
    # Put the sole polygon's centre under the mean CoM.
    poly = hull(np.concatenate([sole(out[s]["ankle"], out[s]["toe"]) for s in SIDES]))
    shift = com_over_ankles_z - poly[:, 1].mean()
    for s in SIDES:
        out[s]["ankle"] = out[s]["ankle"] + np.array([0, 0, shift])
        out[s]["toe"] = out[s]["toe"] + np.array([0, 0, shift])
    return out


def knee_reach(a: float, b: float, knee_deg: float) -> float:
    """Hip-to-ankle distance of a two-bone leg flexed knee_deg at the knee."""
    return float(np.sqrt(a * a + b * b + 2 * a * b * np.cos(np.radians(knee_deg))))


def load_source(spec: str, gltf: dict, bin_: bytes):
    """A track.json path, or clip:<name> from the glb. Returns (track dict to write back,
    bones, quats (F,B,4), source translations for FK, fps)."""
    if spec.startswith("clip:"):
        clip = next(c for c in gltf["animations"] if c["name"] == spec[5:])
        rot, tr, n = clip_locals(gltf, bin_, clip)
        bones = sorted(rot)
        q = np.stack([rot[b] for b in bones], axis=1)
        track = {"fps": 30, "frames": n, "bones": bones, "source": spec}
        return track, bones, q, {"Hips": tr["Hips"]} if "Hips" in tr else {}, 30.0
    track = json.loads(Path(spec).read_text())
    F = track["frames"]
    bones = track["bones"]
    q = np.array(track["quats"], dtype=float).reshape(F, len(bones), 4)
    return track, bones, q, {}, float(track["fps"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="track.json (rewritten in place unless --out) or clip:<name> from the glb")
    ap.add_argument("--avatar", type=Path, default=DEFAULT_AVATAR)
    ap.add_argument("--sway", type=float, default=0.6,
                    help="fraction of the source's weight shift (pelvis over its support) kept")
    ap.add_argument("--tilt", type=float, default=0.5,
                    help="fraction of the pelvis's roll/pitch kept (its yaw is kept whole); the legs "
                         "are solved to the feet afterwards, so this only straightens the body")
    ap.add_argument("--once", action="store_true",
                    help="a one-shot (greeting): no loop seam, filters do not wrap around")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    wrap = not args.once

    gltf, bin_ = read_glb(args.avatar)
    skel = Skeleton(gltf)
    track, bones, q, src_tr, fps = load_source(args.source, gltf, bin_)
    if track.get("grounded") and not args.force:
        print(f"refusing: {args.source} is already grounded (v{track['grounded']['version']}); --force to redo")
        return
    if args.source.startswith("clip:") and not args.out:
        raise SystemExit("clip sources need --out")
    F = len(q)
    idx = {b: k for k, b in enumerate(bones)}
    for b in ("Hips",) + tuple(f"{s}{p}" for s in SIDES for p in ("UpLeg", "Leg", "Foot", "ToeBase")):
        if b not in idx:                                       # a bone the source never rotates
            bones.append(b)
            idx[b] = len(bones) - 1
            q = np.concatenate([q, np.broadcast_to(skel.bind_rot[skel.name2idx[b]], (F, 1, 4))], axis=1)
    q = q / np.linalg.norm(q, axis=2, keepdims=True)
    if wrap:
        q = close_seam(q, fps)

    hips_i = skel.name2idx["Hips"]
    parent = skel.nodes[skel.parent[hips_i]]
    assert not parent.get("rotation") and not parent.get("translation") and not parent.get("scale"), \
        "Hips parent must be identity (the hips array is written as world translation)"
    # Pelvis tilt: keep the yaw (turning to the audience), attenuate roll/pitch (leaning).
    R0 = mats(q[:, idx["Hips"]])
    fwd = R0 @ np.array([0.0, 0.0, 1.0])
    yaw = np.arctan2(fwd[:, 0], fwd[:, 2])
    Ry = Rotation.from_rotvec(np.stack([np.zeros(F), yaw, np.zeros(F)], 1)).as_matrix()
    tilt = Rotation.from_matrix(np.transpose(Ry, (0, 2, 1)) @ R0).as_rotvec()
    tilt_ref = Rotation.from_matrix(mats(skel.bind_rot[hips_i][None])).as_rotvec()[0]
    R_hips = Ry @ Rotation.from_rotvec(tilt_ref + args.tilt * (tilt - tilt_ref)).as_matrix()
    q[:, idx["Hips"]] = quats(R_hips)
    t_ul = {s: skel.bind_tr[skel.name2idx[f"{s}UpLeg"]] for s in SIDES}
    t_leg = {s: skel.bind_tr[skel.name2idx[f"{s}Leg"]] for s in SIDES}
    t_foot = {s: skel.bind_tr[skel.name2idx[f"{s}Foot"]] for s in SIDES}
    ab = {s: (np.linalg.norm(t_leg[s]), np.linalg.norm(t_foot[s])) for s in SIDES}

    # The source's own weight-shift and knee-flex signals: its pelvis over its planted ankles.
    spos = fk_world(skel, track_rot(bones, q), src_tr, F)
    toes = np.stack([spos[f"{s}ToeBase"][:, 1] for s in SIDES], axis=1)
    planted = toes - toes.min(axis=1, keepdims=True) < 0.02
    ank = np.stack([spos[f"{s}Foot"] for s in SIDES], axis=1)
    support = (ank * planted[..., None]).sum(1) / planted.sum(1, keepdims=True)
    over = spos["Hips"] - support
    lim = np.array(SWAY_LIMIT)
    sway = hann_lowpass(over[:, [0, 2]] - np.median(over[:, [0, 2]], axis=0), SWAY_CUTOFF_HZ, fps, wrap)
    sway = lim * np.tanh(args.sway * sway / lim)                     # soft limit: small shifts pass as is
    dip = hann_lowpass(over[:, 1] - np.median(over[:, 1]), HEIGHT_CUTOFF_HZ, fps, wrap)
    dip = -DIP_LIMIT * np.tanh(np.maximum(-dip, 0.0) / DIP_LIMIT)    # only ever lower, and softly bounded

    def solve(p: np.ndarray, stance: dict) -> dict[str, np.ndarray]:
        out = {}
        for s in SIDES:
            H = p + np.einsum("fij,j->fi", R_hips, t_ul[s])
            R_ul = R_hips @ mats(q[:, idx[f"{s}UpLeg"]])
            R_l = R_ul @ mats(q[:, idx[f"{s}Leg"]])
            target = np.broadcast_to(stance[s]["ankle"], (F, 3))
            toe_dir = stance[s]["toe"] - stance[s]["ankle"]
            toe_dir[1] = 0
            R_ul, R_l = leg_ik(H, R_hips, R_ul, R_l, t_leg[s], t_foot[s], target, unit(toe_dir))
            R_foot = np.broadcast_to(stance[s]["R_foot"], (F, 3, 3))
            out[f"{s}UpLeg"] = quats(np.transpose(R_hips, (0, 2, 1)) @ R_ul)
            out[f"{s}Leg"] = quats(np.transpose(R_ul, (0, 2, 1)) @ R_l)
            out[f"{s}Foot"] = quats(np.transpose(R_l, (0, 2, 1)) @ R_foot)
            out[f"{s}ToeBase"] = np.broadcast_to(stance[s]["toe_local"], (F, 4))
        return out

    def height_cap(p_xz: np.ndarray, stance: dict, knee_deg: float) -> np.ndarray:
        """Pelvis y at which the more-stretched leg is flexed exactly knee_deg."""
        cap = np.full(F, np.inf)
        for s in SIDES:
            d = np.einsum("fij,j->fi", R_hips, t_ul[s])
            r = knee_reach(*ab[s], knee_deg)
            A = stance[s]["ankle"]
            dx, dz = p_xz[:, 0] + d[:, 0] - A[0], p_xz[:, 1] + d[:, 2] - A[2]
            cap = np.minimum(cap, A[1] - d[:, 1] + np.sqrt(np.maximum(r * r - dx * dx - dz * dz, 0.0)))
        return cap

    def pose(p: np.ndarray, stance: dict):
        p[:, 1] = capped_lowpass(height_cap(p[:, [0, 2]], stance, REST_KNEE_DEG) + dip,
                                 height_cap(p[:, [0, 2]], stance, MIN_KNEE_DEG), HEIGHT_CUTOFF_HZ, fps, wrap)
        q2 = q.copy()
        for b, v in solve(p, stance).items():
            q2[:, idx[b]] = v
        return q2, fk_world(skel, track_rot(bones, q2), {"Hips": p}, F)

    # Pass 0 finds where the CoM sits relative to the pelvis, so the feet can go under it.
    p = np.zeros((F, 3))
    p[:, [0, 2]] = sway
    stance = natural_stance(skel, gltf, bin_, 0.0)
    _, pos = pose(p, stance)
    com_z = float(np.median(com(pos)[:, 2]))
    stance = natural_stance(skel, gltf, bin_, com_z)
    poly = hull(np.concatenate([sole(stance[s]["ankle"], stance[s]["toe"]) for s in SIDES]))
    # check_track.py's balance gate audits the capture point against the ankle+toe joints only
    # (no heel): conservative, and held here too, so the gate passes as written.
    joints = hull(np.array([stance[s][k][[0, 2]] for s in SIDES for k in ("ankle", "toe")]))

    for it in range(12):
        q2, pos = pose(p, stance)
        c3 = com(pos)
        cp = capture_point(c3, fps, float(min(stance[s]["toe"][1] for s in SIDES)))
        fix = push_inside(c3[:, [0, 2]], poly, MARGIN)
        fix += push_inside(cp + fix, joints, CP_MARGIN)
        worst = float(np.linalg.norm(fix, axis=1).max())
        print(f"  pass {it + 1}: CoM needed up to {worst * 100:.2f} cm to stay {MARGIN * 100:.0f} cm inside the feet")
        if worst < 1e-3:
            break
        # a smoothed push undershoots a sharp peak; the next pass re-measures and adds the rest
        p[:, [0, 2]] += hann_lowpass(fix * 1.5, SWAY_CUTOFF_HZ * 2, fps, wrap)
    q2, pos = pose(p, stance)                 # the legs for the pelvis path actually written

    if not args.source.startswith("clip:") and gltf.get("animations"):
        surfaces = calibrated_surfaces(gltf, bin_, skel)
        n = clear_hands(skel, surfaces, bones, q2, idx, p)
        pos = fk_world(skel, track_rot(bones, q2), {"Hips": p}, F)
        touch, deep = penetration_counts(pos, surfaces)
        print(f"  hands cleared of the legs on {n} frames; clipping now touch {touch}, deep {deep}")

    err = max(float(np.linalg.norm(pos[f"{s}Foot"] - stance[s]["ankle"], axis=1).max()) for s in SIDES)
    knee = []
    for s in SIDES:
        H, K, A = pos[f"{s}UpLeg"], pos[f"{s}Leg"], pos[f"{s}Foot"]
        knee.append(180 - np.degrees(np.arccos(np.clip(np.sum(unit(H - K) * unit(A - K), axis=1), -1, 1))))
    knee = np.concatenate(knee)
    width = np.linalg.norm(stance["Left"]["ankle"] - stance["Right"]["ankle"])
    print(f"  stance {width * 100:.0f} cm ankle to ankle, toes out {TOE_OUT_DEG:g} deg;"
          f" ankle error max {err * 1000:.2f} mm; knee flexion {knee.min():.1f}-{knee.max():.1f} deg"
          f" (median {np.median(knee):.1f}); pelvis range x {np.ptp(p[:, 0]) * 100:.1f}, z {np.ptp(p[:, 2]) * 100:.1f},"
          f" y {np.ptp(p[:, 1]) * 100:.1f} cm")

    track["bones"] = bones
    track["frames"] = F
    track["quats"] = [round(float(x), 6) for x in q2.reshape(-1)]
    track["hips"] = [round(float(x), 5) for x in p.reshape(-1)]
    track["grounded"] = {"version": VERSION, "sway": args.sway, "tilt": args.tilt, "loop": wrap,
                         "stance_half_width_m": STANCE_HALF_WIDTH, "toe_out_deg": TOE_OUT_DEG,
                         "rest_knee_deg": REST_KNEE_DEG, "margin_m": MARGIN,
                         "sway_cutoff_hz": SWAY_CUTOFF_HZ, "loop_blend_s": LOOP_BLEND_S if wrap else 0}
    out = args.out or Path(args.source)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(track, separators=(",", ":")))
    os.replace(tmp, out)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
