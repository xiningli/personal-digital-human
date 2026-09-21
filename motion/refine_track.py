"""Elegance post-pass for a generated motion track: recenter the chronic static offsets
that make GVHMR-derived tracks read as a crouched gorilla, while keeping the gestures.

    .venv/bin/python refine_track.py <track.json> [--reference ref_means.json]
        [--alpha-arms 1.0] [--alpha-hips 1.0] [--alpha-hips-dynamic 0.4] [--alpha-legs 0.6]
        [--force] [--out out.json]

Measured on the Brunton lecture profile (5400 frames, 2026-09-21): both forearms sit ~85 deg
away from the rig's rest pose on time average (the "Talking" capture sits at ~40), the hands
are above the chest in half the frames, and the hips keep twisting (48 deg of yaw range)
because GVHMR never saw the lecturer's feet. Angular velocity is already slower than the
captures, so nothing here is a low-pass; the fixes are static re-centering and dynamic
shrinks around the track's own mean:

  arms  (Shoulder/Arm/ForeArm/Hand): recentered onto the reference clip's time-mean local
        rotation: q'(t) = slerp(I, ref_mean * track_mean^-1, alpha_arms) * q(t). A constant
        left-multiplied rotation moves the whole time series, so the mean lands on the
        reference pose and every gesture survives unchanged in shape.
  torso (Hips/Spine/Spine1/Spine2/Neck/Head): same recentering (alpha_hips / alpha_torso).
        The torso must move as a chain: the retargeter's zero convention differs from the
        capture clips' (the track compensates a 30 deg hip offset with ~38 deg of spine bend;
        Talking carries no such compensation), so recentering the hips alone pitches the body
        forward ~20 deg (measured). On top of recentering, the hips' dynamic component is
        shrunk toward the track mean:
        q'(t) = delta * slerp(track_mean, q(t), alpha_hips_dynamic), which is what kills the
        left-right twisting.
  legs  (UpLeg/Leg): GVHMR estimated them from a lecturer whose feet are out of frame, so
        they are not trusted at all: dynamics shrunk by 1/2 toward the track mean, then the
        result converged toward the rig's rest pose by alpha_legs * 0.5. Foot/ToeBase are
        left alone so the baked foot contact is not disturbed. Straightening the legs moves
        the feet, so the FK report at the end checks the toes still land at sole height
        (player/ground.ts: SOLE_BELOW_TOE = 0.0524); if they do not, lower --alpha-legs.

Fingers (version 3). GVHMR does not predict fingers, so a raw track freezes all 40 finger
bones and the hands read as plaster casts. No capture is retargeted onto them either, so
they are synthesized: the Talking clip's time-mean local rotation per bone (a natural
speaking hand; the bind pose is flat and stiff) plus a slow deterministic drift — spectral
noise brick-walled to 0.3-0.8 Hz, 3-6 deg peak per bone, fixed seed, seamless under looping
— and up to 4 deg of opening while the wrist moves fast. See finger_pass() for the recipe.

Clipping guard (version 2). The arm recentering is a constant rotation: it pulls every
frame closer to the body by the same amount, and on the Brunton track that pushed the
hands through the thighs/pelvis at 16 spots (frames where the raw GVHMR pose was 18 cm
clear of the body, measured). The guard therefore fades the arm recentering out exactly
where it would collide:

  1. Calibrate per-capsule "surface" distances from the asset's own capture clips: the
     closest any hand joint (wrist + finger joints 2-4) ever gets to each body axis
     (pelvis/torso/chest/thigh/shin capsules) in any approved clip. A pose closer than
     that is doing something no approved capture does.
  2. FK the recentered track per frame; frames where either hand's minimum distance ratio
     d/surface drops under 1.0 get that side's recentering weight ramped to 0 (the raw
     GVHMR pose there is measured safe), dilated by the smoothing radius, then Hann-
     smoothed (±9 frames) so the weight never steps. Iterate up to 4 times: each pass
     re-poses with the current weights and re-checks, so residual collisions widen the
     protected zone until clean. Frames still penetrating at weight 0 (the raw pose
     itself brought the hand in) get an escape rotation on the forearm — the hand is
     pushed out along the shortest escape direction, Hann-faded over the same window.
  3. The final FK report counts touch (d < surface) and deep (d < 0.7 * surface) frames;
     deep must be 0 or the run says so loudly.

Idempotent: the output carries a top-level "refined" marker and a second run refuses to
touch the file unless --force is given.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

MOTION_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE = MOTION_DIR / "reference" / "ref_means.json"
DEFAULT_AVATAR = MOTION_DIR.parent / "public" / "assets" / "model-clips.glb"

ARM_BONES = ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
             "RightShoulder", "RightArm", "RightForeArm", "RightHand"]
TORSO_BONES = ["Spine", "Spine1", "Spine2", "Neck", "Head"]
LEG_BONES = ["LeftUpLeg", "LeftLeg", "RightUpLeg", "RightLeg"]
TOES = ["LeftToeBase", "RightToeBase"]

# Clipping guard: hand joints probed per side, and the body approximated as capsules
# (bone axis endpoints, nominal radius). The surfaces actually used are calibrated from
# the asset's clips at runtime; the radii here are only a fallback if the glb has none.
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
HAND_POINTS = [f"{s}Hand" for s in ("Left", "Right")] + [
    f"{s}Hand{f}{k}" for s in ("Left", "Right") for f in FINGERS for k in (2, 3, 4)]
CAPSULES = {
    "pelvis": ("Hips", "Spine", 0.16),
    "torsoLow": ("Spine", "Spine1", 0.15),
    "torsoMid": ("Spine1", "Spine2", 0.15),
    "chest": ("Spine2", "Neck", 0.15),
    "thighL": ("LeftUpLeg", "LeftLeg", 0.09),
    "thighR": ("RightUpLeg", "RightLeg", 0.09),
    "shinL": ("LeftLeg", "LeftFoot", 0.07),
    "shinR": ("RightLeg", "RightFoot", 0.07),
}
GUARD_RADIUS = 9          # Hann half-window, frames (±0.3 s at 30 fps)
GUARD_ITERATIONS = 4

# Finger pass: GVHMR does not predict fingers, so a raw track freezes all 40 finger
# bones (measured 0-1.6 deg of travel where a forearm travels 65-71) and the hands read
# as plaster casts. No capture is retargeted onto them either; instead each finger bone
# gets the Talking clip's time-mean local rotation (the rig's "natural speaking hand";
# bind pose is flat and stiff) plus a slow drift:
FINGER_BONES = [f"{s}Hand{f}{k}" for s in ("Left", "Right") for f in FINGERS for k in (1, 2, 3, 4)]
FINGER_SEED = 20260921        # fixed: same input track -> same fingers
FINGER_AMP_DEG = (3.0, 6.0)   # per-bone peak of the smooth noise, uniform in this range
FINGER_FREQ_HZ = (0.3, 0.8)   # per-bone lowpass cutoff, uniform in this range
FINGER_OPEN_MAX_DEG = 4.0     # extra finger opening at the fastest wrist motion

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])  # xyzw


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, xyzw; broadcasts (...,4) x (...,4)."""
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def qinv(q: np.ndarray) -> np.ndarray:
    out = np.array(q, dtype=float, copy=True)
    out[..., :3] *= -1
    return out


def mean_quat(qs: np.ndarray) -> np.ndarray:
    """Sign-aligned running average: each frame is flipped into the hemisphere of the
    accumulated sum before being added, so the mean never crosses the double cover."""
    acc = np.zeros(4)
    for q in qs:
        acc += q if float(np.dot(q, acc)) >= 0 else -q
    return acc / np.linalg.norm(acc)


def slerp_to_identity(delta: np.ndarray, alpha: float) -> np.ndarray:
    """slerp(I, delta, alpha), taking the short way around."""
    d = np.array(delta, dtype=float)
    if d[3] < 0:
        d = -d
    return Slerp([0, 1], Rotation.from_quat([IDENTITY, d]))([alpha]).as_quat()[0]


def slerp_batch(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """slerp(q0, q1, t) for a fixed q0 (4,) and a batch q1 (F,4)."""
    out = np.array(q1, dtype=float, copy=True)
    dot = out @ q0
    out[dot < 0] *= -1
    dot = np.abs(dot).clip(max=1.0)
    theta = np.arccos(dot)
    small = theta < 1e-8
    s1 = np.where(small, t, np.sin(t * theta) / np.where(small, 1.0, np.sin(theta)))
    s0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / np.where(small, 1.0, np.sin(theta)))
    res = s0[:, None] * q0[None, :] + s1[:, None] * out
    return res / np.linalg.norm(res, axis=1, keepdims=True)


def angle_deg(q: np.ndarray) -> float:
    """Rotation angle of a unit quaternion (any hemisphere)."""
    return float(np.degrees(2 * np.arccos(min(1.0, abs(q[3])))))


def read_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    length = struct.unpack("<I", raw[12:16])[0]
    return json.loads(raw[20:20 + length]), raw[20 + length + 8:]


class Skeleton:
    """Node hierarchy of the glb: top-down order, parents, bind local TRS, name lookup."""

    def __init__(self, gltf: dict):
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
        self.nodes = nodes
        self.order = order
        self.parent = parent
        self.names = {i: nodes[i].get("name", f"node{i}") for i in order}
        self.name2idx = {v: k for k, v in self.names.items()}
        self.bind_rot = {i: np.array(nodes[i].get("rotation", IDENTITY), dtype=float) for i in order}
        self.bind_tr = {i: np.array(nodes[i].get("translation", [0.0, 0.0, 0.0]), dtype=float) for i in order}


def fk_world(skel: Skeleton, rot: dict[str, np.ndarray], tr: dict[str, np.ndarray],
             nframes: int, want_rot: bool = False):
    """FK the whole tree for nframes poses. rot/tr map bone name -> (F,4)/(F,3) local
    overrides; anything missing keeps its bind local transform. Returns name -> (F,3)
    world positions; with want_rot also name -> (F,3,3) world rotations."""
    wrot: dict[int, np.ndarray] = {}
    wpos: dict[int, np.ndarray] = {}
    out: dict[str, np.ndarray] = {}
    outr: dict[str, np.ndarray] = {}
    for i in skel.order:
        name = skel.names[i]
        lq = rot.get(name)
        if lq is None:
            lq = np.broadcast_to(skel.bind_rot[i], (nframes, 4))
        lt = tr.get(name)
        if lt is None:
            lt = np.broadcast_to(skel.bind_tr[i], (nframes, 3))
        lm = Rotation.from_quat(lq).as_matrix()
        p = skel.parent.get(i)
        if p is None:
            wrot[i] = lm
            wpos[i] = lt.copy()
        else:
            wrot[i] = wrot[p] @ lm
            wpos[i] = wpos[p] + np.einsum("fij,fj->fi", wrot[p], lt)
        out[name] = wpos[i]
        if want_rot:
            outr[name] = wrot[i]
    return (out, outr) if want_rot else out


def track_rot(track_bones: list[str], quats: np.ndarray) -> dict[str, np.ndarray]:
    return {b: quats[:, k] for k, b in enumerate(track_bones)}


def accessor_array(gltf: dict, bin_: bytes, idx: int) -> np.ndarray:
    acc = gltf["accessors"][idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    comp = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}[acc["componentType"]]
    ncomp = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[acc["type"]]
    off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    a = np.frombuffer(bin_, dtype=comp, count=acc["count"] * ncomp, offset=off)
    return a.reshape(acc["count"], ncomp) if ncomp > 1 else a


def clip_locals(gltf: dict, bin_: bytes, clip: dict, fps: int = 30
                ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Sample a glb animation's rotation+translation channels onto a uniform fps grid."""
    nodes = gltf["nodes"]
    dur = 0.0
    channels = []
    for ch in clip["channels"]:
        s = clip["samplers"][ch["sampler"]]
        times = accessor_array(gltf, bin_, s["input"]).astype(float)
        vals = accessor_array(gltf, bin_, s["output"]).astype(float)
        node = ch["target"].get("node")
        if node is None or ch["target"]["path"] not in ("rotation", "translation"):
            continue
        dur = max(dur, float(times[-1]))
        channels.append((nodes[node].get("name"), ch["target"]["path"], times, vals))
    n = int(round(dur * fps)) + 1
    grid = np.arange(n) / fps

    def sample(times: np.ndarray, vals: np.ndarray) -> np.ndarray:
        idx = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(times) - 2)
        t0, t1 = times[idx], times[idx + 1]
        f = np.clip(np.where(t1 > t0, (grid - t0) / np.maximum(t1 - t0, 1e-9), 0.0), 0, 1)[:, None]
        v0, v1 = vals[idx], vals[idx + 1]
        if vals.shape[1] == 4:  # quaternion: sign-align then nlerp
            v1 = np.where((np.sum(v0 * v1, axis=1) < 0)[:, None], -v1, v1)
            out = v0 * (1 - f) + v1 * f
            return out / np.linalg.norm(out, axis=1, keepdims=True)
        return v0 * (1 - f) + v1 * f

    rot: dict[str, np.ndarray] = {}
    tr: dict[str, np.ndarray] = {}
    for name, path, times, vals in channels:
        if name is None:
            continue
        (rot if path == "rotation" else tr)[name] = sample(times, vals)
    return rot, tr, n


def point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(F,3) points, (F,3) segment ends -> (F,) distances."""
    ab = b - a
    t = np.sum((p - a) * ab, axis=1, keepdims=True) / np.maximum(np.sum(ab * ab, axis=1, keepdims=True), 1e-12)
    closest = a + np.clip(t, 0, 1) * ab
    return np.linalg.norm(p - closest, axis=1)


def calibrated_surfaces(gltf: dict, bin_: bytes, skel: Skeleton) -> dict[str, float]:
    """Per-capsule surface distance: the closest any hand joint ever comes to each body
    axis across every capture clip baked into the asset. Those clips are owner-approved,
    so this is the empirical 'still not clipping' boundary for this rig and wardrobe."""
    surfaces = {cap: float("inf") for cap in CAPSULES}
    for clip in gltf.get("animations", []):
        rot, tr, n = clip_locals(gltf, bin_, clip)
        pos = fk_world(skel, rot, tr, n)
        for cap, (a, b, r) in CAPSULES.items():
            for hp in HAND_POINTS:
                if hp in pos and a in pos and b in pos:
                    surfaces[cap] = min(surfaces[cap], float(point_seg_dist(pos[hp], pos[a], pos[b]).min()))
    return {cap: (d if np.isfinite(d) else CAPSULES[cap][2]) for cap, d in surfaces.items()}


def side_ratios(pos: dict[str, np.ndarray], surfaces: dict[str, float]) -> dict[str, np.ndarray]:
    """Per frame, per side: the minimum d/surface ratio over that hand's joints and all
    body capsules. < 1 means closer to the body than any approved capture ever gets."""
    out: dict[str, np.ndarray] = {}
    for side in ("Left", "Right"):
        r = None
        for hp in HAND_POINTS:
            if not hp.startswith(side) or hp not in pos:
                continue
            for cap, (a, b, _r) in CAPSULES.items():
                d = point_seg_dist(pos[hp], pos[a], pos[b]) / surfaces[cap]
                r = d if r is None else np.minimum(r, d)
        out[side] = r
    return out


def penetration_counts(pos: dict[str, np.ndarray], surfaces: dict[str, float]) -> tuple[int, int]:
    """Frames with any hand joint closer than the calibrated surface, and closer than
    0.7x it (deep)."""
    touch = deep = None
    for hp in HAND_POINTS:
        if hp not in pos:
            continue
        for cap, (a, b, _r) in CAPSULES.items():
            d = point_seg_dist(pos[hp], pos[a], pos[b])
            t, dp = d < surfaces[cap], d < surfaces[cap] * 0.7
            touch = t if touch is None else touch | t
            deep = dp if deep is None else deep | dp
    return int(touch.sum()), int(deep.sum())


def hann_smooth(x: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return x
    win = np.hanning(2 * radius + 1)
    win /= win.sum()
    return np.convolve(np.pad(x, radius, mode="edge"), win, mode="valid")


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    from scipy.ndimage import maximum_filter
    return maximum_filter(mask.astype(float), size=2 * radius + 1) > 0


def slerp_to_identity_batch(delta: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """slerp(I, delta, a) for per-frame alphas (F,) -> (F,4)."""
    d = np.array(delta, dtype=float)
    if d[3] < 0:
        d = -d
    theta = 2.0 * np.arccos(np.clip(d[3], -1.0, 1.0))
    s = np.sin(theta / 2.0)
    axis = d[:3] / s if s > 1e-12 else np.array([0.0, 0.0, 0.0])
    half = 0.5 * alphas * theta
    return np.stack([axis[0] * np.sin(half), axis[1] * np.sin(half),
                     axis[2] * np.sin(half), np.cos(half)], axis=-1)


def finger_pass(quats: np.ndarray, idx: dict[str, int], ref_means: dict[str, list[float]],
                fps: float, report: list[str]) -> dict | None:
    """Replace the frozen finger series with the Talking mean pose plus a slow drift.

    Overwrites rather than adds, so re-running on an already-fingered track is a no-op
    by construction. Deterministic: one fixed seed drives amplitude, cutoff and the
    noise itself, so the same input track yields byte-identical fingers.

    Per bone: q(t) = delta(t) x base, where base is the clip's time-mean local rotation
    and delta(t) is a small rotation around the base's own curl axis. The drift is
    spectral, not filtered-after-the-fact: white noise is FFTed, every bin above the
    bone's cutoff (0.3-0.8 Hz) is zeroed, and the inverse FFT gives a drift with exactly
    no energy above 0.8 Hz — and, being circular, no seam when the track loops.
    On top of the drift the fingers open up to FINGER_OPEN_MAX_DEG when the wrist moves
    fast (gesturing hands open, resting hands curl), a linear coupling between the
    wrist's median and p95 angular speed, brick-walled into the same band.
    """
    present = [b for b in FINGER_BONES if b in idx and b in ref_means]
    missing = [b for b in FINGER_BONES if b not in present]
    if not present:
        report.append("  finger pass skipped: no finger bones in track or reference")
        return None
    if missing:
        report.append(f"  finger pass: {len(missing)} bones missing ({missing[:3]}...)")

    frames = quats.shape[0]
    rng = np.random.default_rng(FINGER_SEED)

    def brickwall(x: np.ndarray, fc: float) -> np.ndarray:
        """Zero every FFT bin above fc Hz; the inverse transform is circular, which also
        makes the result seamless under the player's LoopRepeat."""
        X = np.fft.rfft(x)
        X[np.fft.rfftfreq(len(x), 1.0 / fps) > fc] = 0.0
        return np.fft.irfft(X, len(x))

    def lowpass_noise(fc: float) -> np.ndarray:
        out = brickwall(rng.standard_normal(frames), fc)
        return out / np.abs(out).max()

    # wrist angular speed per side (deg/s); constant left-multiplied corrections don't
    # change it, so measuring before the arm recenter is exact.
    open_deg: dict[str, np.ndarray] = {}
    for side in ("Left", "Right"):
        hand = f"{side}Hand"
        if hand not in idx:
            continue
        q = quats[:, idx[hand]]
        dq = qmul(qinv(q[:-1]), q[1:])
        w = np.degrees(2.0 * np.arccos(np.clip(np.abs(dq[:, 3]), 0.0, 1.0))) * fps
        w = brickwall(np.concatenate([[w[0]], w]), 0.8)
        w50, w95 = np.percentile(w, 50), np.percentile(w, 95)
        # the clip is nonlinear and reintroduces high frequencies at the onset kink,
        # so brickwall again after it
        open_deg[side] = brickwall(
            np.clip((w - w50) * FINGER_OPEN_MAX_DEG / max(w95 - w50, 1e-6),
                    0.0, FINGER_OPEN_MAX_DEG), 0.8)

    for b in present:
        base = np.array(ref_means[b], dtype=float)
        axis = base[:3]
        n = np.linalg.norm(axis)
        axis = axis / n if n > 1e-3 else np.array([1.0, 0.0, 0.0])
        amp = np.radians(FINGER_AMP_DEG[0] + (FINGER_AMP_DEG[1] - FINGER_AMP_DEG[0]) * rng.random())
        fc = FINGER_FREQ_HZ[0] + (FINGER_FREQ_HZ[1] - FINGER_FREQ_HZ[0]) * rng.random()
        noise = lowpass_noise(fc)
        side = "Left" if b.startswith("Left") else "Right"
        theta = amp * noise - np.radians(open_deg.get(side, 0.0))
        delta = Rotation.from_rotvec(axis[None, :] * theta[:, None]).as_quat()
        quats[:, idx[b]] = qmul(delta, np.broadcast_to(base, (frames, 4)))

    report.append(f"  finger pass: {len(present)} bones, Talking-mean base + "
                  f"{FINGER_AMP_DEG[0]:g}-{FINGER_AMP_DEG[1]:g} deg drift at "
                  f"{FINGER_FREQ_HZ[0]:g}-{FINGER_FREQ_HZ[1]:g} Hz, wrist-open "
                  f"{FINGER_OPEN_MAX_DEG:g} deg, seed {FINGER_SEED}")
    return {"version": 1, "seed": FINGER_SEED, "bones": len(present),
            "amp_deg": list(FINGER_AMP_DEG), "freq_hz": list(FINGER_FREQ_HZ),
            "wrist_open_deg": FINGER_OPEN_MAX_DEG}


def bind_locals(skel: Skeleton) -> dict[str, np.ndarray]:
    """Per-node bind local rotation (xyzw) by name."""
    return {skel.names[i]: skel.bind_rot[i] for i in skel.order}


def toe_world_y(skel: Skeleton, bones: list[str], quats: np.ndarray, stride: int = 10) -> np.ndarray:
    """FK the skeleton on a strided frame subset; (2, S) world Y of the toe bones."""
    frames = quats[::stride]
    pos = fk_world(skel, track_rot(bones, frames), {}, len(frames))
    return np.array([pos[t][:, 1] for t in TOES])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("track", type=Path)
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument("--avatar", type=Path, default=DEFAULT_AVATAR)
    ap.add_argument("--alpha-arms", type=float, default=1.0,
                    help="recentering strength for shoulder/arm/forearm/hand (1 = land on the reference mean)")
    ap.add_argument("--alpha-hips", type=float, default=1.0,
                    help="recentering strength for the hips")
    ap.add_argument("--alpha-hips-dynamic", type=float, default=0.4,
                    help="fraction of the hips' dynamic component kept (0.4 kills the twisting)")
    ap.add_argument("--alpha-torso", type=float, default=1.0,
                    help="recentering strength for spine/neck/head; must follow the hips "
                         "(the track's spine compensates the hips' zero-convention offset)")
    ap.add_argument("--alpha-legs", type=float, default=1.5,
                    help="convergence of upleg/leg toward the rest pose; applied at half strength. "
                         "Measured on the Brunton profile: stronger convergence grounds the feet "
                         "better (the asymmetry came from the GVHMR legs, not from the rest pose), "
                         "so 1.5 -- nearly at rest, a little residual knee life")
    ap.add_argument("--no-fingers", action="store_true",
                    help="skip the finger pass (keep the track's frozen finger series)")
    ap.add_argument("--no-guard", action="store_true",
                    help="skip the clipping guard (v1 behavior: constant arm recentering)")
    ap.add_argument("--force", action="store_true", help="re-refine a track that already carries the marker")
    ap.add_argument("--out", type=Path, help="default: overwrite the input in place (atomically)")
    args = ap.parse_args()

    track = json.loads(args.track.read_text())
    if track.get("refined") and not args.force:
        print(f"refusing: {args.track} already carries a 'refined' marker "
              f"(version {track['refined'].get('version')}); pass --force to redo")
        return
    ref = json.loads(args.reference.read_text())
    gltf, bin_ = read_glb(args.avatar)
    skel = Skeleton(gltf)
    bind_rot = bind_locals(skel)

    bones: list[str] = track["bones"]
    idx = {b: k for k, b in enumerate(bones)}
    quats = np.array(track["quats"], dtype=float).reshape(track["frames"], len(bones), 4)
    ref_means: dict[str, list[float]] = ref["means"]

    print(f"{args.track.name}: {track['frames']} frames x {len(bones)} bones, "
          f"reference={ref['source']}:{ref['clip']}")

    report: list[str] = []

    def recenter(bone: str, alpha: float) -> np.ndarray | None:
        """The constant left-multiplied rotation that lands the bone's mean on the reference."""
        delta = recenter_delta(bone)
        if delta is None:
            return None
        corr = slerp_to_identity(delta, alpha)
        report.append(f"  {bone:15s} recenter {angle_deg(delta):6.1f} deg, applied x{alpha:.2f}")
        return corr

    def recenter_delta(bone: str) -> np.ndarray | None:
        """The full recentering rotation for a bone, before any strength scaling."""
        if bone not in idx or bone not in ref_means:
            report.append(f"  {bone:15s} skipped (missing in track or reference)")
            return None
        track_mean = mean_quat(quats[:, idx[bone]])
        return qmul(np.array(ref_means[bone]), qinv(track_mean))

    # 1. hips + torso chain: recenter together (see docstring), hips dynamics also shrunk.
    if "Hips" in idx and "Hips" in ref_means:
        hips_mean = mean_quat(quats[:, idx["Hips"]])
        corr = recenter("Hips", args.alpha_hips)
        if corr is not None:
            shrunk = slerp_batch(hips_mean, quats[:, idx["Hips"]], args.alpha_hips_dynamic)
            quats[:, idx["Hips"]] = qmul(corr, shrunk)
    for bone in TORSO_BONES:
        corr = recenter(bone, args.alpha_torso)
        if corr is not None:
            quats[:, idx[bone]] = qmul(corr, quats[:, idx[bone]])

    # 2. legs: halve the dynamics, then converge toward the rig's rest pose.
    for bone in LEG_BONES:
        if bone not in idx:
            continue
        series = quats[:, idx[bone]]
        shrunk = slerp_batch(mean_quat(series), series, 0.5)
        rest = bind_rot.get(bone, IDENTITY)
        quats[:, idx[bone]] = slerp_batch(rest, shrunk, 1.0 - args.alpha_legs * 0.5)
        report.append(f"  {bone:15s} dynamics x0.5, rest-converge x{args.alpha_legs * 0.5:.2f}")

    # 3. fingers: the track's 40 finger bones are frozen (GVHMR has no hand keypoints).
    #    Replace with the Talking-mean base pose plus a slow deterministic drift. Before
    #    the arms/guard so the guard's FK sees the final finger pose.
    fingers = None if args.no_fingers else finger_pass(quats, idx, ref_means, track["fps"], report)

    # 4. arms: recenter onto the reference mean, dynamics untouched. With the guard on,
    #    the per-frame recentering weight fades to 0 wherever the recentered pose would
    #    put a hand closer to the body than any capture clip ever comes (see docstring).
    arm_orig = {b: quats[:, idx[b]].copy() for b in ARM_BONES if b in idx}
    arm_delta = {b: recenter_delta(b) for b in arm_orig}
    arm_delta = {b: d for b, d in arm_delta.items() if d is not None}
    for b, d in arm_delta.items():
        report.append(f"  {b:15s} recenter {angle_deg(d):6.1f} deg, applied x{args.alpha_arms:.2f}"
                      + ("" if args.no_guard else " (guarded)"))

    guard = None
    surfaces: dict[str, float] = {}
    if not args.no_guard and arm_delta and gltf.get("animations"):
        surfaces = calibrated_surfaces(gltf, bin_, skel)
        report.append("  guard surfaces (m): "
                      + ", ".join(f"{c}={s:.3f}" for c, s in surfaces.items()))
        frames = track["frames"]
        zero = {"Left": np.zeros(frames, bool), "Right": np.zeros(frames, bool)}
        weight = {"Left": np.ones(frames), "Right": np.ones(frames)}
        side_of = {b: ("Left" if b.startswith("Left") else "Right") for b in arm_delta}

        def apply_arms() -> None:
            for b, d in arm_delta.items():
                quats[:, idx[b]] = qmul(slerp_to_identity_batch(d, args.alpha_arms * weight[side_of[b]]),
                                        arm_orig[b])

        # Ramp pass: fade the recentering out wherever it collides. A penetrating frame
        # is protected until its weight is exactly 0 (raw pose); frames on a ramp with
        # partial weight still count as fresh, so zones widen until clean or stuck.
        stuck = {"Left": np.zeros(frames, bool), "Right": np.zeros(frames, bool)}
        iteration = 0
        for iteration in range(GUARD_ITERATIONS):
            apply_arms()
            pos = fk_world(skel, track_rot(bones, quats), {}, frames)
            ratios = side_ratios(pos, surfaces)
            grew = False
            for side in ("Left", "Right"):
                penetrating = ratios[side] < 1.0
                fresh = penetrating & (weight[side] > 1e-3)
                if fresh.any():
                    grew = True
                    zero[side] |= penetrating
                zone = dilate(zero[side], GUARD_RADIUS + 1)
                weight[side] = hann_smooth(1.0 - zone.astype(float), GUARD_RADIUS)
                stuck[side] = penetrating & (weight[side] <= 1e-3)
            report.append(f"  guard pass {iteration}: protected frames "
                          f"L={int(zero['Left'].sum())} R={int(zero['Right'].sum())}, "
                          f"stuck L={int(stuck['Left'].sum())} R={int(stuck['Right'].sum())}")
            if not grew:
                break
        apply_arms()

        # Escape pass: frames still penetrating at weight 0 (the raw pose itself, or the
        # recentered torso/legs, brought the hand in). Rotate the forearm so the hand
        # leaves along the shortest escape direction, Hann-faded over the same window.
        escaped = 0
        for _ in range(2):
            if not (stuck["Left"].any() or stuck["Right"].any()):
                break
            pos, rots = fk_world(skel, track_rot(bones, quats), {}, frames, want_rot=True)
            for side in ("Left", "Right"):
                if not stuck[side].any():
                    continue
                fore, hand = f"{side}ForeArm", f"{side}Hand"
                # per frame: worst point, its capsule's closest point, escape direction
                depth = np.zeros(frames)
                esc = np.zeros((frames, 3))
                for hp in HAND_POINTS:
                    if not hp.startswith(side) or hp not in pos:
                        continue
                    for cap, (a, b, _r) in CAPSULES.items():
                        ab = pos[b] - pos[a]
                        t = np.clip(np.sum((pos[hp] - pos[a]) * ab, axis=1, keepdims=True)
                                    / np.maximum(np.sum(ab * ab, axis=1, keepdims=True), 1e-12), 0, 1)
                        c = pos[a] + t * ab
                        d = np.linalg.norm(pos[hp] - c, axis=1)
                        need = surfaces[cap] * 1.05 - d
                        better = need > depth
                        depth = np.where(better, need, depth)
                        esc = np.where(better[:, None], pos[hp] - c, esc)
                lever = np.linalg.norm(pos[hand] - pos[fore], axis=1)
                theta = np.clip(depth / np.maximum(lever, 0.05), 0, 0.5)
                theta[~dilate(stuck[side], GUARD_RADIUS + 1)] = 0.0
                theta = hann_smooth(theta, GUARD_RADIUS)
                if not (theta > 1e-4).any():
                    continue
                # rotation axis: constant per contiguous interval, from its deepest frame
                axis = np.zeros((frames, 3))
                v = pos[hand] - pos[fore]
                idxs = np.nonzero(theta > 1e-4)[0]
                if len(idxs):
                    groups = np.split(idxs, np.nonzero(np.diff(idxs) > 1)[0] + 1)
                    for g in groups:
                        w = g[int(np.argmax(theta[g]))]
                        a3 = np.cross(v[w], esc[w])
                        n = np.linalg.norm(a3)
                        axis[g] = a3 / n if n > 1e-9 else np.array([1.0, 0, 0])
                rotvec = axis * theta[:, None]
                Rw = Rotation.from_rotvec(rotvec).as_matrix()          # world-space delta
                gp = rots[skel.names[skel.parent[skel.name2idx[fore]]]]  # parent global rot
                Rl = np.transpose(gp, (0, 2, 1)) @ Rw @ gp             # into parent frame
                ql = Rotation.from_matrix(Rl).as_quat()
                quats[:, idx[fore]] = qmul(ql, quats[:, idx[fore]])
                escaped += int((theta > 1e-4).sum())
            pos = fk_world(skel, track_rot(bones, quats), {}, frames)
            ratios = side_ratios(pos, surfaces)
            for side in ("Left", "Right"):
                stuck[side] = ratios[side] < 1.0
        guard = {"version": 1, "window": GUARD_RADIUS, "iterations": iteration + 1,
                 "protected_frames_left": int(zero["Left"].sum()),
                 "protected_frames_right": int(zero["Right"].sum()),
                 "escape_frames": escaped}
    else:
        for b, d in arm_delta.items():
            quats[:, idx[b]] = qmul(slerp_to_identity(d, args.alpha_arms), arm_orig[b])
        if not args.no_guard:
            report.append("  guard skipped: avatar asset has no capture clips to calibrate against")

    quats /= np.linalg.norm(quats, axis=2, keepdims=True)

    # FK sanity: the toes must stay at sole height, or the runtime ground() correction has to
    # haul the whole avatar and the stance reads as floating/sunken.
    stride = max(1, track["frames"] // 500)
    toe_before = toe_world_y(skel, bones, np.array(track["quats"], dtype=float).reshape(track["frames"], len(bones), 4), stride)
    toe_after = toe_world_y(skel, bones, quats, stride)
    report.append(f"  toe world Y: before mean={toe_before.mean():.3f} m, "
                  f"after mean={toe_after.mean():.3f} m (sole at 0.052)")

    # Guard verification on the final track: deep penetration must be zero, touch frames
    # should be at the raw track's level (its hands-safely-clear poses are what the guard
    # falls back to).
    if surfaces:
        pos = fk_world(skel, track_rot(bones, quats), {}, track["frames"])
        touch, deep = penetration_counts(pos, surfaces)
        pos_raw = fk_world(skel, track_rot(bones, np.array(track["quats"], dtype=float)
                                           .reshape(track["frames"], len(bones), 4)), {}, track["frames"])
        touch_raw, deep_raw = penetration_counts(pos_raw, surfaces)
        report.append(f"  guard verify: touch {touch} (raw {touch_raw}), deep {deep} (raw {deep_raw}) "
                      f"of {track['frames']} frames"
                      + ("  <-- DEEP PENETRATION REMAINS" if deep else ""))
        guard = guard or {}
        guard.update({"touch_frames": touch, "deep_frames": deep})

    out = args.out or args.track
    payload = {
        **track,
        "quats": np.round(quats, 4).reshape(-1).tolist(),
        "refined": {
            "version": 3,
            "alpha_arms": args.alpha_arms,
            "alpha_hips": args.alpha_hips,
            "alpha_hips_dynamic": args.alpha_hips_dynamic,
            "alpha_torso": args.alpha_torso,
            "alpha_legs": args.alpha_legs,
            "reference": f"{ref['source']}:{ref['clip']}",
            "fingers": fingers,
            "guard": guard,
        },
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out)
    print("\n".join(report))
    print(f"  -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
