"""First-principles audit of a track as the player shows it: does the body obey contact and gravity?

    .venv/bin/python physics_audit.py <model-clips.glb> <track.json | clip:<name>> [...]

check_track.py gates per-frame symptoms (skate cm/frame, angular velocity). This measures
the three physical laws a viewer's eye checks, on the pose the runtime actually renders:
the track's rotations (and Hips translation, when the track carries one) FK'd on the glb
skeleton, then the whole body shifted vertically so the lower sole touches the floor, as
player/ground.ts does every frame.

1. **A planted foot does not move.** Friction holds a loaded foot still; the body moves
   over it. Per stance (a maximal run with the same foot lower and slower than 1.5
   cm/frame) the planted foot's horizontal travel from where it landed. Reported as the
   p95 / max over stances, and total skate distance per minute.
2. **Legs carry the body; the body does not carry the legs.** With the pelvis pinned in
   space, a swing leg lifting and landing goes nowhere: a step in place, or the planted
   foot dragged under a still pelvis. Counted: swing events (the upper toe rises > 3 cm
   above the lower) per minute.
3. **Gravity is the only vertical force besides the floor.** The floor can push, not pull,
   so the CoM's vertical acceleration is at least -g, and a person standing and talking
   keeps it within a few % of g. The ground shift (root Y) is added here exactly as the
   runtime adds it, so a bob it introduces shows up as CoM acceleration. The zero-moment
   point, zmp = com_xz - com_acc_xz * h / g, is where the floor has to push to produce the
   observed horizontal acceleration; outside the feet the floor cannot push there, so the
   motion is not physically producible. Reported: share of frames with the ZMP more than 2
   cm outside the support polygon, and the worst excursion.

Plus torso lean (hips to head against vertical), since a lean with the CoM still inside the
feet can still read as tipping.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from refine_track import Skeleton, clip_locals, fk_world, read_glb, track_rot  # noqa: E402
from balance import FEET, com_track, cp_distance, support_polygon  # noqa: E402

SOLE_BELOW_TOE = 0.0524
G = 9.81
PLANT_SPEED = 0.015      # m/frame: slower than this and lower = planted (check_track.py)
SWING_LIFT = 0.03        # m above the lower toe = the foot is off the floor
ZMP_MARGIN = 0.02


def load_track(glb_path: Path, spec: str):
    gltf, bin_ = read_glb(glb_path)
    skel = Skeleton(gltf)
    if spec.startswith("clip:"):
        clip = next(c for c in gltf["animations"] if c["name"] == spec[5:])
        rot, tr, n = clip_locals(gltf, bin_, clip)
        return skel, rot, tr, n, 30.0
    t = json.loads(Path(spec).read_text())
    q = np.array(t["quats"], dtype=float).reshape(t["frames"], len(t["bones"]), 4)
    q /= np.linalg.norm(q, axis=2, keepdims=True)
    tr = {}
    if t.get("hips"):
        tr["Hips"] = np.array(t["hips"], dtype=float).reshape(t["frames"], 3)
    return skel, track_rot(t["bones"], q), tr, t["frames"], float(t["fps"])


def as_played(skel, rot, tr, n):
    """FK, then the runtime's vertical ground shift (lower sole on y = 0)."""
    pos = fk_world(skel, rot, tr, n)
    toes = np.stack([pos[toe][:, 1] for _, toe in FEET], axis=1)
    shift = -(toes.min(axis=1) - SOLE_BELOW_TOE)
    return {k: v + np.stack([np.zeros(n), shift, np.zeros(n)], axis=1) for k, v in pos.items()}, shift


def stances(pos, n):
    toes = [pos[toe] for _, toe in FEET]
    lower = np.argmin(np.stack([t[:, 1] for t in toes], axis=1), axis=1)
    out = []
    f = 0
    while f < n:
        s = lower[f]
        e = f
        while e + 1 < n and lower[e + 1] == s:
            e += 1
        xz = toes[s][f:e + 1][:, [0, 2]]
        if e > f:
            out.append((s, f, e, float(np.linalg.norm(xz - xz[0], axis=1).max())))
        f = e + 1
    return out, lower


def audit(glb: Path, spec: str) -> dict:
    skel, rot, tr, n, fps = load_track(glb, spec)
    pos, shift = as_played(skel, rot, tr, n)
    minutes = n / fps / 60
    toes = np.stack([pos[toe][:, 1] for _, toe in FEET], axis=1)

    st, lower = stances(pos, n)
    travel = np.array([d for *_, d in st]) if st else np.zeros(1)
    lower_xz = np.stack([pos[FEET[s][1]][f, [0, 2]] for f, s in enumerate(lower)])
    step = np.linalg.norm(np.diff(lower_xz, axis=0), axis=1)
    same = lower[1:] == lower[:-1]
    skate_m = float(step[same & (step < PLANT_SPEED)].sum())

    lifted = (toes.max(axis=1) - toes.min(axis=1)) > SWING_LIFT
    swings = int(np.sum(lifted[1:] & ~lifted[:-1]))

    com = com_track(pos)
    dt = 1 / fps
    k = max(1, int(round(fps / 15)))            # 15 Hz derivative, below the mocap jitter band
    vel = (com[2 * k:] - com[:-2 * k]) / (2 * k * dt)
    acc = (com[2 * k:] - 2 * com[k:-k] + com[:-2 * k]) / (k * dt) ** 2
    h = np.maximum(com[k:-k, 1], 0.3)
    zmp = com[k:-k][:, [0, 2]] - acc[:, [0, 2]] * (h / G)[:, None]
    exc = np.array([cp_distance(zmp[i], support_polygon(pos, i + k, 0.0)) for i in range(len(zmp))])

    lean = pos["Head"] - pos["Hips"]
    lat = np.degrees(np.arctan2(np.abs(lean[:, 0]), lean[:, 1]))
    fwd = np.degrees(np.arctan2(np.abs(lean[:, 2]), lean[:, 1]))
    hips = pos["Hips"][:, [0, 2]]
    return {
        "name": spec if spec.startswith("clip:") else Path(spec).name,
        "seconds": n / fps,
        "stance_travel_p95_cm": float(np.percentile(travel, 95) * 100),
        "stance_travel_max_cm": float(travel.max() * 100),
        "skate_cm_per_min": skate_m * 100 / minutes,
        "swings_per_min": swings / minutes,
        "root_shift_range_cm": float((shift.max() - shift.min()) * 100),
        "com_vert_acc_p99_g": float(np.percentile(np.abs(acc[:, 1]), 99) / G),
        "com_vert_acc_max_g": float(np.abs(acc[:, 1]).max() / G),
        "zmp_outside_pct": float(100 * np.mean(exc > ZMP_MARGIN)),
        "zmp_excursion_max_cm": float(exc.max() * 100),
        "hips_xz_range_cm": float(np.ptp(hips, axis=0).max() * 100),
        "lean_lat_p95_deg": float(np.percentile(lat, 95)),
        "lean_fwd_p95_deg": float(np.percentile(fwd, 95)),
    }


ROWS = [
    ("stance_travel_p95_cm", "planted-foot travel per stance, p95 (cm)"),
    ("stance_travel_max_cm", "planted-foot travel per stance, max (cm)"),
    ("skate_cm_per_min", "total planted-foot skate (cm/min)"),
    ("swings_per_min", "foot lifts (steps) per minute"),
    ("root_shift_range_cm", "runtime ground shift range (cm)"),
    ("com_vert_acc_p99_g", "CoM vertical accel p99 (g)"),
    ("com_vert_acc_max_g", "CoM vertical accel max (g)"),
    ("zmp_outside_pct", "ZMP > 2 cm outside feet (% frames)"),
    ("zmp_excursion_max_cm", "ZMP worst excursion (cm)"),
    ("hips_xz_range_cm", "pelvis horizontal range (cm)"),
    ("lean_lat_p95_deg", "torso lean lateral p95 (deg)"),
    ("lean_fwd_p95_deg", "torso lean forward p95 (deg)"),
]


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        raise SystemExit(2)
    glb = Path(args[0])
    results = [audit(glb, s) for s in args[1:]]
    width = max(len(r["name"]) for r in results)
    width = max(width, 12)
    print(f"{'':44s}" + "".join(f"{r['name'][:width]:>{width + 2}s}" for r in results))
    for key, label in ROWS:
        print(f"{label:44s}" + "".join(f"{r[key]:>{width + 2}.2f}" for r in results))


if __name__ == "__main__":
    main()
