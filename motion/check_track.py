"""Quality gate for the final playback artifact: the refined track.json the player serves.

Every other check in the profile pipeline fires on the upstream npz (verify_retarget,
metrics.py); refine_track.py and its clipping guard then rewrite the track, and nothing
verified the result. This is that verification. It FKs the exact quaternion data of
track.json on the exact glb skeleton — no hip translation, bind translations, the same
convention as player/avatar.ts trackToClip — so what is measured is what plays.

    .venv/bin/python check_track.py <model-clips.glb> <track.json> [--check]
        [--foot-skate-max-cm X] ... (gate overrides in metrics.py style)

Four checks:

- **clipping**: hand joints against the body capsules, surfaces calibrated from the
  asset's own capture clips (same machinery and same口径 as refine_track.py's guard:
  touch = closer than any approved capture ever comes, deep = closer than 0.7x that).
  Gates: deep frames = 0, touch frames <= 20. The refined Brunton track measures 0/0;
  the failed v1 guard (constant arm recentering) measured 96 deep frames.
- **foot float**: per-frame world Y of the lower ToeBase. The runtime (player/ground.ts)
  presses the lower foot's sole to the floor — the toe sits at SOLE_BELOW_TOE = 0.0524 —
  so the raw FK height of the lower toe should sit near that: gate the mean inside
  [0.02, 0.08] m. The refined Brunton track measures 0.0437. This catches a systematic
  "balloon" offset; per-frame spikes are the runtime's job, not this gate's.
- **foot skate**: a foot is planted in a frame when it is the lower one and its horizontal
  speed is under 1.5 cm/frame (45 cm/s — the built-in clips' lower-foot speed p99 is
  0.084 cm/frame, so the threshold cleanly separates stance from swing). Gate: mean
  horizontal displacement while planted <= 0.25 cm/frame. That limit is calibrated on the
  asset's 9 built-in clips (freshly foot-locked): their per-clip planted means top out at
  0.031 cm/frame and the worst per-clip planted p95 is 0.106 (Talking, the one clip with
  real weight shifts) — the gate is ~2.4x that p95. Rerun with --calibrate to see the
  baseline printed. Note the track has no hip translation, so this is pure-rotation FK,
  consistent with the runtime — and body sway through the hips moves both feet, which is
  why a track's mean (refined Brunton: 0.104) sits above the capture clips' without
  anything sliding.
- **teleport / seizure**: adjacent-frame quaternion angular velocity per bone, deg/s.
  Gate: max <= 600 (the refined Brunton track's max is 429; the capture clips sit far
  below). p99 is reported for context.

All metrics print always; --check lists the FAIL lines and exits nonzero.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from refine_track import (  # noqa: E402
    Skeleton, calibrated_surfaces, clip_locals, fk_world, penetration_counts,
    qinv, qmul, read_glb, track_rot, TOES,
)

SOLE_BELOW_TOE = 0.0524          # player/ground.ts: toe bone sits this far above the sole
PLANT_MAX_SPEED_CM = 1.5         # planted = lower foot AND horizontal speed under this

GATES = {
    "deep_frames_max": 0.0,
    "touch_frames_max": 20.0,
    "foot_float_mean_min_cm": 2.0,
    "foot_float_mean_max_cm": 8.0,
    "foot_skate_max_cm": 0.25,   # 2x the built-in-clips baseline p95 (see docstring)
    "angvel_max_deg_s": 600.0,
}


def gate_overrides(argv: list[str]) -> list[str]:
    """--<gate> X overrides GATES (key dashes, e.g. --foot-skate-max-cm 0.2), same style
    as metrics.py. Returns the remaining (positional) args."""
    positional = []
    i = 0
    while i < len(argv):
        a = argv[i]
        key = a[2:].replace("-", "_") if a.startswith("--") else ""
        if key in GATES and i + 1 < len(argv):
            try:
                GATES[key] = float(argv[i + 1])
                i += 2
                continue
            except ValueError:
                pass
        if not a.startswith("--"):
            positional.append(a)
        i += 1
    return positional


def foot_stats(pos: dict[str, np.ndarray], nframes: int) -> dict[str, float]:
    """Lower-toe height and planted-foot horizontal drift from FK world positions."""
    ys = np.array([pos[t][:, 1] for t in TOES])                    # (2, F)
    xz = np.array([pos[t][:, [0, 2]] for t in TOES])               # (2, F, 2)
    lower_y = ys.min(axis=0)
    speed = np.linalg.norm(np.diff(xz, axis=1), axis=2) * 100      # (2, F-1) cm/frame
    lower = ys[:, :-1].argmin(axis=0)
    planted = np.concatenate([
        speed[s][(lower == s) & (speed[s] < PLANT_MAX_SPEED_CM)] for s in (0, 1)
    ])
    return {
        "float_mean_m": float(lower_y.mean()),
        "float_max_m": float(lower_y.max()),
        "planted_frames": float(planted.size),
        "skate_mean_cm": float(planted.mean()) if planted.size else 0.0,
        "skate_p95_cm": float(np.percentile(planted, 95)) if planted.size else 0.0,
        "skate_max_cm": float(planted.max()) if planted.size else 0.0,
    }


def angvel_stats(quats: np.ndarray, fps: float) -> dict[str, float]:
    """Adjacent-frame angular velocity (deg/s) over all bones: p99 and max."""
    dq = qmul(qinv(quats[:-1]), quats[1:])                          # (F-1, B, 4)
    w = np.clip(np.abs(dq[..., 3]), 0.0, 1.0)
    deg_s = np.degrees(2 * np.arccos(w)) * fps
    return {"p99": float(np.percentile(deg_s, 99)), "max": float(deg_s.max())}


def main() -> None:
    argv = sys.argv[1:]
    check = "--check" in argv
    calibrate = "--calibrate" in argv
    argv = [a for a in argv if a not in ("--check", "--calibrate")]
    args = gate_overrides(argv)
    if len(args) < 2:
        print(__doc__)
        raise SystemExit(2)
    glb, track_path = Path(args[0]), Path(args[1])

    gltf, bin_ = read_glb(glb)
    skel = Skeleton(gltf)
    surfaces = calibrated_surfaces(gltf, bin_, skel) if gltf.get("animations") else None

    track = json.loads(track_path.read_text())
    bones: list[str] = track["bones"]
    fps = float(track["fps"])
    quats = np.array(track["quats"], dtype=float).reshape(track["frames"], len(bones), 4)
    quats /= np.linalg.norm(quats, axis=2, keepdims=True)
    print(f"{track_path.name}: {track['frames']} frames x {len(bones)} bones at {fps:g} fps"
          + (f", refined v{track['refined']['version']}" if track.get("refined") else ", unrefined"))

    pos = fk_world(skel, track_rot(bones, quats), {}, track["frames"])

    if surfaces is not None:
        touch, deep = penetration_counts(pos, surfaces)
    else:
        touch, deep = 0, 0
        print("  (no capture clips in the glb: clipping check skipped)")

    feet = foot_stats(pos, track["frames"])
    vel = angvel_stats(quats, fps)

    print()
    print(f"  clipping touch frames  {touch}   (deep {deep})  of {track['frames']}")
    print(f"  foot float mean/max    {feet['float_mean_m'] * 100:.2f} / {feet['float_max_m'] * 100:.2f} cm"
          f"   (lower toe; sole at {SOLE_BELOW_TOE * 100:.2f} cm)")
    print(f"  foot skate mean/max    {feet['skate_mean_cm']:.4f} / {feet['skate_max_cm']:.3f} cm/frame"
          f"   ({int(feet['planted_frames'])} planted frames)")
    print(f"  angular vel p99/max    {vel['p99']:.1f} / {vel['max']:.1f} deg/s")

    if calibrate and gltf.get("animations"):
        print()
        print("  foot-skate baseline, built-in clips (how the foot-skate gate was set):")
        p95s = []
        for clip in gltf["animations"]:
            rot, tr, n = clip_locals(gltf, bin_, clip)
            st = foot_stats(fk_world(skel, rot, tr, n), n)
            p95s.append(st["skate_p95_cm"])
            print(f"    {clip['name']:22s} planted mean {st['skate_mean_cm']:.4f}, "
                  f"p95 {st['skate_p95_cm']:.4f}, max {st['skate_max_cm']:.3f} cm/frame")
        worst = max(p95s)
        print(f"    worst clip p95 {worst:.4f} cm/frame -> gate {GATES['foot_skate_max_cm']:g} "
              f"(~{GATES['foot_skate_max_cm'] / worst:.1f}x)")

    if not check:
        return
    failures = []
    if deep > GATES["deep_frames_max"]:
        failures.append(f"deep penetration {deep} frames > {GATES['deep_frames_max']:g} "
                        f"(a hand is through the body)")
    if touch > GATES["touch_frames_max"]:
        failures.append(f"touch {touch} frames > {GATES['touch_frames_max']:g} "
                        f"(hands closer than any approved capture ever gets)")
    float_cm = feet["float_mean_m"] * 100
    if not GATES["foot_float_mean_min_cm"] <= float_cm <= GATES["foot_float_mean_max_cm"]:
        failures.append(f"foot float mean {float_cm:.2f} cm outside "
                        f"[{GATES['foot_float_mean_min_cm']:g}, {GATES['foot_float_mean_max_cm']:g}] "
                        f"(the stance hovers or sinks)")
    if feet["skate_mean_cm"] > GATES["foot_skate_max_cm"]:
        failures.append(f"foot skate {feet['skate_mean_cm']:.4f} cm/frame > {GATES['foot_skate_max_cm']:g} "
                        f"(planted feet slide)")
    if vel["max"] > GATES["angvel_max_deg_s"]:
        failures.append(f"angular velocity max {vel['max']:.0f} deg/s > {GATES['angvel_max_deg_s']:g} "
                        f"(teleport/seizure flicker)")
    print()
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        raise SystemExit(1)
    print("  PASS  every gate")


if __name__ == "__main__":
    main()
