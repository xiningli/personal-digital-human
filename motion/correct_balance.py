"""Balance correction pass: rewrite the fall-risk runs check_track.py's balance gate flags.

    .venv/bin/python correct_balance.py <track.json> [--avatar model-clips.glb]
        [--force] [--out out.json]

The TED stage profile measured 4 sustained runs (1.3-3.7 s) where the LIPM capture
point sat up to 8.9 cm outside the foot support polygon with BOTH feet planted — a
pose no human holds without falling (GVHMR hallucinated the stance of a speaker whose
feet were out of frame). This pass pulls the body back over its feet.

How, on a rig with no hip translation (pure-rotation FK, hips pinned at bind
translation — check_track.py's docstring): rotating the Hips alone is useless, the
CoM sits ~10 cm above the hip joint while the feet sit ~1 m below, so a hip
pre-rotation swings the feet and barely moves the CoM. The correction is the human
"hip strategy" instead:

  1. pre-rotate the Hips local quaternion by a world-space delta about the
     horizontal axis that moves the CoM toward the support polygon (the torso,
     head and arms — ~47% of body mass, ~30 cm above the hips — ride along), then
  2. restore each UpLeg's WORLD rotation to its pre-correction value per frame
     (q_local' = R_hips_world'^-1 @ R_upleg_world_ref). The leg chain's orientations
     below are then untouched; the feet only ride the hip joints' displacement under
     the pelvis rotation (<= 2.5 cm at the largest TED correction, Hann-smooth over
     the window), so the foot float/skate gates are undisturbed and no re-lock is
     needed (measured: skate mean 0.1409 -> 0.1415 cm/frame).

Per run, the needed angle is measured, not modeled: a 0.02 rad probe rotation at the
run's peak frame (legs restored, exactly as the real pass applies it — with the legs
free they swing opposite and cancel most of the CoM motion) gives the CoM's horizontal
displacement per radian, and the angle is (excursion + 2 cm margin) / sensitivity,
capped per iteration. The per-frame profile is full strength across the run, Hann-
tapered to zero over 0.5 s on each side (same GUARD_RADIUS style as refine_track.py),
so nothing steps. The loop re-FKs and re-measures after each pass — the capture point
includes the CoM velocity the ramp itself introduces, so one analytic shot would miss.
Measured on the TED track (2026-09-23): converges in 2 passes, 4 runs fixed, peak
correction 20 deg; max both-planted excursion 8.87 cm -> 3.74 cm and what remains is
sub-0.3 s dynamic transients (weight shifts), no sustained run. Foot float/skate and
angular velocity are essentially untouched (skate mean 0.1409 -> 0.1415 cm/frame;
the hip joints ride the pelvis rotation by <= 2.5 cm, Hann-smooth, so stance never
steps).

Deterministic (no randomness) and idempotent: the output carries a top-level
"balance" marker and a second run refuses to touch the file unless --force is given.
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

from balance import (  # noqa: E402
    EXCURSION_MARGIN, SUSTAIN_S, balance_stats, com_track, support_polygon,
)
from refine_track import (  # noqa: E402
    DEFAULT_AVATAR, Skeleton, fk_world, qmul, read_glb, track_rot,
)

HIPS = "Hips"
UPLEGS = ["LeftUpLeg", "RightUpLeg"]
UP = np.array([0.0, 1.0, 0.0])

RAMP_S = 0.5               # Hann taper each side of a run, seconds
PROBE_RAD = 0.02           # sensitivity probe rotation
MAX_ITER = 6               # correction passes before giving up (loudly)
MAX_STEP_RAD = np.radians(10.0)   # per-iteration cap; converges in 2-3 passes
MAX_TOTAL_RAD = np.radians(30.0)  # beyond this the pose would read as bowing, not standing


def run_window_weight(nframes: int, start: int, end: int, ramp: int) -> np.ndarray:
    """1 across [start, end), Hann-tapered to 0 over `ramp` frames on each side."""
    w = np.zeros(nframes)
    w[start:end] = 1.0
    d = np.arange(1, ramp + 1)
    taper = 0.5 * (1.0 + np.cos(np.pi * d / ramp))
    lo = start - d
    hi = end - 1 + d
    w[lo[lo >= 0]] = taper[lo >= 0]
    w[hi[hi < nframes]] = np.maximum(w[hi[hi < nframes]], taper[hi < nframes])
    return w


def world_to_local_delta(skel: Skeleton, parent_rot: np.ndarray | None,
                         rotvec: np.ndarray) -> np.ndarray:
    """World-space per-frame delta rotations (F,3 rotvec) as Hips-local pre-rotations
    (F,4): Rl = gp^T @ Rw @ gp, the same conversion as refine_track's escape pass."""
    Rw = Rotation.from_rotvec(rotvec).as_matrix()
    if parent_rot is None:
        return Rotation.from_matrix(Rw).as_quat()
    Rl = np.transpose(parent_rot, (0, 2, 1)) @ Rw @ parent_rot
    return Rotation.from_matrix(Rl).as_quat()


def correct(track: dict, skel: Skeleton) -> tuple[np.ndarray, dict]:
    """Run the correction loop on the track's quats; returns (quats, report-marker)."""
    bones: list[str] = track["bones"]
    fps = float(track["fps"])
    idx = {b: k for k, b in enumerate(bones)}
    quats = np.array(track["quats"], dtype=float).reshape(track["frames"], len(bones), 4)
    nframes = track["frames"]
    ramp = max(1, int(round(RAMP_S * fps)))
    hips_parent = skel.parent.get(skel.name2idx[HIPS])

    # Reference leg world rotations: the legs keep exactly these orientations, so the
    # feet only ride the hip joints' small displacement under the pelvis rotation.
    _, rots_ref = fk_world(skel, track_rot(bones, quats), {}, nframes, want_rot=True)
    upleg_ref = {b: rots_ref[b].copy() for b in UPLEGS if b in idx}

    total_theta = np.zeros(nframes)          # accumulated correction magnitude, for the cap
    runs_fixed = 0
    iterations = 0
    for iterations in range(1, MAX_ITER + 1):
        pos = fk_world(skel, track_rot(bones, quats), {}, nframes)
        st = balance_stats(pos, fps)
        if not st["runs"]:
            break
        runs_fixed = max(runs_fixed, len(st["runs"]))
        rotvec = np.zeros((nframes, 3))
        for run in st["runs"]:
            peak = run["peak"]
            hull = support_polygon(pos, peak, st["floor_y"])
            cp = st["cp"][peak]
            # direction: from the CP toward the polygon's centroid (always inward);
            # magnitude: past the nearest edge plus the gate margin.
            centroid = hull.mean(axis=0)
            direction = centroid - cp
            dist = float(np.linalg.norm(direction))
            if dist < 1e-6:
                continue
            direction /= dist
            needed = run["max_excursion_m"] + EXCURSION_MARGIN
            # horizontal axis that moves the (above-pivot) CoM along +direction:
            # axis x up = direction, i.e. axis = up x direction (xz -> 3D).
            axis = np.cross(UP, np.array([direction[0], 0.0, direction[1]]))
            axis /= np.linalg.norm(axis)

            # sensitivity probe: how far does the CoM xz move per radian about this
            # axis? Measured with the legs restored, exactly as the real correction
            # applies it (with the legs free they swing opposite and cancel most of it).
            probe = {b: quats[peak:peak + 1, idx[b]] for b in bones}
            ql = world_to_local_delta(
                skel,
                rots_ref[skel.names[hips_parent]][peak:peak + 1] if hips_parent is not None else None,
                (axis * PROBE_RAD)[None, :])
            probe[HIPS] = qmul(ql, probe[HIPS])
            _, probe_rots = fk_world(skel, probe, {}, 1, want_rot=True)
            for b, ref in upleg_ref.items():
                Rl = np.transpose(probe_rots[HIPS], (0, 2, 1)) @ ref[peak:peak + 1]
                probe[b] = Rotation.from_matrix(Rl).as_quat()
            pos_probe = fk_world(skel, probe, {}, 1)
            dcom = com_track(pos_probe)[0, [0, 2]] - st["com"][peak, [0, 2]]
            sens = float(np.dot(dcom, direction)) / PROBE_RAD
            if sens < 1e-3:
                continue
            angle = min(needed / sens, MAX_STEP_RAD)
            w = run_window_weight(nframes, run["start"], run["end"], ramp)
            rotvec += (axis * angle)[None, :] * w[:, None]
        mag = np.linalg.norm(rotvec, axis=1)
        room = np.maximum(MAX_TOTAL_RAD - total_theta, 0.0)
        over = mag > room
        if over.any():
            rotvec[over] *= (room[over] / mag[over])[:, None]
        if not (mag > 1e-6).any():
            print(f"  pass {iterations}: no correctable runs (probe failed); stopping")
            break
        total_theta += np.minimum(mag, room)
        print(f"  pass {iterations}: {len(st['runs'])} runs, "
              f"peak correction {np.degrees(min(float(mag.max()), float(MAX_TOTAL_RAD))):.1f} deg")
        ql = world_to_local_delta(
            skel,
            rots_ref[skel.names[hips_parent]] if hips_parent is not None else None,
            rotvec)
        quats[:, idx[HIPS]] = qmul(ql, quats[:, idx[HIPS]])
        # restore the legs' world rotations: feet exactly where the refined track put them
        _, rots_now = fk_world(skel, track_rot(bones, quats), {}, nframes, want_rot=True)
        hips_world = rots_now[HIPS]
        for b, ref in upleg_ref.items():
            Rl = np.transpose(hips_world, (0, 2, 1)) @ ref
            quats[:, idx[b]] = Rotation.from_matrix(Rl).as_quat()
        quats /= np.linalg.norm(quats, axis=2, keepdims=True)

    pos = fk_world(skel, track_rot(bones, quats), {}, nframes)
    st = balance_stats(pos, fps)
    return quats, {
        "version": 1,
        "margin_m": EXCURSION_MARGIN,
        "sustain_s": SUSTAIN_S,
        "iterations": iterations,
        "runs_before": runs_fixed,
        "runs_after": len(st["runs"]),
        "max_excursion_cm_after": round(
            float(st["excursion"][st["planted2"]].max(initial=0.0)) * 100, 2),
        "max_correction_deg": round(float(np.degrees(total_theta.max())), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("track", type=Path)
    ap.add_argument("--avatar", type=Path, default=DEFAULT_AVATAR)
    ap.add_argument("--force", action="store_true", help="redo a track that already carries the marker")
    ap.add_argument("--out", type=Path, help="default: overwrite the input in place (atomically)")
    args = ap.parse_args()

    track = json.loads(args.track.read_text())
    if track.get("balance") and not args.force:
        print(f"refusing: {args.track} already carries a 'balance' marker; pass --force to redo")
        return
    gltf, _ = read_glb(args.avatar)
    skel = Skeleton(gltf)

    print(f"{args.track.name}: {track['frames']} frames at {track['fps']:g} fps")
    quats, marker = correct(track, skel)
    print(f"  balance: {marker['runs_before']} runs before, {marker['runs_after']} after, "
          f"max excursion {marker['max_excursion_cm_after']} cm, "
          f"max correction {marker['max_correction_deg']} deg")
    if marker["runs_after"]:
        print("  <-- FALL-RISK RUNS REMAIN", file=sys.stderr)

    out = args.out or args.track
    payload = {
        **track,
        "quats": np.round(quats, 4).reshape(-1).tolist(),
        "balance": marker,
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out)
    print(f"  -> {out} ({out.stat().st_size / 1e6:.2f} MB)")
    if marker["runs_after"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
