"""Make every shot seam of a stitched track velocity-continuous.

    .venv/bin/python smooth_seams.py <track.json> <shots.json> [--half-window 0.6] [--out out.json]

Why (measured 2026-11-06 on the TEDx stage profile shipped as personal-site's speaking track):
extract_shots.py joins shots with a 0.4 s window that samples both sides at half rate and
slerps them on a linear ramp. Across a camera cut GVHMR's per-shot estimates disagree (the
whole-body orientation most of all), so the ramp adds a constant extra angular velocity that
switches on and off at its two ends, and the half-rate sampling slows the gesture inside it.
The eye reads that as a hitch: 8 of the 10 seams had a peak arm acceleration above twice the
track's p99 elsewhere (14.2 rad/s^2), the worst 102.7 rad/s^2, and Hips / UpLeg were the bones
that spiked most often.

Fix: around each seam, the frames [c - W, c + W] of every bone are replaced by a cubic Hermite
curve in rotation-vector space, anchored at the window's end frames with the end angular
velocities taken from the frames just outside it, so rotation and angular velocity are both
continuous (C1). The window is longer than the stitch (1.2 s against 0.4 s by default), so the
residual pose change is spread out instead of crammed in. Frame count and timing are
unchanged, so the track stays aligned with its audio.

Run it on the retargeted track before refine_track.py and ground_track.py: those passes
re-centre the arms and re-solve the legs and pelvis from the smoothed rotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions, broadcasting."""
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def qinv(q: np.ndarray) -> np.ndarray:
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def qlog(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of unit xyzw quaternions, shortest arc."""
    q = np.where(q[..., 3:4] < 0, -q, q)
    v = q[..., :3]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(s, q[..., 3:4])
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(s > 1e-12, angle / s, 2.0)
    return v * k


def qexp(r: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(r, axis=-1, keepdims=True)
    half = angle / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(angle > 1e-12, np.sin(half) / angle, 0.5)
    return np.concatenate([r * k, np.cos(half)], axis=-1)


def hermite_window(q: np.ndarray, a: int, z: int) -> np.ndarray:
    """Replace frames a..z (inclusive) of q (T, B, 4) with a C1 cubic Hermite in rotation-vector
    space relative to q[a]; end slopes from the neighbouring frames a-1 and z+1."""
    base_inv = qinv(q[a])
    rel = lambda f: qlog(qmul(base_inv, q[f]))            # noqa: E731
    r_a, r_z = np.zeros_like(rel(a)), rel(z)
    d_a = -rel(a - 1)                                      # per-frame slope entering the window
    d_z = rel(z + 1) - r_z                                 # per-frame slope leaving it
    n = z - a
    s = (np.arange(n + 1) / n)[:, None, None]
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    r = h00 * r_a + h10 * n * d_a + h01 * r_z + h11 * n * d_z
    out = q.copy()
    out[a:z + 1] = qmul(q[a][None], qexp(r))
    return out


def seam_frames(shots: dict) -> list[int]:
    return [s["outStartFrame"] for s in shots["shots"][1:]]


def angular_speed(q: np.ndarray, fps: float) -> np.ndarray:
    """(T-1, B) rad/s between consecutive frames."""
    dot = np.abs(np.sum(q[1:] * q[:-1], axis=-1)).clip(0, 1)
    return 2 * np.arccos(dot) * fps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track", type=Path)
    ap.add_argument("shots", type=Path)
    ap.add_argument("--half-window", type=float, default=0.6, help="seconds either side of each seam")
    ap.add_argument("--out", type=Path, help="default: overwrite the input")
    args = ap.parse_args()
    track = json.loads(args.track.read_text())
    fps, frames, bones = track["fps"], track["frames"], track["bones"]
    q = np.array(track["quats"], dtype=np.float64).reshape(frames, len(bones), 4)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    w = int(round(args.half_window * fps))
    seams = [c for c in seam_frames(json.loads(args.shots.read_text())) if w + 1 <= c < frames - w - 1]
    before = angular_speed(q, fps)
    for c in seams:
        q = hermite_window(q, c - w, c + w)
    after = angular_speed(q, fps)
    acc_b, acc_a = np.abs(np.diff(before, axis=0)) * fps, np.abs(np.diff(after, axis=0)) * fps
    for c in seams:
        lo, hi = c - w - 2, c + w + 2
        print(f"seam {c / fps:7.2f}s  peak accel {acc_b[lo:hi].max():6.1f} -> {acc_a[lo:hi].max():6.1f} rad/s^2")
    track["quats"] = [round(float(x), 6) for x in q.reshape(-1)]
    track["seams"] = {"version": 1, "half_window_s": args.half_window, "frames": seams}
    out = args.out or args.track
    out.write_text(json.dumps(track))
    print(f"wrote {out} ({len(seams)} seams, ±{w} frames each)")


if __name__ == "__main__":
    main()
