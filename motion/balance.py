"""Static balance audit on the avatar rig: would this pose sequence fall over?

Physics: the Linear Inverted Pendulum Model (LIPM) capture point,

    CP = com_xz + com_vel_xz / omega,   omega = sqrt(g / h),

is where the body would have to step to come to a stop. CP inside the support
polygon of the planted feet = recoverable without stepping; CP outside while
both feet stay planted, sustained for seconds, = a pose no human can hold —
the avatar leans past its feet and reads as "about to tip over".

Measured on the exact FK the player runs (fk_world on the refined track.json —
pure rotation, hips pinned at bind translation, same convention as
check_track.py), so what is measured is what plays. Segment masses are Dempster
fractions mapped onto the Mixamo-style bones (SEGMENTS, normalized); the whole-
body CoM is the mass-weighted assembly of segment midpoints. The floor is the
track's own: the runtime presses the lower toe to the sole offset
(check_track.py SOLE_BELOW_TOE), so floor_y = min toe world Y over the track,
and a foot counts as planted while its toe stays within GROUND_MARGIN of it.

A "fall-risk run" is a maximal run of frames where BOTH feet are planted and
the CP stays more than EXCURSION_MARGIN outside the support polygon; a run of
at least SUSTAIN_S is sustained. Used by check_track.py (gate) and
correct_balance.py (fix).
"""
from __future__ import annotations

import numpy as np

G = 9.81
GROUND_MARGIN = 0.05       # toe within this of the track's floor = planted (m)
EXCURSION_MARGIN = 0.02    # CP must be this far outside the polygon to count (m)
SUSTAIN_S = 1.0            # a both-feet-planted run this long is a fall-risk run

# (Dempster mass fraction, bones whose world-position midpoint is the segment CoM;
#  a single bone = joint point). Fractions are normalized by their sum at assembly.
SEGMENTS: list[tuple[float, list[str]]] = [
    (0.142, ["Hips"]),                       # pelvis
    (0.131, ["Spine", "Spine1"]),            # abdomen
    (0.160, ["Spine2", "Neck"]),             # thorax
    (0.081, ["Neck", "Head"]),               # head + neck
    (0.100, ["LeftUpLeg", "LeftLeg"]),       # thigh
    (0.100, ["RightUpLeg", "RightLeg"]),
    (0.047, ["LeftLeg", "LeftFoot"]),        # shank
    (0.047, ["RightLeg", "RightFoot"]),
    (0.014, ["LeftFoot", "LeftToeBase"]),    # foot
    (0.014, ["RightFoot", "RightToeBase"]),
    (0.027, ["LeftArm", "LeftForeArm"]),     # upper arm
    (0.027, ["RightArm", "RightForeArm"]),
    (0.016, ["LeftForeArm", "LeftHand"]),    # forearm
    (0.016, ["RightForeArm", "RightHand"]),
    (0.006, ["LeftHand"]),                   # hand
    (0.006, ["RightHand"]),
]

FEET = [("LeftFoot", "LeftToeBase"), ("RightFoot", "RightToeBase")]


def com_track(pos: dict[str, np.ndarray]) -> np.ndarray:
    """Whole-body CoM (F,3) from FK world positions: mass-weighted segment midpoints."""
    nframes = next(iter(pos.values())).shape[0]
    total = sum(m for m, _ in SEGMENTS)
    com = np.zeros((nframes, 3))
    for mass, bones in SEGMENTS:
        p = np.mean([pos[b] for b in bones], axis=0)
        com += (mass / total) * p
    return com


def floor_y(pos: dict[str, np.ndarray]) -> float:
    """The track's own floor: lowest toe world Y (the runtime presses the lower sole down)."""
    return float(min(pos[toe][:, 1].min() for _, toe in FEET))


def _hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull of (N,2) xz points, monotonic chain. Returns (M,2) CCW, M>=2."""
    p = sorted(map(tuple, pts))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for pt in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    upper: list[tuple[float, float]] = []
    for pt in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)
    return np.array(lower[:-1] + upper[:-1])


def support_polygon(pos: dict[str, np.ndarray], frame: int, fl: float,
                    ground_margin: float = GROUND_MARGIN) -> np.ndarray | None:
    """Hull of the grounded feet's ankle+toe xz points at a frame; None if airborne."""
    pts = []
    for ankle, toe in FEET:
        if pos[toe][frame, 1] - fl < ground_margin:
            pts.append(pos[ankle][frame, [0, 2]])
            pts.append(pos[toe][frame, [0, 2]])
    if not pts:
        return None
    pts = np.unique(np.array(pts), axis=0)
    return pts if len(pts) <= 2 else _hull(pts)


def cp_distance(pt: np.ndarray, hull: np.ndarray | None) -> float:
    """Distance from a capture point to the support polygon; 0 when inside."""
    if hull is None:
        return np.inf
    if len(hull) < 3:                      # single-edge/point support: distance to it
        a, b = hull[0], hull[-1]
        ab = b - a
        t = np.clip(np.dot(pt - a, ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        return float(np.linalg.norm(pt - (a + t * ab)))
    n = len(hull)
    best = np.inf
    inside = True
    sign = None
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        e = b - a
        c = e[0] * (pt[1] - a[1]) - e[1] * (pt[0] - a[0])
        s = c > 0
        if sign is None:
            sign = s
        elif s != sign:
            inside = False
        # closest point on this edge segment
        t = np.clip(np.dot(pt - a, e) / max(float(e @ e), 1e-12), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(pt - (a + t * e))))
    return 0.0 if inside else best


def both_planted(pos: dict[str, np.ndarray], fl: float,
                 ground_margin: float = GROUND_MARGIN) -> np.ndarray:
    """(F,) bool: both toes within ground_margin of the floor."""
    return np.all([pos[toe][:, 1] - fl < ground_margin for _, toe in FEET], axis=0)


def capture_point(com: np.ndarray, fps: float, fl: float) -> np.ndarray:
    """LIPM capture point (F,2) in xz from the CoM track."""
    dt = 1.0 / fps
    vel = np.gradient(com, dt, axis=0)
    h = np.maximum(com[:, 1] - fl, 0.3)
    omega = np.sqrt(G / h)
    return com[:, [0, 2]] + vel[:, [0, 2]] / omega[:, None]


def fall_risk_runs(excursion: np.ndarray, planted2: np.ndarray, fps: float,
                   margin: float = EXCURSION_MARGIN,
                   sustain_s: float = SUSTAIN_S) -> tuple[list[dict], int]:
    """Maximal runs of both-feet-planted frames with CP outside the polygon by > margin.

    excursion: (F,) CP distance to the support polygon (0 inside; inf airborne).
    Returns dicts {start, end, duration_s, max_excursion_m, peak} for runs of at
    least sustain_s seconds; shorter runs are reported by balance_stats as counts only.
    """
    bad = planted2 & (excursion > margin)
    runs: list[dict] = []
    short = 0
    n = len(bad)
    f = 0
    while f < n:
        if bad[f]:
            s = f
            while f < n and bad[f]:
                f += 1
            if (f - s) / fps >= sustain_s:
                seg = excursion[s:f]
                peak = s + int(np.argmax(seg))
                runs.append({"start": s, "end": f, "duration_s": (f - s) / fps,
                             "max_excursion_m": float(seg.max()), "peak": peak})
            else:
                short += 1
        else:
            f += 1
    return runs, short


def balance_stats(pos: dict[str, np.ndarray], fps: float,
                  margin: float = EXCURSION_MARGIN,
                  sustain_s: float = SUSTAIN_S) -> dict:
    """Full balance audit of an FK position track. See module docstring for the model."""
    nframes = next(iter(pos.values())).shape[0]
    fl = floor_y(pos)
    com = com_track(pos)
    cp = capture_point(com, fps, fl)
    planted2 = both_planted(pos, fl)

    excursion = np.zeros(nframes)
    for f in range(nframes):
        excursion[f] = cp_distance(cp[f], support_polygon(pos, f, fl))

    outside = planted2 & (excursion > margin)
    runs, short = fall_risk_runs(excursion, planted2, fps, margin, sustain_s)
    worst = max(runs, key=lambda r: r["max_excursion_m"], default=None)
    return {
        "floor_y": fl,
        "com": com,
        "cp": cp,
        "planted2": planted2,
        "excursion": excursion,
        "pct_outside": float(100.0 * outside.sum() / nframes),
        "short_runs": short,
        "runs": runs,
        "worst": worst,
    }
