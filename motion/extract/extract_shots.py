"""extract_shots.py <video.mp4> <out.npz> [shots.json] — shot-aware GVHMR extraction.

For edited clips (talks that cut between a full-body medium shot and close-ups, e.g.
TED) a single GVHMR pass is the wrong unit: the tracker/ViTPose see framing jumps and
the static-camera assumption holds per shot, not per clip. This script runs ffmpeg
scene detection (threshold 0.3, shots under 1.5 s merged into their longer neighbour),
cuts the clip into shots, and predicts each shot with one GVHMR instance in one
process — one model load, preprocessors (tracker/ViTPose/feature extractor) reused
across shots, CUDA cache dropped between steps the way extract.py's run_preprocess does.

The per-shot SMPL-X sequences are then stitched. Feet-hidden shots (close-ups, where
ViTPose ankle confidence collapses) carry no leg information — GVHMR hallucinates
legs there and they float — so their leg rotations are first replaced with the slerp
bridge between the nearest feet-visible shots' boundary stances. Then at every cut an
equal-length crossfade: an N-frame window centred on the cut samples BOTH sides at
half rate (the outgoing shot's last N/2 frames and the incoming shot's first N/2,
stretched over the N output frames) and quaternion-slerps them (axis-angle -> quat ->
slerp -> back) on a linear 0->1 ramp; root translation is lerped on the same ramp and
the incoming shot then plays pure from its frame N/2. Every source frame is consumed
exactly once, so the output length is the exact sum of the shot lengths and the
timeline stays sample-aligned with the audio — no cumulative drift; only the 0.4 s
window itself is locally time-warped. (The linear ramp, not a cosine ease: the
cosine's mid-fade crest is 1.57x the average slerp rate and can breach
check_track.py's 600 deg/s gate.) Betas are the median over shots of each shot's
per-frame median. Output is the exact npz contract of extract.py (poses flat (T,165),
betas, mocap_frame_rate) plus the .raw.pt sidecar; shots.json lists the shots, fade
positions and per-seam smoothness measurements for debugging and for the player's
seam checks.

Seam smoothness is reported per cut: "pre-jump" is the pose discontinuity a naive
concatenation would have had (max/mean over the 22 actuated joints, degrees between
the outgoing last frame and the incoming first frame); "post" is the worst
adjacent-frame angular velocity after the fade (deg/s), which must stay well under
check_track.py's 600 deg/s teleport gate.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from extract import (  # noqa: E402
    GVHMR, compose_cfg, copy_video, init_config, load_model, predict,
    save_npz, video_fps, world_fix,
)
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy  # noqa: E402
from hmr4d.utils.preproc import Extractor, Tracker, VitPoseExtractor  # noqa: E402
from hmr4d.utils.video_io_utils import get_video_lwh  # noqa: E402

SCENE_THRESHOLD = 0.3     # same as the reconnaissance pass; TED-style hard cuts score >0.4
MIN_SHOT_S = 1.5          # shorter shots are editing noise (flashes, whip pans): merge away
FADE_S = 0.4              # crossfade length at every seam
ANKLE_CONF_MIN = 0.3      # ViTPose ankle confidence below this = feet not visible in the shot
# 22-layout (global_orient + 21 body joints): hips/knees/ankles/feet — the legs.
LEG_IDX = [1, 2, 4, 5, 7, 8, 10, 11]


def detect_cuts(video_path: Path) -> list[float]:
    """ffmpeg scene detection, same filter as the manual recon pass: cut timestamps."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(video_path),
         "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", r.stderr)]


def merge_short(bounds: list[float]) -> list[float]:
    """Drop shots shorter than MIN_SHOT_S by merging them into the longer neighbour."""
    bounds = list(bounds)
    while len(bounds) > 2:
        segs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        short = [i for i, (a, b) in enumerate(segs) if b - a < MIN_SHOT_S]
        if not short:
            break
        i = short[0]
        if i == 0:
            del bounds[1]
        elif i == len(segs) - 1:
            del bounds[-2]
        else:
            left = segs[i - 1][1] - segs[i - 1][0]
            right = segs[i + 1][1] - segs[i + 1][0]
            if left >= right:
                del bounds[i]
            else:
                del bounds[i + 1]
    return bounds


def cut_shot(video_path: Path, start: float, end: float, out_path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(video_path),
         "-vf", "fps=30", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18",
         "-preset", "fast", "-an", str(out_path)],
        check=True,
    )


def to_quat(aa: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(aa.reshape(-1, 3)).as_quat().reshape(aa.shape[:-1] + (4,))


def to_aa(quat: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(quat.reshape(-1, 4)).as_rotvec().reshape(quat.shape[:-1] + (3,)).astype(np.float32)


def slerp(q0: np.ndarray, q1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Vectorized quaternion slerp; w broadcasts as (..., 1) in [0, 1]."""
    d = (q0 * q1).sum(axis=-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    theta = np.arccos(np.clip(np.abs(d), -1.0, 1.0))
    s = np.sin(theta)
    small = s < 1e-8
    s_safe = np.where(small, 1.0, s)
    w0 = np.where(small, 1.0 - w, np.sin((1.0 - w) * theta) / s_safe)
    w1 = np.where(small, w, np.sin(w * theta) / s_safe)
    return w0 * q0 + w1 * q1


def sample_quat(q: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Fractional-frame sampling of a (T, J, 4) quat sequence (slerp between neighbours)."""
    i0 = np.clip(np.floor(pos).astype(int), 0, len(q) - 1)
    i1 = np.clip(i0 + 1, 0, len(q) - 1)
    frac = (pos - i0)[:, None, None]
    return slerp(q[i0], q[i1], frac)


def sample_vec(v: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Fractional-frame linear sampling of a (T, D) sequence."""
    i0 = np.clip(np.floor(pos).astype(int), 0, len(v) - 1)
    i1 = np.clip(i0 + 1, 0, len(v) - 1)
    frac = (pos - i0)[:, None]
    return v[i0] * (1.0 - frac) + v[i1] * frac


def quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Rotation angle between quaternion pairs, degrees."""
    d = np.clip(np.abs((q0 * q1).sum(axis=-1)), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


def quat_mean(q: np.ndarray) -> np.ndarray:
    """Sign-aligned normalized mean of quaternions, shape (..., 4)."""
    ref = q.reshape(-1, 4)[0]
    aligned = np.where((q * ref).sum(axis=-1, keepdims=True) < 0, -q, q)
    m = aligned.mean(axis=0)
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def bridge_hidden_legs(quats: list[np.ndarray], ankle_conf: list[float]) -> list[bool]:
    """Feet-hidden shots (close-ups: ViTPose ankle confidence under ANKLE_CONF_MIN) carry
    no leg information — GVHMR hallucinates legs there, and they float (measured: one
    17 s close-up put the lower foot 14.6 cm up where every full-body shot sits under 3).
    Replace such a shot's leg rotations with the slerp bridge between the nearest
    feet-visible shots' boundary stances: any plausible legs are equally correct for
    invisible limbs, and the bridge removes both the float and the seam pop. Upper body
    and global_orient stay as predicted. Returns the bridged flags per shot."""
    visible = [i for i, c in enumerate(ankle_conf) if c >= ANKLE_CONF_MIN]
    bridged = [False] * len(quats)
    if not visible:
        return bridged
    for i in range(len(quats)):
        if i in visible:
            continue
        j = max((v for v in visible if v < i), default=None)
        k = min((v for v in visible if v > i), default=None)
        tail = min(15, len(quats[j if j is not None else k]) // 2)
        head = min(15, len(quats[k if k is not None else j]) // 2)
        ref_a = quat_mean(quats[j][-tail:, LEG_IDX]) if j is not None else quat_mean(quats[k][:head, LEG_IDX])
        ref_b = quat_mean(quats[k][:head, LEG_IDX]) if k is not None else quat_mean(quats[j][-tail:, LEG_IDX])
        T = len(quats[i])
        u = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None, None]
        quats[i][:, LEG_IDX] = slerp(np.broadcast_to(ref_a, (T, len(LEG_IDX), 4)),
                                      np.broadcast_to(ref_b, (T, len(LEG_IDX), 4)), u)
        bridged[i] = True
    return bridged


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    video_path = Path(sys.argv[1]).resolve()
    assert video_path.exists(), f"video not found: {video_path}"
    out_npz = Path(sys.argv[2]).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    shots_json = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else out_npz.with_suffix(".shots.json")
    os.chdir(GVHMR)  # GVHMR config/paths are relative to its repo root

    fps = video_fps(video_path)
    duration = get_video_lwh(video_path)[0] / fps
    raw_cuts = detect_cuts(video_path)
    cuts = [c for c in raw_cuts if 0.5 < c < duration - 0.5]
    bounds = merge_short([0.0, *cuts, duration])
    n_shots = len(bounds) - 1
    print(f"[shots] {len(raw_cuts)} raw cuts at scene>{SCENE_THRESHOLD} -> {n_shots} shots "
          f"(<{MIN_SHOT_S}s merged), fade {FADE_S:.2f}s windows, equal-length stitch")

    shot_dir = HERE / "videos" / "shots"
    shot_dir.mkdir(parents=True, exist_ok=True)
    shot_paths = []
    for i in range(n_shots):
        p = shot_dir / f"{video_path.stem}.shot{i:02d}.mp4"
        if not p.exists():
            cut_shot(video_path, bounds[i], bounds[i + 1], p)
        shot_paths.append(p)

    # One model load; preprocessors built once and reused across shots (run_preprocess
    # would rebuild and free them per shot; its cache layout is kept via compose_cfg).
    seqs: list[dict[str, np.ndarray]] = []
    with init_config():
        cfgs = [compose_cfg(p, HERE / "outputs") for p in shot_paths]
        tracker = Tracker()
        vitpose_extractor = VitPoseExtractor()
        extractor = Extractor()
        try:
            for i, cfg in enumerate(cfgs):
                copy_video(shot_paths[i], cfg)
                if not Path(cfg.paths.bbx).exists():
                    bbx_xyxy = tracker.get_one_track(cfg.video_path).float()
                    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
                    torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, cfg.paths.bbx)
                bbx_xys = torch.load(cfg.paths.bbx)["bbx_xys"]
                if not Path(cfg.paths.vitpose).exists():
                    torch.save(vitpose_extractor.extract(cfg.video_path, bbx_xys), cfg.paths.vitpose)
                if not Path(cfg.paths.vit_features).exists():
                    torch.save(extractor.extract_video_features(cfg.video_path, bbx_xys), cfg.paths.vit_features)
                print(f"[shots] preprocessed {i + 1}/{n_shots} ({bounds[i]:.1f}-{bounds[i + 1]:.1f}s)")
        finally:
            del tracker, vitpose_extractor, extractor
            torch.cuda.empty_cache()
        model = load_model(cfgs[0])
        for i, cfg in enumerate(cfgs):
            pred = predict(model, cfg)
            p = pred["smpl_params_global"]
            go, tr = world_fix(p["global_orient"].numpy(), p["transl"].numpy())
            kp2d = torch.load(cfg.paths.vitpose)  # (T, 17, 3), COCO layout
            seqs.append({
                "global_orient": go,                                   # (T, 3)
                "body_pose": p["body_pose"].numpy().astype(np.float32),  # (T, 63)
                "betas": p["betas"].numpy().astype(np.float32),        # (T, 10)
                "transl": tr,                                          # (T, 3)
                "ankle_conf": float(kp2d[:, [15, 16], 2].mean()),
            })
            del pred
            torch.cuda.empty_cache()
            print(f"[shots] predicted {i + 1}/{n_shots}: {go.shape[0]} frames")
        del model
        torch.cuda.empty_cache()

    # ---- stitch: equal-length slerp crossfade at every seam ----
    # Window of N frames centred on the cut: both sides are sampled at HALF rate
    # (A's last N/2 frames and B's first N/2 frames, stretched over the N output
    # frames, ramp 0->1), then B plays pure from its frame N/2. Every source frame is
    # consumed exactly once, so total frames = sum of shot frames — the output
    # timeline stays sample-aligned with the audio (no cumulative drift; only the
    # 0.4 s window itself is locally time-warped). First/last shots lend no frames at
    # the clip ends; shots shorter than N halve the window.
    ankle_conf = [s["ankle_conf"] for s in seqs]
    quats = [to_quat(np.concatenate([s["global_orient"][:, None, :],
                                     s["body_pose"].reshape(-1, 21, 3)], axis=1))
             for s in seqs]  # per shot (T, 22, 4)
    bridged = bridge_hidden_legs(quats, ankle_conf)
    for i, (c, b) in enumerate(zip(ankle_conf, bridged)):
        print(f"[shots] shot {i:02d} ankle conf {c:.3f}{'  -> legs bridged (feet hidden)' if b else ''}")
    N = max(2, int(round(FADE_S * fps)))
    N += N % 2
    pieces_q: list[np.ndarray] = []
    pieces_t: list[np.ndarray] = []
    pieces_b: list[np.ndarray] = []
    seams = []
    prev_q, prev_t, prev_b = quats[0], seqs[0]["transl"], seqs[0]["betas"]
    out_len = 0
    for i in range(1, n_shots):
        cur_q, cur_t, cur_b = quats[i], seqs[i]["transl"], seqs[i]["betas"]
        h = N // 2
        if min(len(prev_q), len(cur_q)) < N:
            h = max(1, min(len(prev_q), len(cur_q)) // 2)
        n2 = 2 * h
        pieces_q.append(prev_q[:-h])
        pieces_t.append(prev_t[:-h])
        pieces_b.append(prev_b[:-h])
        out_len += len(prev_q) - h
        half = np.arange(n2, dtype=np.float64) * 0.5          # 0, 0.5, ..., h-0.5
        ramp = (np.arange(n2, dtype=np.float32) / (n2 - 1))[:, None, None]  # 0 -> 1
        blend_q = slerp(sample_quat(prev_q, len(prev_q) - h + half),
                        sample_quat(cur_q, half), ramp)
        blend_t = ((1.0 - ramp[:, 0]) * sample_vec(prev_t, len(prev_t) - h + half)
                   + ramp[:, 0] * sample_vec(cur_t, half))
        blend_b = ((1.0 - ramp[:, 0]) * sample_vec(prev_b, len(prev_b) - h + half)
                   + ramp[:, 0] * sample_vec(cur_b, half))
        pre = quat_angle_deg(prev_q[-1], cur_q[0])
        around = np.concatenate([prev_q[-h - 1:-h], blend_q, cur_q[h:h + 1]])
        post = quat_angle_deg(around[:-1], around[1:]) * fps  # deg/s per joint
        seams.append({
            "afterShot": i - 1,
            "inputTimeS": round(bounds[i], 3),
            "outFrame": out_len,
            "outTimeS": round((out_len + h) / fps, 3),
            "fadeFrames": n2,
            "preJumpDeg": {"max": round(float(pre.max()), 1), "mean": round(float(pre.mean()), 1)},
            "postFadeMaxDegS": round(float(post.max()), 1),
        })
        pieces_q.append(blend_q)
        pieces_t.append(blend_t.astype(np.float32))
        pieces_b.append(blend_b.astype(np.float32))
        out_len += n2
        prev_q, prev_t, prev_b = cur_q[h:], cur_t[h:], cur_b[h:]
    pieces_q.append(prev_q)
    pieces_t.append(prev_t)
    pieces_b.append(prev_b)

    out_q = np.concatenate(pieces_q)
    transl = np.concatenate(pieces_t).astype(np.float32)
    betas_seq = np.concatenate(pieces_b)                     # (T, 10), raw sidecar
    betas_med = np.median([np.median(s["betas"], axis=0) for s in seqs], axis=0).astype(np.float32)
    poses_aa = to_aa(out_q)                                  # (T, 22, 3)
    global_orient = np.ascontiguousarray(poses_aa[:, 0])
    body_pose = np.ascontiguousarray(poses_aa[:, 1:].reshape(len(poses_aa), 63))
    T = len(poses_aa)

    poses55 = save_npz(out_npz, global_orient, body_pose, betas_med, transl, fps)
    torch.save({
        "smpl_params_global": {
            "global_orient": torch.from_numpy(global_orient),
            "body_pose": torch.from_numpy(body_pose),
            "betas": torch.from_numpy(betas_seq),
            "transl": torch.from_numpy(transl),
        },
        "shots": [{"startS": bounds[i], "endS": bounds[i + 1]} for i in range(n_shots)],
        "seams": seams,
    }, out_npz.with_suffix(".raw.pt"))

    assert np.isfinite(poses55).all(), "non-finite values in poses!"
    steps = quat_angle_deg(out_q[:-1], out_q[1:]) * fps
    aa_step = poses55[1:, :22] - poses55[:-1, :22]
    ang_speed = np.linalg.norm(aa_step, axis=-1).mean() * fps

    meta = {
        "video": str(video_path), "fps": fps,
        "sceneThreshold": SCENE_THRESHOLD, "minShotS": MIN_SHOT_S, "fadeS": FADE_S,
        "stitch": "equal-length: N-frame half-speed slerp window per cut, no frames consumed",
        "rawCuts": [round(c, 3) for c in raw_cuts],
        # equal-length stitch: output frame t is input time t/fps — exact audio alignment
        "shots": [{"i": i, "startS": round(bounds[i], 3), "endS": round(bounds[i + 1], 3),
                   "srcFrames": int(get_video_lwh(shot_paths[i])[0]),
                   "outStartFrame": int(round(bounds[i] * fps)),
                   "outEndFrame": int(round(bounds[i + 1] * fps)),
                   "ankleConf": round(ankle_conf[i], 3), "legsBridged": bridged[i]}
                  for i in range(n_shots)],
        "seams": seams,
        "outFrames": T,
        "postMaxDegS": round(float(steps.max()), 1),
    }
    shots_json.write_text(json.dumps(meta, indent=2))

    print(f"OK  {out_npz}")
    print(f"    frames={T}  fps={fps}  shots={n_shots}  poses=(T, 165) flat, 55 joints x 3")
    print(f"    finite=True  mean joint angular speed={np.degrees(ang_speed):.1f} deg/s")
    print("    seam  in-video-t  pre-jump max/mean deg  post-fade max deg/s")
    for s in seams:
        print(f"    {s['afterShot'] + 1:>4}  {s['inputTimeS']:>9.1f}  "
              f"{s['preJumpDeg']['max']:>10.1f}/{s['preJumpDeg']['mean']:<5.1f}  "
              f"{s['postFadeMaxDegS']:>10.1f}")
    print(f"    shots.json: {shots_json}")
    print(f"    raw GVHMR output: {out_npz.with_suffix('.raw.pt')}")


if __name__ == "__main__":
    main()
