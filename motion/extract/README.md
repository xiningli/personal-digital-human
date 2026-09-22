# motion/extract — video → SMPL-X motion via GVHMR

YouTube URL → trimmed 30 fps clip → GVHMR SMPL-X prediction → EMAGE-contract .npz
that `motion/retarget.py` consumes unchanged.

## Usage

```bash
# 1. download + trim (keeps audio for metrics.py)
./fetch.sh <youtube-url> [start_seconds] [duration_seconds]
#    -> videos/<id>_<start>s_<dur>s.mp4

# 2. extract motion (headless, static camera, no DPVO, no rendering)
.venv/bin/python extract.py videos/<clip>.mp4 out/<clip>.npz
#    -> out/<clip>.npz  (EMAGE format) + out/<clip>.raw.pt (full GVHMR output)

# 2b. edited clips (talks that cut between a medium shot and close-ups): the profile
#     pipeline routes here automatically when scene detection finds >=2 cuts
.venv/bin/python extract_shots.py videos/<clip>.mp4 out/<clip>.npz out/<clip>.shots.json
#    -> same npz contract + .raw.pt + shots.json (shot list, fade positions, per-seam
#       smoothness: pre-fade pose jump vs post-fade peak angular velocity)

# 3. retarget + verify with the existing tooling (from motion/, its own venv)
cd .. && .venv/bin/python retarget.py extract/out/<clip>.npz ../public/assets/model-clips.glb
.venv/bin/python verify_retarget.py extract/out/<clip>.npz extract/out/<clip>.track.json --check
ffmpeg -y -i extract/videos/<clip>.mp4 -vn -ar 16000 -ac 1 extract/out/<clip>.wav
.venv/bin/python metrics.py --check extract/out/<clip>.npz extract/out/<clip>.wav

# 4. sentence segments + embeddings (this venv; GPU whisper, CPU/int8 fallback)
cd extract && .venv/bin/python segment.py out/<clip>.wav <clip>.track.json \
    ../../data/profiles/<profile-id>/segments.json \
    ../../public/motion/profile-<profile-id>.segments.json
```

## extract_shots.py — shot-aware extraction for edited clips

Same output contract as extract.py, for clips with real editing (TED-style talks cut
between a full-body medium shot and close-ups every ~10-15 s). Pipeline:

1. ffmpeg scene detection (`select=gt(scene,0.3)`, pts_time from showinfo); shots
   shorter than 1.5 s are merged into their longer neighbour.
2. Each shot is cut to `videos/shots/<stem>.shotNN.mp4` (30 fps, re-encoded) and
   predicted separately — one process, one GVHMR model load, tracker/ViTPose/feature
   extractor built once and reused, CUDA cache dropped between steps like
   extract.py's run_preprocess. The static-camera assumption holds per shot.
3. Leg repair: shots where ViTPose ankle confidence collapses (< 0.3; close-ups
   measure ~0.1, full-body shots ~0.93) have no leg information — GVHMR hallucinates
   legs there and they float, failing metrics.py's foot-float gate and popping the
   hip at the seam. Their leg rotations (hips/knees/ankles/feet) are replaced with
   the slerp bridge between the nearest feet-visible shots' boundary stances; upper
   body and global_orient stay as predicted. Per-shot ankle confidence and the
   bridged flag are logged and stored in shots.json.
4. Stitching (equal-length, audio-aligned): at every cut an N-frame window (0.4 s)
   centred on the cut samples both sides at half rate — the outgoing shot's last N/2
   frames and the incoming shot's first N/2 stretched over the N output frames —
   quaternion-slerped (axis-angle → quat → slerp → back) on a linear ramp (a cosine
   ease crests 1.57x the average slerp rate mid-fade and can breach the 600 deg/s
   gate), root translation lerped on the same ramp; the incoming shot then plays pure
   from its frame N/2. Every source frame is consumed exactly once: output length is
   the exact sum of shot lengths and output frame t is input time t/fps, so the
   whisper segment mapping never drifts. Betas: median over shots of each shot's
   per-frame median. The world-frame yaw fix of extract.py is applied per shot.

Sidecars: `<out>.raw.pt` (stitched smpl_params_global + shot/seam metadata) and
shots.json — raw cuts, merged shot bounds, per-seam fade positions (in output frames)
and smoothness: "preJumpDeg" (the discontinuity a naive concat would have had) vs
"postFadeMaxDegS" (worst adjacent-frame angular velocity after the fade; must stay
well under check_track.py's 600 deg/s teleport gate). The same table prints to stdout.

The profile pipeline (lib/profiles.ts) chooses automatically: one ffmpeg scene-detect
pass on the trimmed clip, >=2 cuts → extract_shots.py, else extract.py; the decision
is logged to run.log.

## segment.py — audio + track → sentence segments
Transcribes the clip's audio with faster-whisper (model `small`, word timestamps,
auto language; CUDA if usable, else CPU/int8), groups words into sentences on
terminal punctuation `[.!?.!?]`, merges sentences shorter than 0.8 s into the
previous one, maps each sentence to a frame range (`round(t * fps)`, fps/frames
from track.json, clamped to `[0, frames]`) and embeds the text with
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d, L2-normalized).

Writes the segments contract (docs/protocol.md §5) twice: the data-dir file keeps
`embedding` per segment, the public file drops it. No recognizable speech → both
files written with `"segments": []`, exit 0 (warning on stderr) — the profile
pipeline never fails on segmentation.

## npz contract

`poses` (T, 165) float32 — flat 55 SMPL-X joints x axis-angle, joint order as in
retarget.py. GVHMR predicts joints 0–21 only (global_orient + body_pose); jaw, eyes
and all finger joints (22–54) are zero. The world frame is pre-rotated by
`WORLD_YAW_FIX = π` (in extract.py): with static_cam=true GVHMR faces the person -Z
while the stage expects the avatar facing +Z toward the camera — verified visually
in the player. `betas` (300,) is zero by design (EMAGE
convention; retarget.py assumes a zero-beta rest pose — storing the predicted shape
makes verify_retarget report a constant 1–3° rest-direction offset). The GVHMR
shape is kept in `betas_gvhmr` and the full per-frame output (betas, transl,
camera) in the `.raw.pt` sidecar. `mocap_frame_rate` = clip fps (30; fetch.sh
normalizes).

## Environment notes

- `.venv`: python 3.11, torch 2.14.0+cu130 (Blackwell-compatible), deps from
  GVHMR requirements.txt with torch/torchvision/numpy pins relaxed.
- faster-whisper + sentence-transformers for segment.py; ctranslate2 links CUDA
  12, so `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` are installed alongside torch's
  cu13 stack and preloaded by segment.py (CPU/int8 fallback if GPU still fails).
- pytorch3d: pure-python source copy in the venv (no compiled `_C`); only
  `pytorch3d.transforms` (pure torch) is exercised. Rendering stays unavailable.
- ultralytics upgraded to 8.4.x (8.2.42's torch.load breaks on torch>=2.6
  weights_only default); av pinned to 13.0.0 (imageio pyav writer API).
- DPVO skipped; static camera assumed (`--static_cam` equivalent).

## Local patches inside GVHMR/

1. `hmr4d/utils/geo_transform.py` — `import pytorch3d.ops.knn` moved lazily into
   the one function using it (render-only path); the top-level import requires
   the compiled `_C` extension we don't build.
2. `hmr4d/configs/store_gvhmr.py` — dataset/metric registrations wrapped in
   try/except ImportError; they pull in the renderer (needs `_C`) and are unused
   on the demo inference path.

## Checkpoints (downloaded from the GVHMR Google Drive folder)

`GVHMR/inputs/checkpoints/`: gvhmr/gvhmr_siga24_release.ckpt (164 MB),
hmr2/epoch=10-step=25000.ckpt (2.7 GB), vitpose/vitpose-h-multi-coco.pth (2.5 GB),
yolo/yolov8x.pt (137 MB), body_models/smplx/SMPLX_NEUTRAL.npz (copied from
motion/emage_evaltools SMPLX_NEUTRAL_2020.npz). SMPL_{GENDER}.pkl intentionally
absent — only the (skipped) render/eval paths need it.
