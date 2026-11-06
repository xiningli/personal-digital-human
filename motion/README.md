# motion

Generated body motion for the digital human. Audio in, joint rotations out.

## EMAGE (working)

```bash
.venv/bin/python generate.py <in.wav> [out_dir]
```

Produces `<name>_emage.npz`: SMPL-X axis-angle `poses` (frames × 55 joints × 3), FLAME
`expressions`, root `trans`, at 30 fps.

Measured on 2026-09-19, RTX 5070 Ti: weights load in 58 s, then 5.80 s of motion from a
5.8 s clip in 8.44 s. The arms, wrists and fingers are the most active joints, which is what
co-speech gesture should look like.

### What the setup needed

Upstream's `test_emage_audio.py` is an inference loop wrapped in a renderer, and the renderer
drags in pytorch3d, mmcv and chumpy pinned to CUDA 11.8 / Python 3.9 wheels that will not
build here. `generate.py` is the generation path alone: torch, transformers, librosa, numpy,
smplx, omegaconf. Two things had to be pinned or added:

- **`transformers==4.57.3`.** On 5.x, `from_pretrained` reaches for `all_tied_weights_keys`,
  which the model classes (written against 4.x) do not define.
- **`omegaconf`**, imported by the config module but absent from the inference requirements.

Torch is 2.14+cu130, because the RTX 5070 Ti is Blackwell (sm_120) and older builds have no
kernels for it. That is also why DiffSHEG cannot be run as shipped: it pins torch 1.13.1 /
CUDA 11.7 and would need porting to modern torch first.

## Not diffusion

EMAGE is a masked-audio-gesture transformer over VQ tokens. It is here because its weights
are downloadable and its inference path is clean, which makes it the cheapest way to prove
the pipeline end to end. A diffusion model is the next candidate and plugs in at the same
seam: anything that turns a wav into per-frame joint rotations.

## Testing a change instead of eyeballing it

```bash
.venv/bin/python metrics.py --check <motion.npz> <audio.wav>   # exit 1 on a failed gate
```

Three hard gates, chosen by scoring deliberately degraded copies of a real clip (the table is
in `metrics.py`): diversity catches a frozen body, foot skate and foot float catch motion the
floor would not allow. FGD is reported but never gates a single clip.

One metric was disqualified by that exercise. **Beat consistency scores time-shuffled motion
and pure jitter HIGHER than the real clip**, because it rewards sharp velocity changes near
audio onsets and noise has those everywhere. It is not a quality measure.

## Retargeting onto the avatar

```bash
.venv/bin/python retarget.py <motion.npz> ../public/assets/model-clips.glb
.venv/bin/python verify_retarget.py <motion.npz> <track.json> --check   # exit 1 above 1°
.venv/bin/python build_tracks.py            # every arena clip that lacks a track
```

What transfers between two skeletons is each joint's *global* orientation, re-expressed in
the target's rest frame. SMPL-X makes that cheap: its zero pose has every joint frame at
identity, so a joint's composed global rotation is exactly the delta it has moved through.

The neutral it is applied to is not the bind pose, and getting that wrong is invisible until
measured. `verify_retarget.py` runs forward kinematics on both skeletons and compares limb
directions, which separates the two failure modes: a **constant** error per limb is a
rest-pose mismatch, a **varying** one is mangled motion. Two bugs were found that way.

| version | worst limb error | cause |
|---|---|---|
| aligned to the rig's bind pose | 15.9° constant | SMPL-X rests in an A shape, the rig binds in a T |
| head aligned to SMPL-X's first child | large, head thrown back | that child is the jaw, pointing forward, while HeadTop_End points up |
| aligning only shared segments | **0.03°** | quaternion rounding in the JSON, nothing else |

Foot grounding still applies afterwards: generated motion is kinematic and knows nothing
about the floor. The player does it for every motion source.

## Grounding and balance: feet first

```bash
.venv/bin/python ground_track.py <track.json>        # wired after refine_track (lib/profiles.ts)
.venv/bin/python check_track.py ../public/assets/model-clips.glb <track.json> --check
.venv/bin/python physics_audit.py ../public/assets/model-clips.glb <track.json> "clip:Talking"
```

Retargeting drops the root translation, so a raw track pins the pelvis in space and the
feet go wherever the leg rotations put them: measured on the TED stage track as shipped
until 2026-09-28, a planted foot travelled up to 31 cm inside one stance, the source's steps
became steps in place, and the runtime ground shift bobbed the body 6.6 cm. `ground_track.py`
solves the body from the ground up instead: both feet fixed at one natural stance (shared
with every capture clip in the asset — `scripts/ground-avatar-clips.mjs` runs the same pass
over them, which also undoes their crouched 34–43 cm retargeting stance), the pelvis carrying
a slow, bounded weight shift with the CoM and capture point held over the feet, legs by
two-bone IK with the knee over the toes, then hands cleared of the moved thighs. The track
gains a `hips` array that the player plays as `Hips.position`.

`balance.py` audits static balance on the exact FK the player runs: Dempster-fraction
CoM, support polygon from the grounded feet, LIPM capture point CP = com + com_vel/ω.
A **fall-risk run** — ≥1 s with CP >2 cm outside the polygon — is a pose no human holds
without falling. check_track gates it (0 runs; worst excursion ≤4 cm) and, since
2026-09-28, the per-stance travel of a planted foot (≤2 cm; the per-frame skate gate cannot
see slow drift). `physics_audit.py` reports the physical laws a viewer checks — planted-foot
travel, steps in place, CoM vertical acceleration, zero-moment point against the feet — side
by side for any tracks and clips. `correct_balance.py` (pelvis tilting, 2026-09-23) is
superseded by ground_track.py and no longer wired in.

| TED stage track, as played | before (pinned pelvis) | grounded | Talking capture |
|---|---|---|---|
| planted-foot travel per stance, max | 30.8 cm | 0.04 cm | 0.6 cm |
| steps in place per minute | 8.4 | 0 | 0 |
| runtime ground shift range | 6.6 cm | 0 | 0 |
| ZMP > 2 cm outside the feet | 16% of frames | 0.3% | 20% |
| torso lean lateral p95 | 10.8° | 8.0° | 7.1° |

## Next

A diffusion model as a second candidate, ranked against EMAGE and the captures in the arena.
DiffSHEG is the obvious one and needs porting to modern torch first: it pins torch 1.13.1 /
CUDA 11.7, which has no kernels for Blackwell.

## Extracting motion from video (GVHMR)

`extract/` turns a monocular video into the same EMAGE-contract npz: `fetch.sh` downloads
and trims a clip (YouTube via yt-dlp, or any local mp4), `extract.py` runs GVHMR headless
(static camera, no DPVO, no rendering) and writes `poses` (T, 165) with the world frame
pre-rotated so the person faces the stage's camera. retarget.py, verify_retarget.py and
metrics.py consume the result unchanged — see `extract/README.md`. First end-to-end clip:
a 30 s TED segment, worst limb error 0.03°, all metric gates pass, foot float 2 cm mean.
Fingers/jaw/eyes come out zero (GVHMR predicts no hands/face), so extracted clips read as
arm-and-torso gesture over the rig's bind hands. Preview any track in the browser with the
dev page `/player-check?track=<name>` (reads `public/motion/<name>.track.json`).
