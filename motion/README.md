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

## Next

Retarget SMPL-X (55 joints) onto the avatar's Mixamo skeleton, then register the track as an
arena candidate so it can be ranked against the motion capture. Joint correspondence is
direct for the body — pelvis→Hips, spine1..3→Spine..Spine2, collar/shoulder/elbow/wrist→
Shoulder/Arm/ForeArm/Hand, hips/knees/ankles→UpLeg/Leg/Foot — and the fingers map one to one.
Foot grounding still applies afterwards: generated motion is kinematic and knows nothing
about the floor.
