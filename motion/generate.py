"""Turn one wav into generated body motion with EMAGE.

The upstream `test_emage_audio.py` is an inference loop wrapped in a renderer: its
visualisation path pulls in pytorch3d, mmcv and chumpy, all pinned to CUDA 11.8 / Python 3.9
wheels that will not build here. The generation itself needs only torch, transformers,
librosa, numpy and smplx, so this is that path and nothing else. Output is the same
`beat_format_save` npz: SMPL-X axis-angle poses, FLAME expression and root translation at
the model's own frame rate.

    .venv/bin/python generate.py <in.wav> [out_dir]

EMAGE is a masked-audio-gesture transformer over VQ tokens, not a diffusion model; it is here
because its weights are downloadable and its inference path is clean, which makes it the
cheapest way to prove the whole pipeline (audio -> motion -> retarget -> avatar). A diffusion
model is the next candidate and plugs in at the same seam.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent / "pantomatrix"
sys.path.insert(0, str(REPO))

import librosa  # noqa: E402
import torch  # noqa: E402

from emage_utils.motion_io import beat_format_save  # noqa: E402
from models.emage_audio import (  # noqa: E402
    EmageAudioModel,
    EmageVAEConv,
    EmageVQModel,
    EmageVQVAEConv,
)

HF = "H-Liu1997/emage_audio"


def load(device: torch.device):
    part = lambda name: EmageVQVAEConv.from_pretrained(HF, subfolder=f"emage_vq/{name}").to(device)  # noqa: E731
    motion_vq = EmageVQModel(
        face_model=part("face"), upper_model=part("upper"),
        lower_model=part("lower"), hands_model=part("hands"),
        global_model=EmageVAEConv.from_pretrained(HF, subfolder="emage_vq/global").to(device),
    ).to(device).eval()
    model = EmageAudioModel.from_pretrained(HF).to(device).eval()
    return model, motion_vq


def generate(model, motion_vq, wav: Path, out_dir: Path, device: torch.device) -> Path:
    import torch.nn.functional as F

    sr, pose_fps = model.cfg.audio_sr, model.cfg.pose_fps
    audio, _ = librosa.load(str(wav), sr=sr)
    audio_t = torch.from_numpy(audio).to(device).unsqueeze(0)
    speaker_id = torch.zeros(1, 1).long().to(device)
    with torch.no_grad():
        trans = torch.zeros(1, 1, 3).to(device)
        latent = model.inference(audio_t, speaker_id, motion_vq, masked_motion=None, mask=None)
        cfg = model.cfg
        pick = lambda key, use, cls: latent[key] if use > 0 and cls == 0 else None  # noqa: E731
        index = lambda key, cls: torch.max(F.log_softmax(latent[key], dim=2), dim=2)[1] if cls > 0 else None  # noqa: E731
        pred = motion_vq.decode(
            face_latent=pick("rec_face", cfg.lf, cfg.cf),
            upper_latent=pick("rec_upper", cfg.lu, cfg.cu),
            hands_latent=pick("rec_hands", cfg.lh, cfg.ch),
            lower_latent=pick("rec_lower", cfg.ll, cfg.cl),
            face_index=index("cls_face", cfg.cf),
            upper_index=index("cls_upper", cfg.cu),
            hands_index=index("cls_hands", cfg.ch),
            lower_index=index("cls_lower", cfg.cl),
            get_global_motion=True, ref_trans=trans[:, 0],
        )
    motion = pred["motion_axis_angle"]
    frames = motion.shape[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{wav.stem}_emage.npz"
    beat_format_save(
        str(out),
        motion.cpu().numpy().reshape(frames, -1),
        upsample=30 // pose_fps,
        expressions=pred["expression"].cpu().numpy().reshape(frames, -1),
        trans=pred["trans"].cpu().numpy().reshape(frames, -1),
    )
    return out, frames / pose_fps


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    wav = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(__file__).resolve().parent / "out"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    started = time.monotonic()
    model, motion_vq = load(device)
    print(f"weights loaded in {time.monotonic() - started:.1f}s", flush=True)
    started = time.monotonic()
    out, seconds = generate(model, motion_vq, wav, out_dir, device)
    print(f"generated {seconds:.2f}s of motion in {time.monotonic() - started:.2f}s -> {out}")


if __name__ == "__main__":
    main()
