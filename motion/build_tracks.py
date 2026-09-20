"""Generate and retarget a motion track for every arena clip that lacks one.

    .venv/bin/python build_tracks.py [--force]

Generated motion is tied to the audio it was generated from, so a generated candidate only
exists for a round whose line already has a track. This walks `public/audio/arena/*.wav` and
writes `public/motion/<stem>.<model>.json`, which is what the arena serves and the player
turns into an animation clip. The model is loaded once for the whole batch, because loading
it costs about a minute and generating costs seconds.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
AUDIO = REPO / "public" / "audio" / "arena"
TRACKS = REPO / "public" / "motion"
AVATAR = REPO / "public" / "assets" / "model-clips.glb"
MODEL = "emage"


def main() -> None:
    force = "--force" in sys.argv
    wavs = sorted(AUDIO.glob("*.wav"))
    if not wavs:
        print(f"no arena audio under {AUDIO}")
        return
    TRACKS.mkdir(parents=True, exist_ok=True)
    todo = [w for w in wavs if force or not (TRACKS / f"{w.stem}.{MODEL}.json").exists()]
    print(f"{len(wavs)} clips, {len(todo)} to build")
    if not todo:
        return

    sys.path.insert(0, str(HERE))
    from generate import generate, load
    from retarget import read_glb, retarget

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.monotonic()
    model, motion_vq = load(device)
    print(f"model loaded in {time.monotonic() - started:.1f}s", flush=True)
    gltf, _ = read_glb(AVATAR)

    for wav in todo:
        started = time.monotonic()
        npz, seconds = generate(model, motion_vq, wav, HERE / "out", device)
        import numpy as np

        data = np.load(npz, allow_pickle=True)
        bones, quats = retarget(data["poses"], gltf)
        out = TRACKS / f"{wav.stem}.{MODEL}.json"
        out.write_text(json.dumps({
            "fps": int(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else 30,
            "frames": int(quats.shape[0]),
            "bones": bones,
            "quats": np.round(quats, 4).reshape(-1).tolist(),
            "model": MODEL,
            "source": wav.name,
        }))
        print(f"  {wav.name}: {seconds:.1f}s of motion in {time.monotonic() - started:.1f}s "
              f"-> {out.name} ({out.stat().st_size / 1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
