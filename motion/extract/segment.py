"""segment.py <audio.wav> <track.json> <segments.json> <public segments.json> —
sentence-level gesture segments + semantic embeddings for a profile track.

Transcribes the clip's audio with faster-whisper (word timestamps), groups words
into sentences on terminal punctuation, merges sentences shorter than 0.8 s into
the previous one, maps each sentence to a frame range of the track (fps/frames
read from track.json, clamped to [0, frames]), and embeds the sentence text with
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384-d, L2-normalized).

Two outputs, same structure (docs/protocol.md §5): the data-dir file keeps the
embeddings (server-side runtime matching), the public file drops them (frontend
preview/slicing). If whisper recognizes no speech at all, both files are written
with an empty "segments" list and the script exits 0 (warning on stderr) so the
profile pipeline does not fail on speechless clips.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
WHISPER_MODEL = "small"
MIN_SENTENCE_S = 0.8
SENTENCE_END = re.compile(r"[.!?\u3002\uFF01\uFF1F]$")


def _preload_cuda12_libs() -> None:
    """ctranslate2 links against CUDA 12 while this venv's torch is cu13; the
    nvidia-*-cu12 pip wheels ship libcublas/libcudnn.so.12 under site-packages,
    but nothing puts them on the loader path, so preload them globally."""
    import ctypes
    import site

    for sp in site.getsitepackages():
        for pkg, libs in (("cublas", ("libcublasLt.so.12", "libcublas.so.12")),
                          ("cudnn", ("libcudnn.so.9",))):
            for lib in libs:
                p = Path(sp) / "nvidia" / pkg / "lib" / lib
                if p.exists():
                    try:
                        ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass


def _transcribe_once(model, audio_path: str) -> list[dict]:
    words: list[dict] = []
    segments, info = model.transcribe(audio_path, word_timestamps=True)
    print(f"segment.py: language={info.language} p={info.language_probability:.2f}", file=sys.stderr)
    for seg in segments:
        for w in seg.words or []:
            if w.start is None or w.end is None:
                continue
            words.append({"start": float(w.start), "end": float(w.end), "word": w.word})
    return words


def transcribe(audio_path: str) -> list[dict]:
    """Word-level transcription; GPU if it works, CPU/int8 fallback otherwise.
    Construction may succeed while the first kernel fails (missing CUDA libs),
    so the whole attempt — not just WhisperModel() — decides the fallback."""
    _preload_cuda12_libs()
    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
        return _transcribe_once(model, audio_path)
    except Exception as e:
        print(f"segment.py: cuda transcription failed ({e}), falling back to cpu/int8", file=sys.stderr)
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _transcribe_once(model, audio_path)


def group_sentences(words: list[dict]) -> list[dict]:
    """Group words on terminal punctuation, then merge <0.8 s sentences forward."""
    sentences: list[dict] = []
    current: dict | None = None
    for w in words:
        if current is None:
            current = {"startS": w["start"], "endS": w["end"], "text": w["word"]}
        else:
            current["endS"] = w["end"]
            current["text"] += w["word"]
        if SENTENCE_END.search(w["word"].strip()):
            sentences.append(current)
            current = None
    if current is not None:
        sentences.append(current)

    merged: list[dict] = []
    for s in sentences:
        s["text"] = s["text"].strip()
        if merged and s["endS"] - s["startS"] < MIN_SENTENCE_S:
            merged[-1]["endS"] = s["endS"]
            merged[-1]["text"] = (merged[-1]["text"] + " " + s["text"]).strip()
        else:
            merged.append(dict(s))
    return merged


def embed(texts: list[str]) -> list[list[float]]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32).tolist()


def main() -> int:
    audio_path, track_path, out_data, out_public = sys.argv[1:5]

    with open(track_path) as f:
        track = json.load(f)
    fps = track["fps"]
    frames = track["frames"]

    words = transcribe(audio_path)
    sentences = group_sentences(words)
    if not sentences:
        print("segment.py: whisper recognized no speech; writing empty segments", file=sys.stderr)
        payload = {"version": 1, "fps": fps, "frames": frames, "embedModel": EMBED_MODEL, "segments": []}
        for out in (out_data, out_public):
            with open(out, "w") as f:
                json.dump(payload, f)
        return 0

    embeddings = embed([s["text"] for s in sentences])

    segments = []
    for i, (s, emb) in enumerate(zip(sentences, embeddings)):
        start_frame = max(0, min(frames, round(s["startS"] * fps)))
        end_frame = max(0, min(frames, round(s["endS"] * fps)))
        segments.append({
            "i": i,
            "startS": round(s["startS"], 3),
            "endS": round(s["endS"], 3),
            "startFrame": start_frame,
            "endFrame": end_frame,
            "text": s["text"],
            "embedding": emb,
        })

    payload = {"version": 1, "fps": fps, "frames": frames, "embedModel": EMBED_MODEL, "segments": segments}
    with open(out_data, "w") as f:
        json.dump(payload, f)
    public = {**payload, "segments": [{k: v for k, v in s.items() if k != "embedding"} for s in segments]}
    with open(out_public, "w") as f:
        json.dump(public, f)

    print(f"segment.py: {len(segments)} segments, {len(embeddings[0])}-d embeddings", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
