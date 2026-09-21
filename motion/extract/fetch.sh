#!/usr/bin/env bash
# fetch.sh <youtube-url> [start_seconds] [duration_seconds]
# Downloads best mp4 via yt-dlp and trims/normalizes to a 30 fps clip under videos/.
# Audio is kept so metrics.py's wav can be extracted from the same clip.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
YTDLP="$HERE/.venv/bin/yt-dlp"
OUT_DIR="$HERE/videos"
mkdir -p "$OUT_DIR"

URL="$1"
START="${2:-0}"
DUR="${3:-30}"

ID="$("$YTDLP" --get-id "$URL")"
RAW="$OUT_DIR/${ID}_raw.mp4"
OUT="$OUT_DIR/${ID}_${START}s_${DUR}s.mp4"

if [ ! -f "$RAW" ]; then
    "$YTDLP" -f "bv[ext=mp4][vcodec^=avc1][height<=720]+ba[ext=m4a]/bv[ext=mp4]+ba/b" \
        --merge-output-format mp4 -o "$RAW" "$URL"
fi

# Trim, force 30 fps + yuv420p so downstream tools see a uniform 30 fps stream.
ffmpeg -y -hide_banner -loglevel error \
    -ss "$START" -t "$DUR" -i "$RAW" \
    -vf fps=30 -pix_fmt yuv420p \
    -c:v libx264 -crf 20 -preset fast -c:a aac \
    "$OUT"

echo "$OUT"
