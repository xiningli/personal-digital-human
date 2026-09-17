#!/usr/bin/env bash
# Copy the avatar with the Mixamo clips baked in from the personal site (never committed there
# or here; see THIRD_PARTY.md). Falls back to the clipless asset if the clip one is absent.
set -euo pipefail
cd "$(dirname "$0")/.."
SITE="${SITE_ROOT:-$HOME/personal-site}"
mkdir -p public/assets
if [ -f "$SITE/public/assets/model-clips.glb" ]; then
  cp "$SITE/public/assets/model-clips.glb" public/assets/model-clips.glb
  echo "copied model-clips.glb ($(du -h public/assets/model-clips.glb | cut -f1))"
elif [ -f "$SITE/public/assets/model.glb" ]; then
  cp "$SITE/public/assets/model.glb" public/assets/model-clips.glb
  echo "clip asset absent; copied the clipless model.glb instead (procedural body only)"
else
  echo "no avatar found under $SITE/public/assets" >&2; exit 1
fi
