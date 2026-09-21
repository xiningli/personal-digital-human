#!/usr/bin/env bash
# Copy the avatar with the Mixamo clips baked in from the personal site (never committed there
# or here; see THIRD_PARTY.md). Falls back to the clipless asset if the clip one is absent.
set -euo pipefail
cd "$(dirname "$0")/.."
# personal-site is a sibling checkout under the same digital-human directory.
SITE="${SITE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)/personal-site}"
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

# The Blender retargeting that produces this asset does not guarantee well-formed rotation
# keyframes (see clean-avatar-animations.mjs for what was found and why this belongs here,
# once, rather than patched around at playback time): clean every import automatically.
node scripts/clean-avatar-animations.mjs public/assets/model-clips.glb

# Playability fixes, also once in the asset (see fix-avatar-motion.mjs): drop the retargeter's
# leading-transient frames, close the loop seam so repeats don't snap, and bake a hips
# translation track that keeps the planted foot from skating. Idempotent, so re-importing an
# already-fixed source is a no-op.
node scripts/fix-avatar-motion.mjs public/assets/model-clips.glb
