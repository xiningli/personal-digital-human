#!/usr/bin/env bash
# EMAGE motion (SMPL-X npz from generate.py) -> the speaking track the player loops.
#
#     motion/emage_track.sh <motion_emage.npz> <out.track.json>
#
# Pipeline chosen 2026-11-06 (owner picked it over the TEDx track after a side-by-side):
#   1. retarget.py      SMPL-X -> MetaPerson; neutrals align bone direction AND hinge axis
#   2. refine_track.py  --alpha-arms 0: hips/torso recentering only. The arm recentering is a
#                       constant rotation that folded nearly straight elbows backwards
#                       (joint_limits.py: 27 deg on EMAGE, 40 deg on the TEDx track); EMAGE's
#                       arms come from mocap and need no recentering.
#   3. ground_track.py  feet planted, legs by hinge IK, loop seam closed
#   4. gates            verify_retarget, check_track, joint_limits; any failure exits nonzero
set -euo pipefail
npz=$1 out=$2
M=$(cd "$(dirname "$0")" && pwd)
G=$M/../public/assets/model-clips.glb
P=$M/.venv/bin/python
cd "$M"
$P retarget.py "$npz" "$G" "$out"
$P verify_retarget.py "$npz" "$out" --check | tail -2
$P refine_track.py "$out" --alpha-arms 0 | tail -1
$P ground_track.py "$out" | tail -2
$P check_track.py "$G" "$out" --check | tail -10
$P joint_limits.py "$out" | tail -8
