#!/usr/bin/env node
/**
 * Export per-bone mean local rotations of an avatar clip, for refine_track.py to recenter
 * a generated track against. The reference clip is "Talking" in model-clips.glb: an
 * actual-capture clip the owner approved, so its time-averaged pose is what "elegant" means
 * for this rig (forearms ~40 deg off rest instead of the GVHMR tracks' ~85).
 *
 * Rotation channels are resampled onto a 30 fps grid (same sampler as fix-avatar-motion.mjs),
 * then averaged as quaternions: sign-align every frame against the running sum, add,
 * normalize. (Markley's eigendecomposition converges to the same mean on these clips; the
 * direct average is exact for the near-constant bones and adequate for the gesturing ones.)
 *
 * Usage: node scripts/extract-ref-means.mjs <glb> <clipName> <out.json> [bone ...]
 *   With no bone list, exports the default refine set.
 */
import { NodeIO } from '@gltf-transform/core';
import * as THREE from 'three';

const FPS = 30;
const FINGER_BONES = ['Left', 'Right'].flatMap((s) =>
  ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'].flatMap((f) =>
    [1, 2, 3, 4].map((k) => `${s}Hand${f}${k}`)));
const DEFAULT_BONES = [
  'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
  'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
  'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
  'LeftUpLeg', 'LeftLeg', 'RightUpLeg', 'RightLeg',
  // Finger means feed refine_track.py's finger pass: Talking's time-averaged local
  // rotation per finger bone is the "natural speaking hand" base pose (bind is flat).
  ...FINGER_BONES,
];

function dot4(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; }

function makeSampler(times, values) {
  const n = times.length;
  const v = new Float32Array(values);
  for (let f = 1; f < n; f++) {
    if (dot4(v.subarray((f - 1) * 4, (f - 1) * 4 + 4), v.subarray(f * 4, f * 4 + 4)) < 0)
      for (let k = 0; k < 4; k++) v[f * 4 + k] = -v[f * 4 + k];
  }
  const qa = new THREE.Quaternion(), qb = new THREE.Quaternion();
  return (t, out) => {
    if (t <= times[0]) return out.fromArray(v, 0);
    if (t >= times[n - 1]) return out.fromArray(v, (n - 1) * 4);
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (times[mid] <= t) lo = mid; else hi = mid; }
    const f = (t - times[lo]) / (times[hi] - times[lo]);
    qa.fromArray(v, lo * 4); qb.fromArray(v, hi * 4);
    return out.copy(qa).slerp(qb, f);
  };
}

async function main() {
  const [glbPath, clipName, outPath, ...bones] = process.argv.slice(2);
  if (!glbPath || !clipName || !outPath) {
    console.error('usage: extract-ref-means.mjs <glb> <clipName> <out.json> [bone ...]');
    process.exit(1);
  }
  const wanted = bones.length ? bones : DEFAULT_BONES;

  const io = new NodeIO();
  const doc = await io.read(glbPath);
  const anim = doc.getRoot().listAnimations().find((a) => a.getName() === clipName);
  if (!anim) {
    console.error(`clip "${clipName}" not found; clips: ${doc.getRoot().listAnimations().map((a) => a.getName()).join(', ')}`);
    process.exit(1);
  }

  let dur = 0;
  const rotChannels = [];
  for (const ch of anim.listChannels()) {
    const s = ch.getSampler();
    if (!s?.getInput() || !s?.getOutput()) continue;
    if (ch.getTargetPath() !== 'rotation') continue;
    const times = s.getInput().getArray();
    const out = s.getOutput().getArray();
    if (out.length !== times.length * 4) continue;
    dur = Math.max(dur, times[times.length - 1]);
    rotChannels.push({ name: ch.getTargetNode()?.getName(), sampler: makeSampler(times, out) });
  }
  const gridLen = Math.round(dur * FPS) + 1;

  const means = {};
  const q = new THREE.Quaternion();
  for (const { name, sampler } of rotChannels) {
    if (!name || !wanted.includes(name)) continue;
    const sum = [0, 0, 0, 0];
    for (let f = 0; f < gridLen; f++) {
      sampler(f / FPS, q);
      const a = [q.x, q.y, q.z, q.w];
      if (dot4(a, sum) < 0) for (let k = 0; k < 4; k++) a[k] = -a[k];
      for (let k = 0; k < 4; k++) sum[k] += a[k];
    }
    const n = Math.hypot(...sum);
    means[name] = sum.map((v) => Number((v / n).toFixed(6)));
  }
  const missing = wanted.filter((b) => !means[b]);
  if (missing.length) console.error(`warning: no rotation channel for: ${missing.join(', ')}`);

  const fs = await import('node:fs');
  fs.writeFileSync(outPath, JSON.stringify({
    source: glbPath.replace(/^.*\//, ''), clip: clipName, fps: FPS, frames: gridLen,
    // xyzw per bone, time-averaged local rotation of the clip
    means,
  }, null, 1));
  console.log(`${Object.keys(means).length} bones x ${gridLen} frames from "${clipName}" -> ${outPath}`);
}

await main();
