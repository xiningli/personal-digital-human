#!/usr/bin/env node
/**
 * Make the avatar's clips playable: fix, once in the asset itself, the three things that made
 * the rig read as seizing and floating. Runs after clean-avatar-animations.mjs (which repairs
 * corrupted rotation keyframes); this script assumes well-formed keys and fixes what is
 * well-formed but wrong to play:
 *
 * 1. Leading transient. Every clip's first ~4 frames (t <= 0.13s) carry the retargeter's
 *    convergence artifact: feet/legs/neck rotating at 10-19 rad/s (measured 2026-09-20 on this
 *    asset; motion after 0.2s stays <= ~11 rad/s). With looping playback that artifact fired
 *    every cycle. The leading frames are dropped, but only when such a transient is measured,
 *    so re-running this script on an already-fixed asset trims nothing.
 * 2. Loop seam. The clips' last-frame pose is far from their first-frame pose (measured up to
 *    2.75 rad on finger bones in "Breathing Idle", 0.86 rad on the forearm in "Talking"), so
 *    every loop restart snapped. The last 0.5s of every rotation track is blended toward the
 *    first frame's pose, making the wrap seamless.
 * 3. Foot skate. The clips were retargeted with the hips' translation locked, so the mocap's
 *    weight shifts became pure hip rotation and the planted foot swept sideways (measured:
 *    0.53-0.73 m of planted-foot slide per clip, up to 16 cm in a single frame -- the
 *    "floating balloon" the owner reported). This pass measures the skeleton forward-kinematically
 *    per frame, finds the support foot (lower toe, with hysteresis), and bakes a hips
 *    translation track that holds the support foot where it planted. Vertical grounding
 *    (previously the runtime-only `ground()` correction) is baked into the same track, so the
 *    asset stands on its own in any viewer; the runtime correction becomes a near-zero no-op.
 *
 * All tracks are resampled onto one 30 fps grid per clip and written to fresh accessors --
 * never mutated in place, because a clip's times accessor is shared across channels (see
 * clean-avatar-animations.mjs for the incident that taught this).
 *
 * Some clips are additionally scaled down in amplitude (AMPLITUDE below): "Breathing Idle"
 * sways the hips 16 cm side to side, which with the foot lock became a visible gorilla-like
 * weight-lurch the owner rejected. Scaling every rotation toward the clip's start pose keeps
 * the breathing alive but calm; the foot lock is derived afterwards, so it stays consistent.
 *
 * Usage: node scripts/fix-avatar-motion.mjs <in.glb> [out.glb]
 *   out.glb defaults to overwriting in.glb.
 */
import { NodeIO } from '@gltf-transform/core';
import { prune } from '@gltf-transform/functions';
import * as THREE from 'three';

const FPS = 30;
/** Leading transient: drop 4 frames at a time while the first 0.2s still exceeds this. */
const TRIM_VEL = 9; // rad/s; post-0.2s motion on this asset peaks ~11 rad/s, transients 15-19
const TRIM_FRAMES = 4;
const TRIM_MAX_PASSES = 3;
/** Loop closure: blend this much of the tail toward the first frame. */
const BLEND_S = 0.5;
/** Support-foot hysteresis: the other toe must be this much lower to take over. */
const SUPPORT_HYSTERESIS_M = 0.01;
/** Toe bone sits this far above the sole in the bind pose (player/ground.ts: SOLE_BELOW_TOE). */
const SOLE_BELOW_TOE = 0.0524;

/** Per-clip rotation-amplitude scale, applied after trimming, before loop closure and the
 *  foot lock (which then measures the already-scaled pose). */
const AMPLITUDE = { 'Breathing Idle': 0.4 };

const TOES = ['LeftToeBase', 'RightToeBase'];
/** Eyes and head-top are constant in these clips; excluding them keeps the trim trigger honest. */
const TRANSIENT_IGNORE = /Eye|HeadTop/;

function dot4(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; }
function smoothstep(t) { return t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t); }

/** Slerp-sample a quaternion track at time t, keeping sign continuity while scanning. */
function makeSampler(times, values) {
  const n = times.length;
  // Sign-fix the source once so slerp never crosses the double cover mid-segment.
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
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; times[mid] <= t ? lo = mid : hi = mid; }
    const f = (t - times[lo]) / (times[hi] - times[lo]);
    qa.fromArray(v, lo * 4); qb.fromArray(v, hi * 4);
    return out.copy(qa).slerp(qb, f);
  };
}

/** Max frame-to-frame angular velocity (rad/s) across tracks, within t <= window. */
function earlyVelocity(grid, tracks, window) {
  let max = 0;
  const dt = grid[1] - grid[0];
  const frames = Math.min(tracks[0].length, Math.floor(window / dt) + 1);
  for (const q of tracks) {
    for (let f = 1; f < frames; f++) {
      if (grid[f] > window) break;
      const d = 2 * Math.acos(Math.min(1, Math.abs(q[f - 1].dot(q[f]))));
      max = Math.max(max, d / dt);
    }
  }
  return max;
}

function buildSkeleton(doc) {
  const objByNode = new Map();
  for (const node of doc.getRoot().listNodes()) {
    const o = new THREE.Object3D();
    o.name = node.getName();
    o.position.fromArray(node.getTranslation());
    o.quaternion.fromArray(node.getRotation());
    o.scale.fromArray(node.getScale());
    objByNode.set(node, o);
  }
  const sceneRoot = new THREE.Object3D();
  for (const scene of doc.getRoot().listScenes())
    for (const child of scene.listChildren()) sceneRoot.add(objByNode.get(child));
  for (const node of doc.getRoot().listNodes())
    for (const child of node.listChildren()) objByNode.get(node).add(objByNode.get(child));
  return { objByNode, sceneRoot };
}

async function main() {
  const [inPath, outPath = inPath] = process.argv.slice(2);
  if (!inPath) { console.error('usage: fix-avatar-motion.mjs <in.glb> [out.glb]'); process.exit(1); }

  const io = new NodeIO();
  const doc = await io.read(inPath);
  const { objByNode, sceneRoot } = buildSkeleton(doc);
  const hips = sceneRoot.getObjectByName('Hips');
  const toes = TOES.map((n) => sceneRoot.getObjectByName(n));
  if (!hips || toes.some((t) => !t)) { console.error('rig lacks Hips/ToeBase bones'); process.exit(1); }
  const hipsBind = hips.position.clone();

  for (const anim of doc.getRoot().listAnimations()) {
    // --- gather and resample rotation tracks onto one grid ---
    // The grid spans the rotation data; untouched constant translation tracks may carry the
    // clip's original (longer) time range, and whatever sticks out past the grid becomes a
    // frozen tail inside three.js's computed clip duration -- so they are re-anchored below.
    let dur = 0;
    const rotChannels = [];
    const constChannels = [];
    for (const ch of anim.listChannels()) {
      const s = ch.getSampler();
      if (!s?.getInput() || !s?.getOutput()) continue;
      const times = s.getInput().getArray();
      if (ch.getTargetPath() === 'rotation' && s.getOutput().getArray().length === times.length * 4) {
        dur = Math.max(dur, times[times.length - 1]);
        rotChannels.push({ ch, sampler: makeSampler(times, s.getOutput().getArray()), node: ch.getTargetNode() });
      } else if (ch.getTargetPath() !== 'translation' || ch.getTargetNode()?.getName() !== 'Hips') {
        constChannels.push(ch);
      }
    }
    let gridLen = Math.round(dur * FPS) + 1;
    let tracks = rotChannels.map(({ sampler }) => {
      const q = [];
      for (let f = 0; f < gridLen; f++) q.push(sampler(f / FPS, new THREE.Quaternion()));
      return q;
    });

    // --- 1. leading transient: drop 4 frames at a time while the start is implausibly fast ---
    let trimmed = 0;
    const movingIdx = rotChannels.map((_, i) => i).filter((i) => !TRANSIENT_IGNORE.test(rotChannels[i].node?.getName() ?? ''));
    while (trimmed < TRIM_MAX_PASSES) {
      const times = [...Array(tracks[0].length).keys()].map((f) => f / FPS);
      if (earlyVelocity(times, movingIdx.map((i) => tracks[i]), 0.2) <= TRIM_VEL) break;
      tracks = tracks.map((q) => q.slice(TRIM_FRAMES));
      trimmed++;
    }
    gridLen = tracks[0].length;
    const grid = new Float32Array(gridLen);
    for (let f = 0; f < gridLen; f++) grid[f] = f / FPS;

    // --- 1b. amplitude: scale every rotation toward the clip's start pose. Marked in the
    // animation's extras, because unlike every other pass here this one is not naturally
    // idempotent -- a second run would shrink the motion again. ---
    const amp = AMPLITUDE[anim.getName()];
    let scaled = false;
    if (amp && amp !== 1 && !anim.getExtras()?.amplitudeScaled) {
      for (let i = 0; i < tracks.length; i++) {
        const q = tracks[i];
        const ref = q[0].clone();
        for (let f = 1; f < gridLen; f++) {
          if (ref.dot(q[f]) < 0) q[f].set(-q[f].x, -q[f].y, -q[f].z, -q[f].w);
          q[f] = ref.clone().slerp(q[f], amp);
        }
      }
      anim.setExtras({ ...anim.getExtras(), amplitudeScaled: true });
      scaled = true;
    }

    // --- 2. loop closure: blend the last BLEND_S seconds toward the first frame ---
    const blendFrom = grid[gridLen - 1] - BLEND_S;
    for (const q of tracks) {
      const q0 = q[0];
      for (let f = 0; f < gridLen; f++) {
        if (grid[f] < blendFrom) continue;
        q[f].slerp(q0, smoothstep((grid[f] - blendFrom) / BLEND_S));
      }
    }

    // --- 3. foot lock + grounding, baked into the hips' translation ---
    // Pose the skeleton per frame with the fixed rotations and measure the toes.
    const boneObjs = rotChannels.map(({ node }) => objByNode.get(node));
    const toeXZ = [], toeY = [[], []];
    const v = new THREE.Vector3();
    for (let f = 0; f < gridLen; f++) {
      for (let b = 0; b < boneObjs.length; b++) boneObjs[b].quaternion.copy(tracks[b][f]);
      sceneRoot.updateWorldMatrix(true, true);
      for (let t = 0; t < 2; t++) {
        toes[t].getWorldPosition(v);
        if (t === 0) toeXZ.push([v.x, v.z]);
        toeY[t].push(v.y);
        if (t === 1) toeXZ[f].push(v.x, v.z); // [Lx, Lz, Rx, Rz]
      }
    }
    // Support foot per frame: lower toe, with hysteresis so a near-tie doesn't flicker.
    const support = new Array(gridLen);
    let cur = toeY[0][0] <= toeY[1][0] ? 0 : 1;
    for (let f = 0; f < gridLen; f++) {
      const other = 1 - cur;
      if (toeY[other][f] < toeY[cur][f] - SUPPORT_HYSTERESIS_M) cur = other;
      support[f] = cur;
    }
    // Compensation: hold the support foot where it planted. At a support change the anchor
    // hands off in COMPENSATED space -- the new foot is simply held at the position it is
    // already displayed at -- so the compensation is continuous and the hips never pop.
    // (Easing the compensation back toward zero at a switch instead produced a measured 5 cm
    // whole-body jolt per weight transfer.) Blend to zero over the loop-closure window so the
    // wrap stays seamless.
    const comp = new Array(gridLen);
    let anchor = null;
    for (let f = 0; f < gridLen; f++) {
      const s = support[f];
      if (f === 0) { anchor = [toeXZ[0][s * 2], toeXZ[0][s * 2 + 1]]; comp[0] = [0, 0]; continue; }
      if (support[f - 1] !== s)
        anchor = [toeXZ[f][s * 2] + comp[f - 1][0], toeXZ[f][s * 2 + 1] + comp[f - 1][1]];
      comp[f] = [anchor[0] - toeXZ[f][s * 2], anchor[1] - toeXZ[f][s * 2 + 1]];
    }
    for (let f = 0; f < gridLen; f++) {
      if (grid[f] < blendFrom) continue;
      const w = 1 - smoothstep((grid[f] - blendFrom) / BLEND_S);
      comp[f][0] *= w; comp[f][1] *= w;
    }
    // Vertical: the offset the runtime ground() would apply, baked instead.
    const lift = new Array(gridLen);
    for (let f = 0; f < gridLen; f++) lift[f] = SOLE_BELOW_TOE - Math.min(toeY[0][f], toeY[1][f]);
    for (let f = 0; f < gridLen; f++) {
      if (grid[f] < blendFrom) continue;
      const w = smoothstep((grid[f] - blendFrom) / BLEND_S);
      lift[f] = lift[f] + (lift[0] - lift[f]) * w;
    }
    // Convert world-space delta into the hips' parent space per frame and write the track.
    const hipsParentInv = new THREE.Matrix4();
    const world = new THREE.Vector3();
    const hipsPos = new Float32Array(gridLen * 3);
    for (let f = 0; f < gridLen; f++) {
      for (let b = 0; b < boneObjs.length; b++) boneObjs[b].quaternion.copy(tracks[b][f]);
      sceneRoot.updateWorldMatrix(true, true);
      hipsParentInv.copy(hips.parent.matrixWorld).invert();
      hips.getWorldPosition(world);
      world.x += comp[f][0]; world.z += comp[f][1]; world.y += lift[f];
      world.applyMatrix4(hipsParentInv);
      hipsPos.set([world.x, world.y, world.z], f * 3);
    }

    // --- write back: fresh accessors everywhere, never mutate a shared one ---
    const timesAcc = doc.createAccessor(`${anim.getName()}-grid`).setType('SCALAR').setArray(grid);
    const gridEnd = grid[gridLen - 1];
    // Re-anchor constant tracks that still span the clip's original time range.
    for (const ch of constChannels) {
      const s = ch.getSampler();
      const times = s.getInput().getArray();
      if (times[times.length - 1] <= gridEnd) continue;
      const values = s.getOutput().getArray();
      const comps = ch.getTargetPath() === 'rotation' ? 4 : 3;
      const first = values.slice(0, comps), last = values.slice((times.length - 1) * comps, times.length * comps);
      const two = new Float32Array(comps * 2);
      two.set(first, 0); two.set(last, comps);
      s.setInput(doc.createAccessor(`${anim.getName()}-const-times`).setType('SCALAR').setArray(new Float32Array([0, gridEnd])));
      s.setOutput(doc.createAccessor(`${anim.getName()}-const-vals`).setType(ch.getTargetPath() === 'rotation' ? 'VEC4' : 'VEC3').setArray(two));
    }
    for (let i = 0; i < rotChannels.length; i++) {
      const values = new Float32Array(gridLen * 4);
      for (let f = 0; f < gridLen; f++) tracks[i][f].toArray(values, f * 4);
      const valuesAcc = doc.createAccessor(`${anim.getName()}-${rotChannels[i].node?.getName() ?? i}-rot`).setType('VEC4').setArray(values);
      rotChannels[i].ch.getSampler().setInput(timesAcc).setOutput(valuesAcc);
    }
    const hipsAcc = doc.createAccessor(`${anim.getName()}-Hips-pos`).setType('VEC3').setArray(hipsPos);
    let hipsChannel = null;
    for (const ch of anim.listChannels())
      if (ch.getTargetPath() === 'translation' && ch.getTargetNode()?.getName() === 'Hips') hipsChannel = ch;
    if (hipsChannel) hipsChannel.getSampler().setInput(timesAcc).setOutput(hipsAcc);
    else {
      const hipsNode = doc.getRoot().listNodes().find((n) => n.getName() === 'Hips');
      const sampler = doc.createAnimationSampler().setInput(timesAcc).setOutput(hipsAcc).setInterpolation('LINEAR');
      const channel = doc.createAnimationChannel().setTargetNode(hipsNode).setTargetPath('translation').setSampler(sampler);
      anim.addSampler(sampler).addChannel(channel);
    }

    const slideBefore = totalSlide(toeXZ, support);
    console.log(`${anim.getName()}: ${trimmed * TRIM_FRAMES} leading frame(s) trimmed${scaled ? `, amplitude x${amp}` : ''}, grid ${gridLen} frames, ` +
      `foot lock max |C|=${Math.max(...comp.map((c) => Math.hypot(c[0], c[1]))).toFixed(3)}m ` +
      `(uncompensated planted-foot slide ${slideBefore.toFixed(3)}m), lift range ${(Math.max(...lift) - Math.min(...lift)).toFixed(3)}m`);
  }

  await doc.transform(prune());
  await io.write(outPath, doc);
  console.log(`wrote ${outPath}`);
}

function totalSlide(toeXZ, support) {
  let total = 0;
  for (let f = 1; f < support.length; f++) {
    const s = support[f];
    total += Math.hypot(toeXZ[f][s * 2] - toeXZ[f - 1][s * 2], toeXZ[f][s * 2 + 1] - toeXZ[f - 1][s * 2 + 1]);
  }
  return total;
}

main().catch((err) => { console.error(err); process.exit(1); });
