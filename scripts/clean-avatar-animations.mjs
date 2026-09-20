#!/usr/bin/env node
/**
 * Clean rotation keyframes in a .glb's animations, in the asset itself, once, instead of
 * papering over bad frames every time three.js plays the clip back (see git history for the
 * runtime patch this replaces, and its measurements). Two passes, via @gltf-transform/core
 * (MIT), not hand-rolled quaternion math wired into the render loop:
 *
 * 1. Sign continuity: q and -q are the same rotation, but SLERP is not sign-invariant, so a
 *    track that crosses that double cover between two keyframes spins the long way around.
 *    Every quaternion output accessor is walked once, flipping a keyframe's sign whenever it
 *    opposes the one before it.
 * 2. Implausible angular velocity: even sign-correct, one keyframe pair can still encode a
 *    rotation no real body part performs in that little time (measured on this asset's own
 *    "Talking" clip: the head bone's second and third keyframes, 0.033 s apart at its native
 *    30 fps, are 2.98 rad -- about 171 degrees -- apart: roughly 5100 degrees/s). This is not
 *    a spike that returns to where it started, so simple outlier-and-restore does not fit it;
 *    the offending keyframe is dropped instead, so its neighbours interpolate straight across
 *    the gap it leaves, and the pass repeats until no segment exceeds the velocity ceiling.
 *
 * Usage: node scripts/clean-avatar-animations.mjs <in.glb> [out.glb]
 *   out.glb defaults to overwriting in.glb.
 */
import { NodeIO } from '@gltf-transform/core';

// ~1150 deg/s: well above a real fast head-shake or gesture, well below what a bad
// retargeted keyframe produced here (measured ~5100 deg/s).
const MAX_RAD_PER_S = 20;

function dot4(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; }
function angleBetween(a, b) { return 2 * Math.acos(Math.min(1, Math.abs(dot4(a, b)))); }

function fixSignContinuity(values, n) {
  let flips = 0;
  for (let f = 1; f < n; f++) {
    const p = (f - 1) * 4, c = f * 4;
    if (dot4(values.subarray(p, p + 4), values.subarray(c, c + 4)) < 0) {
      for (let k = 0; k < 4; k++) values[c + k] = -values[c + k];
      flips++;
    }
  }
  return flips;
}

/** One pass: which keyframes would the clip be smoother without. */
function worstKeyframes(times, values) {
  const drop = new Set();
  for (let f = 1; f < times.length - 1; f++) {
    const dtPrev = times[f] - times[f - 1];
    const dtNext = times[f + 1] - times[f];
    if (dtPrev <= 0 || dtNext <= 0) continue;
    const prev = values.subarray((f - 1) * 4, (f - 1) * 4 + 4);
    const cur = values.subarray(f * 4, f * 4 + 4);
    const next = values.subarray((f + 1) * 4, (f + 1) * 4 + 4);
    const velKept = Math.max(angleBetween(prev, cur) / dtPrev, angleBetween(cur, next) / dtNext);
    if (velKept <= MAX_RAD_PER_S) continue;
    const velRemoved = angleBetween(prev, next) / (dtPrev + dtNext);
    if (velRemoved < velKept) drop.add(f);
  }
  return drop;
}

function removeKeyframes(times, values, drop) {
  const kept = [...Array(times.length).keys()].filter((i) => !drop.has(i));
  const newTimes = new Float32Array(kept.length);
  const newValues = new Float32Array(kept.length * 4);
  kept.forEach((i, j) => { newTimes[j] = times[i]; newValues.set(values.subarray(i * 4, i * 4 + 4), j * 4); });
  return { times: newTimes, values: newValues };
}

/**
 * A keyframe is dropped when the clip is smoother without it than with it: the velocity
 * bridging straight from its predecessor to its successor is lower than the velocity either
 * side of it actually needed, and that need exceeded the ceiling. Direct rather than a
 * spike-and-restore heuristic, so it does not matter whether the bad sample bounces back
 * afterwards or the clip carries on from wherever it landed. Repeats, because a run of
 * several consecutive bad samples (measured on this asset's own "Talking" clip: three
 * keyframes in a row, all far from each other and from the good ones either side) needs its
 * worst point removed, its neighbours re-evaluated, and the next-worst removed in turn.
 */
function dropImplausibleKeyframes(times, values) {
  let t = times, v = values, dropped = 0;
  for (let pass = 0; pass < 10; pass++) {
    const drop = worstKeyframes(t, v);
    if (!drop.size || t.length - drop.size < 3) break;
    ({ times: t, values: v } = removeKeyframes(t, v, drop));
    dropped += drop.size;
  }
  return { times: t, values: v, dropped };
}
async function main() {
  const [inPath, outPath = inPath] = process.argv.slice(2);
  if (!inPath) { console.error('usage: clean-avatar-animations.mjs <in.glb> [out.glb]'); process.exit(1); }

  const io = new NodeIO();
  const doc = await io.read(inPath);
  const animations = doc.getRoot().listAnimations();
  let totalFlips = 0, totalDropped = 0;

  for (const anim of animations) {
    for (const channel of anim.listChannels()) {
      if (channel.getTargetPath() !== 'rotation') continue;
      const sampler = channel.getSampler();
      if (!sampler) continue;
      const input = sampler.getInput();
      const output = sampler.getOutput();
      if (!input || !output) continue;

      const times = input.getArray();
      const values = output.getArray();
      if (values.length !== times.length * 4) continue; // not a quaternion track

      const flips = fixSignContinuity(values, times.length);
      const { times: newTimes, values: newValues, dropped } = dropImplausibleKeyframes(times, values);
      if (dropped) {
        // The input (times) accessor is commonly shared by every bone's channel in a clip
        // (they all key off the same frame numbers); trimming it in place for this one bone
        // would silently truncate every other bone's track too. This bone gets its own copy.
        const trimmedInput = doc.createAccessor(`${input.getName() || 'times'}-trimmed`).setType('SCALAR').setArray(newTimes);
        sampler.setInput(trimmedInput);
        output.setArray(newValues);
      } else if (flips) {
        output.setArray(values);
      }

      totalFlips += flips;
      totalDropped += dropped;
      if (flips || dropped) {
        const node = channel.getTargetNode();
        console.log(`${anim.getName()} / ${node?.getName() ?? '?'}: ${flips} sign flip(s), ${dropped} implausible keyframe(s) dropped (of ${times.length})`);
      }
    }
  }

  await io.write(outPath, doc);
  console.log(`wrote ${outPath}: ${totalFlips} sign flip(s), ${totalDropped} implausible keyframe(s) total across ${animations.length} animation(s)`);
}

main().catch((err) => { console.error(err); process.exit(1); });
