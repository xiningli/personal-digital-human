import test from "node:test";
import assert from "node:assert/strict";
import { Quaternion, Vector3 } from "three";
import { assembleTrack, splitSentences } from "../lib/motion-segments.ts";
import type { MotionTrack } from "../player/index.ts";

test("splitSentences splits on Chinese and English punctuation and drops blanks", () => {
  assert.deepEqual(splitSentences("你好。欢迎来到这个地方!有什么问题吗？  "), ["你好", "欢迎来到这个地方", "有什么问题吗"]);
  // English full stops split too, so multi-sentence answers get per-sentence gestures…
  assert.deepEqual(splitSentences("Hello there! How are you? Great; thanks."), ["Hello there", "How are you", "Great", "thanks"]);
  // …but a stop only splits before whitespace or the end: decimals and abbreviations survive.
  assert.deepEqual(splitSentences("The error is 1.5 cm, e.g. in the foot. Next point."), ["The error is 1.5 cm, e.g. in the foot", "Next point"]);
  assert.deepEqual(splitSentences("第一句\n第二句;第三句"), ["第一句", "第二句", "第三句"]);
  assert.deepEqual(splitSentences("   "), []);
});

test("splitSentences merges the tail beyond twelve sentences", () => {
  const text = Array.from({ length: 15 }, (_, i) => `第${i + 1}句`).join("。") + "。";
  const parts = splitSentences(text);
  assert.equal(parts.length, 12);
  assert.equal(parts[0], "第1句");
  assert.ok(parts[11].includes("第12句") && parts[11].includes("第15句"));
});

/** A track whose bones rotate steadily, one axis per segment, so seam jumps are measurable. */
function makeTrack(): MotionTrack {
  const bones = ["Hips", "Spine"];
  const frames = 60;
  const quats: number[] = [];
  const q = new Quaternion();
  const xAxis = new Vector3(1, 0, 0);
  const yAxis = new Vector3(0, 1, 0);
  for (let f = 0; f < frames; f++) {
    // Frames 0..29 rotate about x, 30..59 about y: the two ranges end in unrelated poses.
    if (f < 30) q.setFromAxisAngle(xAxis, f * 0.02);
    else q.setFromAxisAngle(yAxis, (f - 30) * 0.02);
    for (let b = 0; b < bones.length; b++) quats.push(q.x, q.y, q.z, q.w);
  }
  return { fps: 30, frames, bones, quats };
}

const qa = new Quaternion();
const qb = new Quaternion();
function quatAt(track: MotionTrack, f: number, b: number, q: Quaternion) {
  const o = (f * track.bones.length + b) * 4;
  q.set(track.quats[o], track.quats[o + 1], track.quats[o + 2], track.quats[o + 3]);
}

test("assembleTrack preserves frame count and bone order", () => {
  const track = makeTrack();
  const out = assembleTrack(track, [{ startFrame: 0, endFrame: 30 }, { startFrame: 30, endFrame: 60 }], 9);
  assert.equal(out.frames, 30 + 30 - 9);
  assert.equal(out.quats.length, out.frames * track.bones.length * 4);
  assert.deepEqual(out.bones, track.bones);
  assert.equal(out.fps, track.fps);
});

test("assembleTrack blends seams instead of jumping", () => {
  const track = makeTrack();
  // Both segments rotate by 0.02 rad per frame; unblended the seam would jump by the ~0.58 rad
  // between the two boundary poses, blended every step stays near the per-frame rate.
  const out = assembleTrack(track, [{ startFrame: 0, endFrame: 30 }, { startFrame: 30, endFrame: 60 }], 9);
  let maxStep = 0;
  for (let b = 0; b < out.bones.length; b++) {
    for (let f = 1; f < out.frames; f++) {
      quatAt(out, f - 1, b, qa);
      quatAt(out, f, b, qb);
      maxStep = Math.max(maxStep, qa.angleTo(qb));
    }
  }
  assert.ok(maxStep < 0.15, `seam step ${maxStep} rad should stay near the per-frame rate`);
});

test("assembleTrack halves the blend for picks shorter than the blend window", () => {
  const track = makeTrack();
  const out = assembleTrack(track, [{ startFrame: 0, endFrame: 30 }, { startFrame: 30, endFrame: 36 }], 9);
  assert.equal(out.frames, 30 + 6 - 4);
});

test("assembleTrack tolerates empty and out-of-range picks", () => {
  const track = makeTrack();
  assert.deepEqual(assembleTrack(track, []), { fps: 30, frames: 0, bones: track.bones, quats: [] });
  const out = assembleTrack(track, [{ startFrame: -5, endFrame: 10 }, { startFrame: 55, endFrame: 999 }], 4);
  assert.equal(out.frames, 10 + 5 - 4);
});
