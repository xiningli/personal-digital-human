import assert from "node:assert/strict";
import { test } from "node:test";
import { trackToClip, type MotionTrack } from "../player/avatar";

const base: MotionTrack = { fps: 30, frames: 3, bones: ["Hips", "Spine"], quats: Array(3 * 2 * 4).fill(0).map((_, i) => (i % 4 === 3 ? 1 : 0)) };

test("a grounded track plays its pelvis translation", () => {
  const hips = [0, 0.92, 0, 0.01, 0.92, 0, 0.02, 0.91, 0];
  const clip = trackToClip({ ...base, hips });
  const pos = clip.tracks.find((t) => t.name === "Hips.position");
  assert.ok(pos, "Hips.position track present");
  assert.deepEqual(Array.from(pos.values).map((v) => Math.round(v * 100) / 100), hips);
  assert.equal(clip.tracks.filter((t) => t.name.endsWith(".quaternion")).length, 2);
});

test("an ungrounded or malformed hips array adds no translation", () => {
  assert.equal(trackToClip(base).tracks.some((t) => t.name === "Hips.position"), false);
  assert.equal(trackToClip({ ...base, hips: [0, 1, 0] }).tracks.some((t) => t.name === "Hips.position"), false);
});

test("assembleTrack splices and blends the pelvis with the rotations", async () => {
  const { assembleTrack } = await import("../lib/motion-segments");
  const frames = 20;
  const quats = Array(frames * 2 * 4).fill(0).map((_, i) => (i % 4 === 3 ? 1 : 0));
  const hips = Array.from({ length: frames * 3 }, (_, i) => (i % 3 === 0 ? Math.floor(i / 3) / 100 : 0.92));
  const out = assembleTrack({ ...base, frames, quats, hips }, [{ startFrame: 0, endFrame: 10 }, { startFrame: 15, endFrame: 20 }], 3);
  assert.equal(out.hips?.length, out.frames * 3);
  assert.equal(out.hips?.[0], 0);
  assert.equal(out.hips?.[out.frames * 3 - 3], 0.19);
  for (let f = 1; f < out.frames; f++) assert.ok(Math.abs(out.hips![f * 3] - out.hips![(f - 1) * 3]) <= 0.07, `pelvis jumps at ${f}`);
  assert.equal(assembleTrack({ ...base, frames, quats }, [{ startFrame: 0, endFrame: 5 }]).hips, undefined);
});
