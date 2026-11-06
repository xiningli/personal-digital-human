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

test("fadeFor scales with the pose gap and stays clamped", async () => {
  const { fadeFor } = await import("../player/avatar");
  assert.equal(fadeFor(0), 0.35);                 // nothing to travel: the old fade
  assert.ok(Math.abs(fadeFor(1.08) - 0.9) < 1e-9); // the measured idle<->speaking gap: 0.9 s
  assert.equal(fadeFor(5), 1.2);                  // never longer than FADE_MAX
});

test("matchFrame finds the frame nearest the pose within the window", async () => {
  const { matchFrame, quatAngle } = await import("../player/avatar");
  const ang = (a: number) => [Math.sin(a / 2), 0, 0, Math.cos(a / 2)];   // rotation about x
  const frames = 30;
  const quats = Array.from({ length: frames }, (_, f) => ang(f === 21 ? 0.1 : 1.0)).flat();
  const track: MotionTrack = { fps: 30, frames, bones: ["LeftHand"], quats };
  const pose = new Map([["LeftHand", ang(0.12)]]);
  const hit = matchFrame(track, pose, 0, 1);
  assert.equal(hit.frame, 21);
  assert.ok(Math.abs(hit.gap - quatAngle(ang(0.12), ang(0.1))) < 1e-6);
  assert.equal(matchFrame(track, new Map([["Nope", ang(0)]]), 5).frame, 5);   // no watched bone: stay
});
