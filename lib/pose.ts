/**
 * Procedural body language, parameterized by a MotionPolicy (docs/protocol.md §1).
 *
 * This is the rule set the personal site ships (its src/pose.ts, ADR-028), with the
 * amplitudes exposed as policy parameters so the arena can rank variants of it against each
 * other and against motion-capture clips and, later, generated motion. Pure functions of
 * time and speech loudness; the renderer eases towards `pose()` and adds `gestures()`.
 *
 * Bone axes, probed on the asset: upper arms lower about local +X (rest 1.25 rad, 0 is the
 * T-pose); with the arm down the forearm bends forward about local Z (+ right, - left);
 * the head pitches about +X (positive down), turns about +Y, tilts about +Z.
 */
import type { Mood, MotionPolicy } from "./types";

export type Triple = [number, number, number];

export interface Pose {
  Head: Triple;
  Neck: Triple;
  Spine1: Triple;
  Spine2: Triple;
  LeftArm: Triple;
  RightArm: Triple;
  LeftForeArm: Triple;
  RightForeArm: Triple;
}

export const POSE_BONES: (keyof Pose)[] = ["Head", "Neck", "Spine1", "Spine2", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm"];

/**
 * Rest pose for the procedural body, read out of the avatar's own Idle clip (first frame,
 * quaternion converted to XYZ Euler) instead of guessed. The hand-written approximation this
 * replaces lowered both upper arms 1.25 rad about X and nothing else, which left them splayed
 * away from the torso because the natural inward Y rotation was missing, bent the forearms
 * about X where this rig bends them about Z, so they hung straight down, and never touched the
 * wrists, so the hands kept their T-pose splay. That is exactly what the owner saw on
 * 2026-09-19. Hands are set once here and no rule moves them afterwards.
 */
export const REST_POSE: Record<string, Triple> = {
  LeftArm: [1.338, 0.15, -0.003],
  RightArm: [1.215, -0.336, 0.049],
  LeftForeArm: [-0.106, -0.016, 0.323],
  RightForeArm: [-0.103, 0.016, -0.414],
  LeftHand: [0.027, -0.577, -0.091],
  RightHand: [0.171, 0.374, 0.131],
};

/** Put the rig into REST_POSE; used when no clip plays underneath. */
export function applyRestPose(find: (name: string) => { rotation: { set(x: number, y: number, z: number): void } } | undefined): void {
  for (const [name, r] of Object.entries(REST_POSE)) find(name)?.rotation.set(r[0], r[1], r[2]);
}
/** With a clip playing the clip owns the body; these stay procedural on top. */
export const HEAD_BONES: (keyof Pose)[] = ["Head", "Neck"];

export interface Beat { arm: "left" | "right"; at: number }

export const BEAT_S = 0.6;
export const NOD_S = 0.8;
export const BEAT_RISE = 0.08;
export const BEAT_FLOOR = 0.22;
export const BEAT_GAP_S = 0.4;
export const EASE_S: Record<Mood, number> = { idle: 0.45, listening: 0.35, hearing: 0.3, thinking: 0.4, speaking: 0.14, greeting: 0.25 };

const ZERO: Triple = [0, 0, 0];
const add = (a: Triple, b: Triple): Triple => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

export function pulse(age: number, length: number): number {
  if (age < 0 || age > length) return 0;
  return Math.sin((age / length) * Math.PI);
}

/** A loudness rise earns a beat when the policy allows beats at all; `beats` scales the threshold. */
export function wantsBeat(level: number, last: number, sinceBeat: number, policy: MotionPolicy): boolean {
  if (policy.beats <= 0) return false;
  const rise = BEAT_RISE / policy.beats;
  return level > BEAT_FLOOR && level - last > rise && sinceBeat >= BEAT_GAP_S;
}

export function pose(mood: Mood, time: number, level: number, policy: MotionPolicy): Pose {
  const t = time;
  const sway = policy.sway;
  const breath = Math.sin(t * 1.15);
  const out: Pose = {
    Head: [breath * 0.006, 0, 0],
    Neck: [breath * 0.004, 0, 0],
    Spine1: [breath * 0.01, 0, 0],
    Spine2: [breath * 0.008, 0, 0],
    LeftArm: ZERO, RightArm: ZERO, LeftForeArm: ZERO, RightForeArm: ZERO,
  };
  switch (mood) {
    case "idle":
      out.Head = add(out.Head, [Math.sin(t * 0.17) * 0.03 * sway, Math.sin(t * 0.23) * 0.07 * sway, Math.sin(t * 0.11 + 1) * 0.02 * sway]);
      out.Spine2 = add(out.Spine2, [0, Math.sin(t * 0.09) * 0.03 * sway, 0]);
      break;
    case "listening":
      out.Head = add(out.Head, [0.05 + Math.sin(t * 0.29) * 0.015, Math.sin(t * 0.19) * 0.03 * sway, 0.07 + Math.sin(t * 0.13) * 0.01]);
      out.Neck = add(out.Neck, [0.03, 0, 0.02]);
      out.Spine2 = add(out.Spine2, [0.04, 0, 0]);
      break;
    case "hearing": {
      const nods = (0.5 - 0.5 * Math.cos((t / 1.7) * Math.PI * 2)) * 0.055;
      out.Head = add(out.Head, [0.07 + nods, Math.sin(t * 0.19) * 0.02, 0.05]);
      out.Neck = add(out.Neck, [0.04 + nods * 0.5, 0, 0.015]);
      out.Spine2 = add(out.Spine2, [0.06, 0, 0]);
      break;
    }
    case "thinking":
      out.Head = add(out.Head, [-0.08 + Math.sin(t * 0.7) * 0.01, 0.16, -0.04]);
      out.Neck = add(out.Neck, [-0.03, 0.05, 0]);
      out.Spine2 = add(out.Spine2, [0.02, 0.03, 0]);
      break;
    case "speaking": {
      const lift = Math.min(1, level * 1.6) * policy.headLift;
      out.Head = add(out.Head, [-0.1 * lift + Math.sin(t * 2.6) * 0.02 * sway, Math.sin(t * 0.55) * 0.09 * sway, Math.sin(t * 0.8) * 0.03 * sway + lift * 0.02]);
      out.Neck = add(out.Neck, [-0.03 * lift, Math.sin(t * 0.55) * 0.03 * sway, 0]);
      out.Spine1 = add(out.Spine1, [0, 0, Math.sin(t * 0.45) * 0.02 * sway]);
      out.Spine2 = add(out.Spine2, [-0.02 * lift, Math.sin(t * 0.35) * 0.03 * sway, Math.sin(t * 0.45) * 0.015 * sway]);
      if (policy.clip === null) {
        // The procedural body alone: hands in front, elbows bent forward, a slow drift.
        out.LeftArm = [-0.12, 0, 0];
        out.RightArm = [-0.12, 0, 0];
        out.LeftForeArm = [0.05, 0, -(0.75 + Math.sin(t * 1.3) * 0.06)];
        out.RightForeArm = [0.05, 0, 0.75 + Math.sin(t * 1.3 + 1.1) * 0.06];
      }
      break;
    }
    case "greeting": {
      const wave = Math.sin(t * 6.5);
      out.RightArm = [-1.0, 0, 0];
      out.RightForeArm = [-1.52 + wave * 0.3, 0, 0];
      out.Head = add(out.Head, [-0.03, -0.06, 0.06]);
      out.Spine2 = add(out.Spine2, [0, -0.04, 0]);
      break;
    }
  }
  return out;
}

export function gestures(time: number, nodAge: number, beats: Beat[], policy: MotionPolicy): Partial<Pose> {
  const out: Partial<Pose> = {};
  const nod = pulse(nodAge, NOD_S);
  if (nod > 0) {
    out.Head = [0.26 * nod, 0, 0];
    out.Neck = [0.08 * nod, 0, 0];
  }
  const amp = Math.min(2, policy.beats);
  for (const beat of beats) {
    const e = pulse(time - beat.at, BEAT_S) * amp;
    if (e <= 0) continue;
    if (beat.arm === "left") {
      out.LeftArm = add(out.LeftArm ?? ZERO, [-0.12 * e, 0, 0]);
      out.LeftForeArm = add(out.LeftForeArm ?? ZERO, [0, 0, -0.6 * e]);
    } else {
      out.RightArm = add(out.RightArm ?? ZERO, [-0.12 * e, 0, 0]);
      out.RightForeArm = add(out.RightForeArm ?? ZERO, [0, 0, 0.6 * e]);
    }
  }
  return out;
}

function unit(n: number): number {
  const s = Math.sin(n * 127.1) * 43758.5453;
  return s - Math.floor(s);
}

export function blink(time: number, interval = 4.3, duration = 0.13): number {
  const slot = Math.floor(time / interval);
  const t = (time - (slot + unit(slot) * 0.55) * interval) / duration;
  return t < 0 || t > 1 ? 0 : Math.sin(t * Math.PI);
}

/** One loudness envelope drives the ARKit mouth shapes (the site's viseme.ts, ADR-018). */
export function speechFace(level: number, time: number): Record<string, number> {
  const open = Math.min(1, Math.max(0, level) * 2.4) * 0.55;
  const closed = blink(time);
  const vowel = Math.sin(time * 6.1) * 0.5 + 0.5;
  const spread = Math.sin(time * 2.7 + 1.3) * 0.5 + 0.5;
  return {
    jawOpen: open,
    jawForward: open * 0.18,
    mouthClose: open * 0.3,
    mouthFunnel: open * vowel * 0.75,
    mouthPucker: open * (1 - vowel) * 0.5,
    mouthLowerDownLeft: open * 0.65,
    mouthLowerDownRight: open * 0.65,
    mouthSmileLeft: 0.12 + open * spread * 0.35,
    mouthSmileRight: 0.12 + open * spread * 0.35,
    cheekSquintLeft: open * 0.12,
    cheekSquintRight: open * 0.12,
    browInnerUp: 0.05 + Math.sin(time * 0.43) * 0.04 + open * 0.12,
    eyeBlinkLeft: closed,
    eyeBlinkRight: closed,
  };
}
