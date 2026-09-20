/**
 * What is left after the hand-written body language was deleted (2026-09-19).
 *
 * The body used to be posed by rules in `lib/pose.ts`: sine waves on the spine, a head that
 * lifted with loudness, "beat" gestures fired by a loudness threshold, and amplitude knobs
 * (headLift / beats / sway / ease) that the arena ranked against each other. The owner judged
 * the result stiff and unconvincing and asked for motion that comes from a model instead of
 * from a guess. None of those rules survive.
 *
 * Two things here are not style and are needed whatever produces the motion:
 *
 * - `groundOffset`, because the clips were retargeted with the hips' translation locked at
 *   their bind value. Real motion capture lowers the hips as the legs bend; pinned, any leg
 *   flexion lifts both feet off the floor, and an asymmetric stance lifts one more than the
 *   other. Measured on the Talking clip: soles 3.7 cm and 4.9 cm above the floor, which is the
 *   floating foot the owner saw. Generated motion needs the same treatment after retargeting.
 * - `speechFace`, the mouth. It is a loudness envelope driving ARKit blendshapes, which is
 *   lip-sync rather than body movement, and replacing it needs a viseme model, not a gesture
 *   model. It stays, and it is the one hand-written thing left on this page.
 */

/** How far the sole sits below the toe bone, from the asset's bind pose: toe bones at world
 *  Y 0.0524, lowest mesh vertex at 0. */
export const SOLE_BELOW_TOE = 0.0524;

/**
 * Vertical offset that puts the lower foot on the floor. Grounding to the LOWER toe keeps the
 * planted foot down and still lets the other one leave the ground.
 */
export function groundOffset(toeWorldY: number[], soleBelowToe = SOLE_BELOW_TOE): number {
  const usable = toeWorldY.filter((y) => Number.isFinite(y));
  if (!usable.length) return 0;
  return -(Math.min(...usable) - soleBelowToe);
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

/** Lip-sync: one loudness envelope driving the asset's ARKit mouth shapes. */
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
