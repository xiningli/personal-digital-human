/**
 * Contract for Personal Digital Human: the motion arena and its preference data.
 * Everything the routes accept or return is typed here (docs/protocol.md).
 */

/** What the body is doing, the same vocabulary the site's dialogue loop uses. */
export type Mood = "idle" | "listening" | "hearing" | "thinking" | "speaking" | "greeting";

/**
 * A motion policy: how the body moves for a spoken line. `clip` is the Mixamo clip that
 * plays underneath (or none: the procedural body alone); the numbers scale the procedural
 * head/torso/arm rules on top. `source` records where the policy came from so learned
 * policies (a co-speech model's output) can be ranked against these in the same table.
 */
export interface MotionPolicy {
  /** Stable id; the ranking key. */
  id: string;
  name: string;
  source: "clip" | "procedural" | "model";
  /** Mixamo clip name for the body, or null for the procedural body. */
  clip: string | null;
  /** Speech loudness -> head lift, 0..2 (1 = the site's default). */
  headLift: number;
  /** Beat gestures per loudness rise, 0..2 (0 = never). */
  beats: number;
  /** Slow head turn and torso sway, 0..2. */
  sway: number;
  /** Time constant scale for easing, 0.5..2 (higher = slower, calmer). */
  ease: number;
  /** For source "model": the id of the model or checkpoint that produced the frames. */
  model?: string;
}

/** Joint rotations per frame for a model-generated candidate (radians, XYZ Euler, per bone). */
export interface MotionTrack {
  fps: number;
  bones: string[];
  /** frames[f][b] = [x, y, z] */
  frames: number[][][];
}

export interface ArenaCandidate {
  id: string;
  /** A/B label under which the rater saw it in this round. */
  label: string;
  policyId: string;
  policy: MotionPolicy;
  /** A recorded track for model candidates; absent for clip/procedural ones (computed live). */
  track?: MotionTrack;
  note?: string;
}

export type TrialType = "test" | "repeat";

export interface ArenaRound {
  id: string;
  createdAt: string;
  trial: TrialType;
  /** The round this one re-serves (trial "repeat"). */
  repeatOf?: string;
  /** Line bank category. */
  category: string;
  text: string;
  /** Public path of the synthesized speech, the same for every candidate. */
  audioPath: string;
  /** Emotion the voice studio delivered the line with. */
  emotion: string;
  /** Voice checkpoint that spoke the line; a partition key like the studio's `model`. */
  voiceModel: string;
  /** Avatar asset the candidates were rendered on. */
  avatar: string;
  candidates: ArenaCandidate[];
  /** Seed of the procedural noise so a repeat renders the same motion. */
  seed: number;
}

export interface CandidateRating {
  /** Does the body look natural for a person saying this? 1-5. */
  naturalness: number;
  /** Does the motion fit the words and their timing? 1-5. */
  fit: number;
  /** Does it move the way the owner moves? 1-5. */
  likeness: number;
}

export interface ArenaVote {
  id: string;
  roundId: string;
  createdAt: string;
  winnerId: string | null;
  ratings: Record<string, CandidateRating>;
  notes: string;
  candidateNotes: Record<string, string>;
}

export interface BradleyTerryEntry {
  policyId: string;
  name: string;
  source: MotionPolicy["source"];
  score: number;
  ci95: [number, number];
  wins: number;
  losses: number;
  rounds: number;
  avgNaturalness: number | null;
  avgFit: number | null;
  avgLikeness: number | null;
}

export interface RaterReliability {
  repeatTrials: number;
  repeatAgreement: number | null;
}

export interface PreferencePair {
  roundId: string;
  createdAt: string;
  trial: TrialType;
  category: string;
  text: string;
  audioPath: string;
  emotion: string;
  voiceModel: string;
  avatar: string;
  seed: number;
  chosen: { candidateId: string; label: string; policy: MotionPolicy; note?: string };
  rejected: { candidateId: string; label: string; policy: MotionPolicy; note?: string };
  ratings: Record<string, CandidateRating>;
  notes: string;
}
