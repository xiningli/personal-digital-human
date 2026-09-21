// Sentence-level gesture assembly for motion profiles. The extraction side
// (motion/segment.py) writes per-sentence gesture segments with embeddings; at runtime the
// arena splits what the digital human is about to say, asks /api/profiles/[id]/match for the
// semantically closest segment per sentence, and splices those frame ranges into one track
// here. Both halves are pure TypeScript so the client can use them directly.

import { Quaternion } from "three";
import type { MotionTrack } from "@/player";

/** One frame range picked out of a profile track, half-open [startFrame, endFrame). */
export interface SegmentPick {
  startFrame: number;
  endFrame: number;
}

/** What the match route returns for one sentence: the winning segment, or null. */
export interface SegmentMatch {
  i: number;
  score: number;
  startS: number;
  endS: number;
  startFrame: number;
  endFrame: number;
  text: string;
}

/** Sentences longer than this get their tail merged, so a rambling answer stays one track. */
const MAX_SENTENCES = 12;

/** Split what the digital human says into sentences, on Chinese and English punctuation. */
export function splitSentences(text: string): string[] {
  // ASCII "." splits only at the end or before a capitalized/quoted next word, so
  // "1.5 cm" and "e.g. in the foot" survive while "Done. Next" splits.
  const parts = text.split(/[\u3002\uff01\uff1f\uff1b!?;\n]+|\.(?=\s+[A-Z0-9"']|$)/).map((s) => s.trim()).filter(Boolean);
  if (parts.length <= MAX_SENTENCES) return parts;
  return [...parts.slice(0, MAX_SENTENCES - 1), parts.slice(MAX_SENTENCES - 1).join("。")];
}

/**
 * Splice picked frame ranges of a track into one continuous track. Where two picks meet the
 * pose would jump, so each seam is a quaternion crossfade over `blendFrames` frames: the two
 * sides overlap and the output slerps from the outgoing pose to the incoming one. A pick no
 * longer than `blendFrames` gets half the blend, or the crossfade would eat the whole gesture.
 */
export function assembleTrack(track: MotionTrack, picks: SegmentPick[], blendFrames = 9): MotionTrack {
  const { fps, frames, bones, quats } = track;
  const B = bones.length;
  const ranges = picks
    .map((p) => ({
      start: Math.max(0, Math.min(Math.min(p.startFrame, p.endFrame), frames - 1)),
      end: Math.max(0, Math.min(Math.max(p.startFrame, p.endFrame), frames)),
    }))
    .filter((r) => r.end > r.start);
  if (!ranges.length) return { fps, frames: 0, bones: bones.slice(), quats: [] };

  const lens = ranges.map((r) => r.end - r.start);
  const blends = ranges.slice(1).map((_, k) => {
    const half = lens[k] <= blendFrames || lens[k + 1] <= blendFrames;
    const b = half ? Math.floor(blendFrames / 2) : blendFrames;
    // Leave at least one unblended frame on the incoming side, or the seam consumes it.
    return Math.max(0, Math.min(b, lens[k], lens[k + 1] - 1));
  });
  const total = lens.reduce((s, n) => s + n, 0) - blends.reduce((s, n) => s + n, 0);
  const out = new Array<number>(total * B * 4);

  const qa = new Quaternion();
  const qb = new Quaternion();
  const qo = new Quaternion();
  const readSrc = (f: number, b: number, q: Quaternion) => {
    const o = (f * B + b) * 4;
    q.set(quats[o], quats[o + 1], quats[o + 2], quats[o + 3]);
  };

  let dst = 0;
  for (let i = 0; i < ranges.length; i++) {
    const { start, end } = ranges[i];
    const blend = i === 0 ? 0 : blends[i - 1];
    if (blend > 0) {
      // The overlap was already copied from the outgoing pick; rewrite it as the crossfade.
      for (let j = 0; j < blend; j++) {
        const t = blend === 1 ? 1 : j / (blend - 1);
        for (let b = 0; b < B; b++) {
          const o = ((dst - blend + j) * B + b) * 4;
          qa.set(out[o], out[o + 1], out[o + 2], out[o + 3]);
          readSrc(start + j, b, qb);
          qo.slerpQuaternions(qa, qb, t);
          out[o] = qo.x; out[o + 1] = qo.y; out[o + 2] = qo.z; out[o + 3] = qo.w;
        }
      }
    }
    for (let f = start + blend; f < end; f++) {
      for (let b = 0; b < B; b++) {
        readSrc(f, b, qa);
        const o = (dst * B + b) * 4;
        out[o] = qa.x; out[o + 1] = qa.y; out[o + 2] = qa.z; out[o + 3] = qa.w;
      }
      dst++;
    }
  }
  return { fps, frames: total, bones: bones.slice(), quats: out };
}
