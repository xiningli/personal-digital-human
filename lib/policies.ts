// The candidate space of the motion arena (docs/protocol.md §1) and the line bank.
// Policies are a small discrete design over the two things the site can do today: play a
// Mixamo clip underneath, and run the procedural rules on top at some amplitude. A learned
// policy (source "model") joins the same table when a co-speech model produces tracks.

import type { MotionPolicy } from "./types";

// "Telling A Secret" was dropped on 2026-09-19: it is a conspiratorial hunch, weight on one
// foot and a hand cupped at the mouth, which reads as the body floating and is wrong for a
// digital human answering a question. "Lengthy Head Nod" gestures with its hands while
// nodding, so it stays as the second thing a speaking body can do; the arena ranks the two.
export const SPEAKING_CLIPS = ["Talking", "Lengthy Head Nod"] as const;
/** The clips the served asset carries; the renderer reports which it actually found. */
export const KNOWN_CLIPS = ["Breathing Idle", "Head Nod Yes", "Idle", "Lengthy Head Nod", "Shrugging", "Talking", "Telling A Secret", "Thoughtful Head Nod", "Waving"] as const;

const LEVELS = { headLift: [0, 1, 1.6], beats: [0, 1, 1.8], sway: [0.5, 1, 1.6], ease: [1] } as const;

/** The site's own default: Talking underneath, the rules at their shipped amplitude. */
export const SITE_DEFAULT: MotionPolicy = { id: "clip:Talking/h1-b1-s1", name: "Talking · site default", source: "clip", clip: "Talking", headLift: 1, beats: 1, sway: 1, ease: 1 };

function name(clip: string | null, h: number, b: number, s: number): string {
  const body = clip ?? "procedural body";
  const parts = [`head ${h}`, `beats ${b}`, `sway ${s}`];
  return `${body} · ${parts.join(", ")}`;
}

/**
 * Every policy the sampler can draw: the speaking clips across the amplitude grid, plus a
 * single procedural-body baseline.
 *
 * Until 2026-09-19 the procedural body took the whole grid too, so 27 of 47 policies were the
 * hand-written body and most rounds pitted two of its variants against each other. It is the
 * weakest candidate and only ships as the fallback for an avatar with no clips, so exactly one
 * of it remains, to answer "is a clip better than the fallback" and nothing more.
 */
export function designSpace(): MotionPolicy[] {
  const out: MotionPolicy[] = [];
  for (const clip of SPEAKING_CLIPS) {
    for (const h of LEVELS.headLift) for (const b of LEVELS.beats) for (const s of LEVELS.sway) {
      // With a clip underneath, beats move arms the clip already moves; keep beats off there
      // except at the site default, so the grid stays small and the arms are not doubled.
      if (b > 0 && !(h === 1 && b === 1 && s === 1)) continue;
      const key = `clip:${clip}/h${h}-b${b}-s${s}`;
      out.push({ id: key, name: name(clip, h, b, s), source: "clip", clip, headLift: h, beats: b, sway: s, ease: 1 });
    }
  }
  out.push({ id: "proc/h1-b1-s1", name: name(null, 1, 1, 1) + " (fallback baseline)", source: "procedural", clip: null, headLift: 1, beats: 1, sway: 1, ease: 1 });
  return out;
}

export function policyById(id: string): MotionPolicy | undefined {
  return designSpace().find((p) => p.id === id);
}

/**
 * Two distinct policies, drawn with inverse-frequency weighting over how often each has
 * already been served, so the table fills evenly instead of by chance.
 */
export function samplePair(served: Map<string, number>, rng: () => number = Math.random): [MotionPolicy, MotionPolicy] {
  const pool = designSpace();
  const weight = (p: MotionPolicy) => 1 / (1 + (served.get(p.id) ?? 0));
  const draw = (exclude?: MotionPolicy): MotionPolicy => {
    const items = pool.filter((p) => p.id !== exclude?.id);
    const total = items.reduce((s, p) => s + weight(p), 0);
    let r = rng() * total;
    for (const p of items) { r -= weight(p); if (r <= 0) return p; }
    return items[items.length - 1];
  };
  const a = draw();
  return [a, draw(a)];
}

export interface LineCategory { id: string; label: string; lines: string[] }

/** Lines the digital human actually says: the same kinds of sentence the voice arena uses. */
export const LINE_BANK: LineCategory[] = [
  { id: "greeting", label: "Greeting", lines: [
    "Hi, welcome in. I'm glad you found your way here. Take a look around, and ask me anything that catches your eye.",
    "Hey, good to see you. Come in, make yourself comfortable.",
    "Welcome. This is the space I built. Feel free to wander, and I'll be right here if you have questions.",
    "Oh, hello! I wasn't expecting anyone this early. Come on in.",
  ] },
  { id: "answer", label: "Answering a question", lines: [
    "So you're asking which fix I'd try first. Honestly, I'd check the learning rate before anything else, because that's where it usually goes wrong.",
    "That flat accuracy while the loss keeps dropping usually means the model is getting more confident about the wrong answers.",
    "Doubling the learning rate mostly makes training bounce around, and sometimes it just blows up. I'd halve it and watch the curve.",
    "Good question. Short answer, yes, but it depends on the data, and I've been burned by that before.",
  ] },
  { id: "explaining", label: "Explaining", lines: [
    "Attention lets every token look at every other token at the same time. That is the whole trick, and everything else is bookkeeping.",
    "There are three parts to this: the intuition, the math, and one worked example. Let's start with the intuition.",
    "Think of the mirror as a sketch that decides where the world breaks. Your lines become the cracks, and the cracks become the doorway.",
  ] },
  { id: "thinking", label: "Thinking aloud", lines: [
    "Hmm, that's a good one. Let me think about it for a second. I believe the short answer is yes, but it depends on the data.",
    "Okay, so there are two ways to look at this, and I'm not sure which one you're asking about.",
    "Right, so, if I remember correctly, the paper reported that, but I'd want to double check the setup.",
  ] },
  { id: "delight", label: "Delighted", lines: [
    "Oh, this is one of my favourite topics! There's something so satisfying about watching all these pieces click into place.",
    "You drew that? That's wonderful. Look at how the world grew out of three lines.",
    "Yes! That is exactly the question I was hoping you'd ask.",
  ] },
];

export function pickLine(category: string, rng: () => number = Math.random): { category: string; text: string } {
  const pool = category === "any" ? LINE_BANK : LINE_BANK.filter((c) => c.id === category);
  const cat = pool[Math.floor(rng() * pool.length)] ?? LINE_BANK[0];
  return { category: cat.id, text: cat.lines[Math.floor(rng() * cat.lines.length)] };
}
