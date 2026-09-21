import { NextRequest } from "next/server";
import fs from "fs/promises";
import path from "path";
import { v4 as uuid } from "uuid";
import { getArenaRounds, getArenaVotes, saveArenaRound } from "@/lib/storage";
import { LINE_BANK, designSpace, pickLine, profileTrackPath, samplePair, trackPath } from "@/lib/policies";
import { profilePolicies } from "@/lib/profiles";
import { EMOTION_FOR_CATEGORY, speak } from "@/lib/tts";
import type { ArenaCandidate, ArenaRound, TrialType } from "@/lib/types";

export const maxDuration = 300;
export const AVATAR = "/assets/model-clips.glb";
/** Trial mix (docs/protocol.md §2): repeats need a decided round to re-serve. */
export const REPEAT_SHARE = 0.15;

/**
 * Sample a round (docs/protocol.md §2): one line, spoken once by the voice studio, and two
 * motion policies drawn by inverse frequency. Body: { category?: "any" | <bank id>, trial?,
 * repeatOf?: string }. A repeat re-serves a decided round with fresh labels and the same
 * audio, seed and policies, so the second vote measures the rater, not the candidates.
 */
export async function POST(request: NextRequest) {
  const body = (await request.json().catch(() => ({}))) as { category?: string; trial?: TrialType; repeatOf?: string };
  const [rounds, votes, profiles] = await Promise.all([getArenaRounds(), getArenaVotes(), profilePolicies()]);
  const pool = [...designSpace(), ...profiles];
  const decided = new Set(votes.map((v) => v.roundId));
  // A repeat measures the rater, so it must re-serve candidates that are still on the table.
  // Rounds built from policies since withdrawn from the design space are skipped, or a retired
  // candidate keeps coming back and the reliability number is about something we no longer ship.
  const live = new Set(pool.map((p) => p.id));
  const repeatable = rounds.filter(
    (r) => r.trial === "test" && decided.has(r.id) && !rounds.some((x) => x.repeatOf === r.id)
      && r.candidates.every((c) => live.has(c.policyId)),
  );
  let trial: TrialType = body.trial ?? (Math.random() < REPEAT_SHARE && repeatable.length ? "repeat" : "test");
  let source: ArenaRound | undefined;
  if (body.repeatOf) { source = rounds.find((r) => r.id === body.repeatOf); trial = "repeat"; }
  else if (trial === "repeat") source = repeatable[Math.floor(Math.random() * repeatable.length)];
  if (trial === "repeat" && !source) trial = "test";

  const labels = ["A", "B"];
  const shuffled = Math.random() < 0.5;
  let round: ArenaRound;
  if (trial === "repeat" && source) {
    const cands = shuffled ? [...source.candidates].reverse() : [...source.candidates];
    round = {
      ...source,
      id: uuid(),
      createdAt: new Date().toISOString(),
      trial: "repeat",
      repeatOf: source.id,
      candidates: cands.map((c, i) => ({ ...c, id: uuid(), label: labels[i], note: undefined })),
    };
  } else {
    const category = body.category && body.category !== "any" && LINE_BANK.some((c) => c.id === body.category) ? body.category : "any";
    const line = pickLine(category);
    const emotion = EMOTION_FOR_CATEGORY[line.category] ?? "neutral";
    let spoken;
    try { spoken = await speak(line.text, emotion); } catch (e) { return Response.json({ error: e instanceof Error ? e.message : String(e) }, { status: 503 }); }
    // Generated motion exists only for audio it was generated from, so a candidate whose
    // track is missing is withheld rather than shown as a failure. A profile's track is one
    // fixed file — the motion is decoupled from its source audio on purpose (§5) — so it is
    // usable in any round as long as the file is there.
    const hasTrack = async (p: { source: string; model?: string; profileId?: string }) => {
      const rel = p.source === "generated" ? trackPath(spoken.audioPath, p.model ?? "")
        : p.source === "profile" ? profileTrackPath(p.profileId ?? "")
        : null;
      if (!rel) return true;
      try {
        await fs.access(path.join(process.cwd(), "public", rel.replace(/^\//, "")));
        return true;
      } catch { return false; }
    };
    const served = new Map<string, number>();
    for (const r of rounds) if (r.trial === "test") for (const c of r.candidates) served.set(c.policyId, (served.get(c.policyId) ?? 0) + 1);
    const usable = new Set<string>();
    for (const p of pool) if (await hasTrack(p)) usable.add(p.id);
    const pair = samplePair(served, undefined, (p) => usable.has(p.id), pool);
    const ordered = shuffled ? [pair[1], pair[0]] : pair;
    const candidates: ArenaCandidate[] = ordered.map((p, i) => ({ id: uuid(), label: labels[i], policyId: p.id, policy: p }));
    round = {
      id: uuid(), createdAt: new Date().toISOString(), trial: "test", category: line.category, text: line.text,
      audioPath: spoken.audioPath, emotion: spoken.emotion, voiceModel: spoken.voiceModel, avatar: AVATAR,
      candidates, seed: Math.floor(Math.random() * 1e9),
    };
  }
  await saveArenaRound(round);
  return Response.json(round, { status: 201 });
}
