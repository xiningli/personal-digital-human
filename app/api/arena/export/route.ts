import { NextRequest } from "next/server";
import { getArenaRounds, getArenaVotes } from "@/lib/storage";
import type { PreferencePair } from "@/lib/types";

export const dynamic = "force-dynamic";

/** preference-pairs.jsonl: one line per decided two-candidate round (?format=json for an array). */
export async function GET(request: NextRequest) {
  const [rounds, votes] = await Promise.all([getArenaRounds(), getArenaVotes()]);
  const byId = new Map(rounds.map((r) => [r.id, r]));
  const pairs: PreferencePair[] = [];
  for (const v of votes) {
    const r = byId.get(v.roundId);
    if (!r || !v.winnerId || r.candidates.length !== 2) continue;
    const chosen = r.candidates.find((c) => c.id === v.winnerId);
    const rejected = r.candidates.find((c) => c.id !== v.winnerId);
    if (!chosen || !rejected) continue;
    const side = (c: typeof chosen) => ({ candidateId: c.id, label: c.label, policy: c.policy, note: v.candidateNotes[c.id] });
    pairs.push({ roundId: r.id, createdAt: v.createdAt, trial: r.trial, category: r.category, text: r.text, audioPath: r.audioPath, emotion: r.emotion,
      voiceModel: r.voiceModel, avatar: r.avatar, seed: r.seed, chosen: side(chosen), rejected: side(rejected), ratings: v.ratings, notes: v.notes });
  }
  if (request.nextUrl.searchParams.get("format") === "json") return Response.json(pairs);
  return new Response(pairs.map((p) => JSON.stringify(p)).join("\n") + (pairs.length ? "\n" : ""), {
    headers: { "Content-Type": "application/x-ndjson", "Content-Disposition": 'attachment; filename="preference-pairs.jsonl"' },
  });
}
