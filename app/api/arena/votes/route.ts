import { NextRequest } from "next/server";
import { v4 as uuid } from "uuid";
import { appendArenaVote, getArenaRound, getArenaVotes } from "@/lib/storage";
import type { ArenaVote, CandidateRating } from "@/lib/types";

const valid = (v: unknown): v is number => Number.isInteger(v) && (v as number) >= 1 && (v as number) <= 5;

export async function GET() { return Response.json(await getArenaVotes()); }

/** Body: { roundId, winnerId: string | null, ratings?: { [candidateId]: {naturalness, fit, likeness} }, notes?, candidateNotes? } */
export async function POST(request: NextRequest) {
  const body = (await request.json().catch(() => ({}))) as {
    roundId?: string; winnerId?: string | null; ratings?: Record<string, Partial<CandidateRating>>; notes?: string; candidateNotes?: Record<string, string>;
  };
  if (!body.roundId) return Response.json({ error: "roundId is required" }, { status: 400 });
  const round = await getArenaRound(body.roundId);
  if (!round) return Response.json({ error: "Round not found" }, { status: 404 });
  const ids = new Set(round.candidates.map((c) => c.id));
  const winnerId = body.winnerId ?? null;
  if (winnerId !== null && !ids.has(winnerId)) return Response.json({ error: "winnerId is not a candidate of this round" }, { status: 400 });
  const ratings: Record<string, CandidateRating> = {};
  for (const [id, r] of Object.entries(body.ratings ?? {})) {
    if (!ids.has(id)) return Response.json({ error: `Unknown candidate in ratings: ${id}` }, { status: 400 });
    if (!r || !valid(r.naturalness) || !valid(r.fit) || !valid(r.likeness)) return Response.json({ error: "Each rating needs naturalness, fit and likeness as integers 1-5" }, { status: 400 });
    ratings[id] = { naturalness: r.naturalness, fit: r.fit, likeness: r.likeness };
  }
  const candidateNotes: Record<string, string> = {};
  for (const [id, note] of Object.entries(body.candidateNotes ?? {})) {
    if (!ids.has(id)) return Response.json({ error: `Unknown candidate in candidateNotes: ${id}` }, { status: 400 });
    if (typeof note === "string" && note.trim()) candidateNotes[id] = note.trim();
  }
  const vote: ArenaVote = { id: uuid(), roundId: round.id, createdAt: new Date().toISOString(), winnerId, ratings, notes: (body.notes ?? "").trim(), candidateNotes };
  await appendArenaVote(vote);
  return Response.json(vote, { status: 201 });
}
