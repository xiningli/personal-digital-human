import { NextRequest } from "next/server";
import { getArenaRounds, getArenaVotes } from "@/lib/storage";
import { fitBradleyTerry, raterReliability } from "@/lib/stats";

export const dynamic = "force-dynamic";

/** Bradley-Terry table per voice checkpoint (?voiceModel=<id> | all; default the newest round's). */
export async function GET(request: NextRequest) {
  const [rounds, votes] = await Promise.all([getArenaRounds(), getArenaVotes()]);
  const voiceModels = [...new Set(rounds.map((r) => r.voiceModel))];
  const newest = [...rounds].sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0];
  const requested = request.nextUrl.searchParams.get("voiceModel");
  const voiceModel = requested || newest?.voiceModel || "all";
  const raw = Number(request.nextUrl.searchParams.get("bootstrap"));
  const bootstrap = Number.isFinite(raw) && raw >= 0 && request.nextUrl.searchParams.has("bootstrap") ? Math.min(5000, raw) : 1000;
  const inScope = rounds.filter((r) => voiceModel === "all" || r.voiceModel === voiceModel);
  const ids = new Set(inScope.map((r) => r.id));
  const trials = { test: 0, repeat: 0 };
  for (const r of inScope) trials[r.trial] += 1;
  return Response.json({
    voiceModel, voiceModels, totalRounds: inScope.length, totalVotes: votes.filter((v) => ids.has(v.roundId)).length, trials, bootstrap,
    bt: fitBradleyTerry(rounds, votes, { voiceModel, bootstrap }),
    reliability: raterReliability(rounds, votes, voiceModel),
  });
}
