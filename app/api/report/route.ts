import { getArenaRounds, getArenaVotes } from "@/lib/storage";
import { fitBradleyTerry, raterReliability } from "@/lib/stats";

export const dynamic = "force-dynamic";

/** A Markdown report of the ranking per voice checkpoint, for the owner's notes and ADRs. */
export async function GET() {
  const [rounds, votes] = await Promise.all([getArenaRounds(), getArenaVotes()]);
  const models = [...new Set(rounds.map((r) => r.voiceModel))];
  const lines = [`# Motion arena report`, ``, `Generated ${new Date().toISOString()} · ${rounds.length} rounds · ${votes.length} votes`, ``];
  for (const m of models) {
    const bt = fitBradleyTerry(rounds, votes, { voiceModel: m });
    const rel = raterReliability(rounds, votes, m);
    lines.push(`## Voice: ${m}`, ``, `Repeat agreement: ${rel.repeatAgreement ?? "n/a"} over ${rel.repeatTrials} repeat trials`, ``,
      `| policy | BT | 95 % CI | W | L | rounds | natural | fit | likeness |`, `|---|---|---|---|---|---|---|---|---|`);
    for (const e of bt) lines.push(`| ${e.name} | ${e.score} | [${e.ci95[0]}, ${e.ci95[1]}] | ${e.wins} | ${e.losses} | ${e.rounds} | ${e.avgNaturalness ?? "-"} | ${e.avgFit ?? "-"} | ${e.avgLikeness ?? "-"} |`);
    lines.push(``);
  }
  return new Response(lines.join("\n"), { headers: { "Content-Type": "text/markdown; charset=utf-8" } });
}
