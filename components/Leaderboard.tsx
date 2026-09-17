"use client";

import { useEffect, useState } from "react";
import type { BradleyTerryEntry, RaterReliability } from "@/lib/types";

interface Stats {
  voiceModel: string;
  voiceModels: string[];
  totalRounds: number;
  totalVotes: number;
  trials: { test: number; repeat: number };
  bt: BradleyTerryEntry[];
  reliability: RaterReliability;
}

export default function Leaderboard({ refresh }: { refresh: number }) {
  const [stats, setStats] = useState<Stats | null>(null);
  const [voiceModel, setVoiceModel] = useState<string>("");
  useEffect(() => {
    const q = voiceModel ? `?voiceModel=${encodeURIComponent(voiceModel)}` : "";
    fetch(`/api/arena/stats${q}`).then((r) => r.json()).then(setStats).catch(() => {});
  }, [refresh, voiceModel]);
  if (!stats) return null;
  const short = (m: string) => m.split("/").pop() ?? m;
  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-sm font-semibold text-gray-700">Ranking (Bradley-Terry, 95 % bootstrap CI)</h2>
        <select value={voiceModel || stats.voiceModel} onChange={(e) => setVoiceModel(e.target.value)} className="border rounded px-2 py-1 text-sm bg-white">
          {stats.voiceModels.map((m) => <option key={m} value={m}>voice: {short(m)}</option>)}
          <option value="all">all voices (pooled)</option>
        </select>
        <span className="text-xs text-gray-500">
          {stats.totalRounds} rounds · {stats.totalVotes} votes · repeat agreement {stats.reliability.repeatAgreement ?? "n/a"} ({stats.reliability.repeatTrials})
        </span>
        <a href="/api/arena/export" className="text-xs text-gray-500 hover:underline">export preference-pairs.jsonl</a>
        <a href="/api/report" className="text-xs text-gray-500 hover:underline">report.md</a>
      </div>
      {stats.bt.length ? (
        <table className="w-full text-sm">
          <thead className="text-xs text-gray-500 text-left">
            <tr><th className="py-1">policy</th><th>BT</th><th>CI</th><th>W</th><th>L</th><th>rounds</th><th>natural</th><th>fit</th><th>like me</th></tr>
          </thead>
          <tbody>
            {stats.bt.map((e) => (
              <tr key={e.policyId} className="border-t">
                <td className="py-1"><span className="text-xs text-gray-400 mr-2">{e.source}</span>{e.name}</td>
                <td className="font-mono">{e.score.toFixed(2)}</td>
                <td className="font-mono text-gray-500">[{e.ci95[0].toFixed(2)}, {e.ci95[1].toFixed(2)}]</td>
                <td>{e.wins}</td><td>{e.losses}</td><td>{e.rounds}</td>
                <td>{e.avgNaturalness ?? "–"}</td><td>{e.avgFit ?? "–"}</td><td>{e.avgLikeness ?? "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <p className="text-sm text-gray-400">No decided test rounds yet for this voice.</p>}
    </section>
  );
}
