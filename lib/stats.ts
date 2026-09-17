// Bradley-Terry ranking of motion policies with bootstrap confidence intervals, and rater
// reliability from repeat trials (docs/protocol.md §2). Pure functions; no I/O. The
// estimator is the one the voice studio uses; only the item key differs (policy, not
// instruction).

import type { ArenaRound, ArenaVote, BradleyTerryEntry, RaterReliability, TrialType } from "./types";

export interface BTOptions {
  /** Partition: only rounds spoken by this voice checkpoint; "all" pools them. */
  voiceModel?: string;
  excludeTrials?: TrialType[];
  bootstrap?: number;
  seed?: number;
  maxIter?: number;
}

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Match { winner: string; loser: string }

const inPartition = (r: ArenaRound, voiceModel?: string) => !voiceModel || voiceModel === "all" || r.voiceModel === voiceModel;

function decidedMatches(rounds: ArenaRound[], votes: ArenaVote[], opts: BTOptions): Match[] {
  const byId = new Map(rounds.map((r) => [r.id, r]));
  const exclude = new Set(opts.excludeTrials ?? ["repeat"]);
  const matches: Match[] = [];
  for (const vote of votes) {
    const round = byId.get(vote.roundId);
    if (!round || !vote.winnerId || exclude.has(round.trial) || !inPartition(round, opts.voiceModel)) continue;
    const winner = round.candidates.find((c) => c.id === vote.winnerId);
    if (!winner) continue;
    for (const loser of round.candidates) {
      if (loser.id === winner.id || loser.policyId === winner.policyId) continue;
      matches.push({ winner: winner.policyId, loser: loser.policyId });
    }
  }
  return matches;
}

/**
 * Bradley-Terry MLE by the MM (Zermelo) iteration with a half-win prior against a virtual
 * opponent, so an item with only wins or only losses stays finite. Mean-centred log strengths.
 */
export function bradleyTerry(items: string[], matches: Match[], maxIter = 500): Map<string, number> {
  const idx = new Map(items.map((it, i) => [it, i]));
  const n = items.length;
  const wins = new Array<number>(n).fill(0);
  const pairs = new Map<string, number>();
  for (const m of matches) {
    const i = idx.get(m.winner);
    const j = idx.get(m.loser);
    if (i === undefined || j === undefined) continue;
    wins[i] += 1;
    const key = i < j ? `${i},${j}` : `${j},${i}`;
    pairs.set(key, (pairs.get(key) ?? 0) + 1);
  }
  const prior = 0.5;
  const p = new Array<number>(n).fill(1);
  for (let iter = 0; iter < maxIter; iter++) {
    let maxDelta = 0;
    const next = new Array<number>(n).fill(1);
    for (let i = 0; i < n; i++) {
      let denom = (2 * prior) / (p[i] + 1);
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const key = i < j ? `${i},${j}` : `${j},${i}`;
        const g = pairs.get(key);
        if (!g) continue;
        denom += g / (p[i] + p[j]);
      }
      next[i] = (wins[i] + prior) / denom;
      maxDelta = Math.max(maxDelta, Math.abs(Math.log(next[i]) - Math.log(p[i])));
    }
    const geo = Math.exp(next.reduce((s, v) => s + Math.log(v), 0) / n);
    for (let i = 0; i < n; i++) p[i] = next[i] / geo;
    if (maxDelta < 1e-6) break;
  }
  return new Map(items.map((it, i) => [it, Math.log(p[i])]));
}

const mean = (values: number[]): number | null => (values.length ? Math.round((values.reduce((s, v) => s + v, 0) / values.length) * 1000) / 1000 : null);

export function fitBradleyTerry(rounds: ArenaRound[], votes: ArenaVote[], opts: BTOptions = {}): BradleyTerryEntry[] {
  const exclude = new Set(opts.excludeTrials ?? ["repeat"]);
  const testRounds = rounds.filter((r) => inPartition(r, opts.voiceModel) && !exclude.has(r.trial));
  const matches = decidedMatches(rounds, votes, opts);
  const policies = new Map<string, ArenaRound["candidates"][number]["policy"]>();
  for (const r of testRounds) for (const c of r.candidates) if (!policies.has(c.policyId)) policies.set(c.policyId, c.policy);
  const items = [...policies.keys()];
  if (!items.length) return [];
  const strengths = bradleyTerry(items, matches, opts.maxIter);

  const B = opts.bootstrap ?? 1000;
  const rng = mulberry32(opts.seed ?? 20260916);
  const samples = new Map<string, number[]>(items.map((it) => [it, []]));
  if (matches.length && B > 0) {
    for (let b = 0; b < B; b++) {
      const resampled: Match[] = [];
      for (let k = 0; k < matches.length; k++) resampled.push(matches[Math.floor(rng() * matches.length)]);
      const s = bradleyTerry(items, resampled, 200);
      for (const it of items) samples.get(it)!.push(s.get(it) ?? 0);
    }
  }

  const acc = new Map<string, { wins: number; losses: number; rounds: Set<string>; nat: number[]; fit: number[]; like: number[] }>();
  const get = (it: string) => {
    let a = acc.get(it);
    if (!a) { a = { wins: 0, losses: 0, rounds: new Set(), nat: [], fit: [], like: [] }; acc.set(it, a); }
    return a;
  };
  for (const r of testRounds) for (const c of r.candidates) get(c.policyId).rounds.add(r.id);
  for (const m of matches) { get(m.winner).wins += 1; get(m.loser).losses += 1; }
  const byId = new Map(testRounds.map((r) => [r.id, r]));
  for (const v of votes) {
    const r = byId.get(v.roundId);
    if (!r) continue;
    for (const c of r.candidates) {
      const rt = v.ratings[c.id];
      if (!rt) continue;
      const a = get(c.policyId);
      a.nat.push(rt.naturalness); a.fit.push(rt.fit); a.like.push(rt.likeness);
    }
  }
  const entries: BradleyTerryEntry[] = items.map((it) => {
    const a = get(it);
    const sorted = [...(samples.get(it) ?? [])].sort((x, y) => x - y);
    const q = (p: number) => (sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))] : 0);
    const score = strengths.get(it) ?? 0;
    const policy = policies.get(it)!;
    return {
      policyId: it,
      name: policy.name,
      source: policy.source,
      score: Math.round(score * 1000) / 1000,
      ci95: sorted.length ? [Math.round(q(0.025) * 1000) / 1000, Math.round(q(0.975) * 1000) / 1000] : [score, score],
      wins: a.wins,
      losses: a.losses,
      rounds: a.rounds.size,
      avgNaturalness: mean(a.nat),
      avgFit: mean(a.fit),
      avgLikeness: mean(a.like),
    };
  });
  return entries.sort((x, y) => y.score - x.score);
}

export function raterReliability(rounds: ArenaRound[], votes: ArenaVote[], voiceModel?: string): RaterReliability {
  const byId = new Map(rounds.map((r) => [r.id, r]));
  const voteByRound = new Map<string, ArenaVote>();
  for (const v of votes) if (!voteByRound.has(v.roundId)) voteByRound.set(v.roundId, v);
  const winningPolicy = (round: ArenaRound, vote: ArenaVote | undefined): string | null =>
    vote?.winnerId ? round.candidates.find((c) => c.id === vote.winnerId)?.policyId ?? null : null;
  let repeatTrials = 0;
  let repeatAgree = 0;
  for (const r of rounds) {
    if (r.trial !== "repeat" || !r.repeatOf || !inPartition(r, voiceModel)) continue;
    const v = voteByRound.get(r.id);
    const orig = byId.get(r.repeatOf);
    const ov = orig ? voteByRound.get(orig.id) : undefined;
    if (!v || !orig || !ov) continue;
    repeatTrials += 1;
    const a = winningPolicy(r, v);
    const b = winningPolicy(orig, ov);
    if (a === b) repeatAgree += 1;
  }
  return { repeatTrials, repeatAgreement: repeatTrials ? Math.round((repeatAgree / repeatTrials) * 1000) / 1000 : null };
}
