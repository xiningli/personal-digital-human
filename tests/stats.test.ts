import test from "node:test";
import assert from "node:assert/strict";
import { bradleyTerry, fitBradleyTerry, raterReliability } from "../lib/stats.ts";
import type { ArenaRound, ArenaVote, MotionPolicy } from "../lib/types.ts";

const P = (id: string): MotionPolicy => ({ id, name: id, source: "clip", clip: "Talking", headLift: 1, beats: 1, sway: 1, ease: 1 });
const round = (id: string, a: string, b: string, trial: ArenaRound["trial"] = "test", repeatOf?: string): ArenaRound => ({
  id, createdAt: "2026-09-17T00:00:00Z", trial, repeatOf, category: "answer", text: "x", audioPath: "/a.wav", emotion: "neutral", voiceModel: "v2", avatar: "/m.glb", seed: 1,
  candidates: [{ id: `${id}-a`, label: "A", policyId: a, policy: P(a) }, { id: `${id}-b`, label: "B", policyId: b, policy: P(b) }],
});
const vote = (roundId: string, winner: string | null): ArenaVote => ({ id: `v-${roundId}`, roundId, createdAt: "2026-09-17T00:00:00Z", winnerId: winner, ratings: {}, notes: "", candidateNotes: {} });

test("a policy that always wins ranks first with a finite score", () => {
  const s = bradleyTerry(["x", "y"], [{ winner: "x", loser: "y" }, { winner: "x", loser: "y" }]);
  assert.ok(s.get("x")! > s.get("y")!);
  assert.ok(Number.isFinite(s.get("x")!));
});

test("repeat trials are excluded from the ranking and counted for reliability", () => {
  const rounds = [round("r1", "x", "y"), round("r2", "x", "y", "repeat", "r1")];
  const votes = [vote("r1", "r1-a"), vote("r2", "r2-a")];
  const bt = fitBradleyTerry(rounds, votes, { bootstrap: 20 });
  assert.equal(bt.find((e) => e.policyId === "x")?.wins, 1);
  assert.deepEqual(raterReliability(rounds, votes), { repeatTrials: 1, repeatAgreement: 1 });
});

test("rounds from another voice checkpoint are never pooled by default", () => {
  const other = { ...round("r3", "z", "y"), voiceModel: "v1" };
  const bt = fitBradleyTerry([round("r1", "x", "y"), other], [vote("r1", "r1-a"), vote("r3", "r3-a")], { voiceModel: "v2", bootstrap: 0 });
  assert.ok(!bt.some((e) => e.policyId === "z"));
});
