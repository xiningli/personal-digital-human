// Human eval scores for a profile's imitation quality (the eval page at /profiles/[id]).
// POST appends one submission to data/profile-evals.jsonl; GET returns the running count
// and per-dimension means for this profile.

import { appendProfileEval, getProfile, getProfileEvals } from "@/lib/storage";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const DIMENSIONS = ["likeness", "timing", "naturalness"] as const;

function validScore(v: unknown): v is number {
  return typeof v === "number" && Number.isInteger(v) && v >= 1 && v <= 5;
}

export async function POST(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const profile = await getProfile(id);
  if (!profile) return Response.json({ error: "not found" }, { status: 404 });

  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "invalid JSON" }, { status: 400 });
  }
  const { likeness, timing, naturalness, note } = body;
  if (![likeness, timing, naturalness].every(validScore)) {
    return Response.json({ error: "likeness, timing and naturalness must be integers 1-5" }, { status: 400 });
  }
  if (note !== undefined && typeof note !== "string") {
    return Response.json({ error: "note must be a string" }, { status: 400 });
  }

  const noteText = typeof note === "string" && note.trim() ? note.trim() : undefined;
  await appendProfileEval({
    profileId: id,
    ts: new Date().toISOString(),
    likeness: likeness as number, timing: timing as number, naturalness: naturalness as number,
    ...(noteText ? { note: noteText } : {}),
  });
  return Response.json({ ok: true });
}

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const profile = await getProfile(id);
  if (!profile) return Response.json({ error: "not found" }, { status: 404 });

  const evals = (await getProfileEvals()).filter((e) => e.profileId === id);
  const means = Object.fromEntries(
    DIMENSIONS.map((d) => [d, evals.length ? evals.reduce((s, e) => s + e[d], 0) / evals.length : null]),
  );
  return Response.json({ count: evals.length, means });
}
