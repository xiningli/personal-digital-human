// Human eval scores for a profile's imitation quality (the eval page at /profiles/[id]).
// POST appends one submission to data/profile-evals.jsonl; a submission may carry `segment`
// (a MotionSegment.i) for the per-segment eval flow, or none for a whole-clip eval. GET
// returns the whole-clip count/means (evals without a segment) plus perSegment stats —
// for each segment the submission count and the means of its LATEST submission (re-rating
// appends a new line; the newest one wins, per docs/protocol.md §5).

import { appendProfileEval, getProfile, getProfileEvals, getProfileSegmentCount } from "@/lib/storage";
import type { ProfileEval } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const DIMENSIONS = ["likeness", "timing", "naturalness"] as const;
const BLAMES = ["extract", "retarget", "unsure"] as const;

function validScore(v: unknown): v is number {
  return typeof v === "number" && Number.isInteger(v) && v >= 1 && v <= 5;
}

function meanOf(evals: ProfileEval[]) {
  return Object.fromEntries(
    DIMENSIONS.map((d) => [d, evals.length ? evals.reduce((s, e) => s + e[d], 0) / evals.length : null]),
  );
}

/** Tallies of the optional blame field over a set of submissions (one per submission). */
function blameCounts(evals: ProfileEval[]): Record<(typeof BLAMES)[number], number> {
  const counts = { extract: 0, retarget: 0, unsure: 0 };
  for (const e of evals) if (e.blame) counts[e.blame] += 1;
  return counts;
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
  const { likeness, timing, naturalness, likeNote, dislikeNote, blame, segment } = body;
  if (![likeness, timing, naturalness].every(validScore)) {
    return Response.json({ error: "likeness, timing and naturalness must be integers 1-5" }, { status: 400 });
  }
  for (const [name, value] of [["likeNote", likeNote], ["dislikeNote", dislikeNote]] as const) {
    if (value !== undefined && typeof value !== "string") {
      return Response.json({ error: `${name} must be a string` }, { status: 400 });
    }
  }
  if (blame !== undefined && !(typeof blame === "string" && (BLAMES as readonly string[]).includes(blame))) {
    return Response.json({ error: `blame must be one of ${BLAMES.join(", ")}` }, { status: 400 });
  }
  if (segment !== undefined) {
    if (typeof segment !== "number" || !Number.isInteger(segment) || segment < 0) {
      return Response.json({ error: "segment must be a non-negative integer" }, { status: 400 });
    }
    const count = await getProfileSegmentCount(id);
    if (count == null) return Response.json({ error: "profile has no segments" }, { status: 400 });
    if (segment >= count) {
      return Response.json({ error: `segment must be < ${count}` }, { status: 400 });
    }
  }

  const text = (v: unknown) => (typeof v === "string" && v.trim() ? v.trim() : undefined);
  await appendProfileEval({
    profileId: id,
    ts: new Date().toISOString(),
    likeness: likeness as number, timing: timing as number, naturalness: naturalness as number,
    ...(text(likeNote) ? { likeNote: text(likeNote) } : {}),
    ...(text(dislikeNote) ? { dislikeNote: text(dislikeNote) } : {}),
    ...(blame !== undefined ? { blame: blame as ProfileEval["blame"] } : {}),
    ...(segment !== undefined ? { segment: segment as number } : {}),
  });
  return Response.json({ ok: true });
}

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const profile = await getProfile(id);
  if (!profile) return Response.json({ error: "not found" }, { status: 404 });

  const evals = await getProfileEvals(id);
  const whole = evals.filter((e) => e.segment === undefined);
  const maxSegment = evals.reduce((m, e) => (e.segment != null ? Math.max(m, e.segment + 1) : m), 0);
  const segmentCount = (await getProfileSegmentCount(id)) ?? maxSegment;

  const perSegment = Array.from({ length: segmentCount }, (_, i) => {
    const segEvals = evals.filter((e) => e.segment === i);
    const latest = segEvals[segEvals.length - 1];
    return {
      count: segEvals.length,
      means: latest
        ? { likeness: latest.likeness, timing: latest.timing, naturalness: latest.naturalness }
        : { likeness: null, timing: null, naturalness: null },
      blames: blameCounts(segEvals),
    };
  });

  return Response.json({ count: whole.length, means: meanOf(whole), blames: blameCounts(whole), perSegment });
}
