// Semantic gesture matching: each sentence the digital human is about to say is embedded with
// the same multilingual MiniLM the extraction side used, and matched against the profile's
// sentence segments by cosine similarity. The profile's segments.json is server-only (it
// carries the embeddings); the client sends plain sentences and gets frame ranges back.

import fs from "fs/promises";
import path from "path";
import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";
import { profileDir } from "@/lib/storage";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const MODEL = "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
/** Gestures shorter than this read as a flinch rather than a movement, so they are penalised. */
const MIN_SECONDS = 0.6;
const SHORT_PENALTY = 0.8;

// The extraction side (motion/segment.py) writes this; defined locally so this route does not
// depend on lib/types.ts, which is being extended in parallel.
interface Segment {
  i: number;
  startS: number;
  endS: number;
  startFrame: number;
  endFrame: number;
  text: string;
  embedding: number[];
}

interface SegmentsFile {
  version: number;
  fps: number;
  frames: number;
  embedModel: string;
  segments: Segment[];
}

/** One matched segment, mirrored in lib/motion-segments.ts for the client. */
interface SegmentMatch {
  i: number;
  score: number;
  startS: number;
  endS: number;
  startFrame: number;
  endFrame: number;
  text: string;
}

let extractorPromise: Promise<FeatureExtractionPipeline> | null = null;

/** One model per server process; the first request pays the load, the rest reuse it. */
function getExtractor(): Promise<FeatureExtractionPipeline> {
  extractorPromise ??= pipeline("feature-extraction", MODEL, { dtype: "q8" });
  return extractorPromise;
}

async function readSegments(id: string): Promise<Segment[]> {
  try {
    const file = JSON.parse(await fs.readFile(path.join(profileDir(id), "segments.json"), "utf-8")) as SegmentsFile;
    return Array.isArray(file.segments) ? file.segments.filter((s) => Array.isArray(s.embedding) && s.embedding.length) : [];
  } catch {
    return [];
  }
}

export async function POST(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  let texts: string[];
  try {
    const body = await request.json();
    if (!Array.isArray(body?.texts) || body.texts.some((t: unknown) => typeof t !== "string")) throw new Error();
    texts = body.texts;
  } catch {
    return Response.json({ error: "expected { texts: string[] }" }, { status: 400 });
  }

  // Blank sentences are never embedded; they stay in place as null so the reply keeps the
  // input's length and order.
  const embedAt = texts.map((t, idx) => ({ t: t.trim(), idx })).filter((x) => x.t);
  const segments = await readSegments(id);
  const matches: (SegmentMatch | null)[] = texts.map(() => null);
  if (!segments.length || !embedAt.length) return Response.json({ matches });

  let queries: number[][];
  try {
    const extractor = await getExtractor();
    const out = await extractor(embedAt.map((x) => x.t), { pooling: "mean", normalize: true });
    queries = out.tolist() as number[][];
  } catch (error) {
    extractorPromise = null; // a failed load must not poison every later request
    return Response.json({ error: `embedding model unavailable: ${(error as Error).message}` }, { status: 502 });
  }

  for (const [k, q] of queries.entries()) {
    const qn = Math.hypot(...q) || 1;
    let best: Segment | null = null;
    let bestScore = -Infinity;
    for (const s of segments) {
      let dot = 0;
      for (let d = 0; d < q.length; d++) dot += q[d] * s.embedding[d];
      const sn = Math.hypot(...s.embedding) || 1;
      let score = dot / (qn * sn);
      if (s.endS - s.startS < MIN_SECONDS) score *= SHORT_PENALTY;
      if (score > bestScore) { bestScore = score; best = s; }
    }
    if (best) {
      matches[embedAt[k].idx] = {
        i: best.i,
        score: bestScore,
        startS: best.startS,
        endS: best.endS,
        startFrame: best.startFrame,
        endFrame: best.endFrame,
        text: best.text,
      };
    }
  }
  return Response.json({ matches });
}
