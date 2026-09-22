import { NextRequest } from "next/server";
import { createWriteStream } from "node:fs";
import fs from "fs/promises";
import path from "path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { v4 as uuid } from "uuid";
import { getProfiles, profileDir, saveProfile } from "@/lib/storage";
import { startProfileExtraction } from "@/lib/profiles";
import type { MotionProfile } from "@/lib/types";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

const MAX_BYTES = 200 * 1024 * 1024;
const VIDEO_EXT = /\.(mp4|mov|webm)$/i;
const STYLES = new Set(["presentation", "stage"]);

/** Optional profile style (docs/protocol.md §5); an invalid value is a 400, not a silent drop. */
function parseStyle(raw: unknown): { style?: MotionProfile["style"] } | { error: string } {
  if (raw === undefined || raw === null || raw === "") return {};
  if (typeof raw === "string" && STYLES.has(raw)) return { style: raw as MotionProfile["style"] };
  return { error: `style must be one of: ${[...STYLES].join(", ")}` };
}

export async function GET() {
  return Response.json(await getProfiles());
}

/**
 * Add a profile (docs/protocol.md §5). JSON `{name, url, start?, duration?, style?}` fetches and
 * trims a YouTube clip; multipart with `file` + `name` uses an uploaded video. The profile
 * is created as "processing" and returned at once; extraction runs in the background.
 */
export async function POST(request: NextRequest) {
  const type = request.headers.get("content-type") ?? "";
  const id = uuid();
  let profile: MotionProfile;
  let opts: { start?: number; duration?: number; maxFootFloatCm?: number } = {};

  if (type.includes("multipart/form-data")) {
    const form = await request.formData();
    const file = form.get("file");
    const name = String(form.get("name") ?? "").trim();
    if (!(file instanceof File)) return Response.json({ error: "a video file is required" }, { status: 400 });
    if (!name) return Response.json({ error: "a name is required" }, { status: 400 });
    if (file.size > MAX_BYTES) return Response.json({ error: "the clip is over 200 MB" }, { status: 413 });
    if (!VIDEO_EXT.test(file.name) && !file.type.startsWith("video/"))
      return Response.json({ error: "the clip must be mp4, mov or webm" }, { status: 415 });
    const parsed = parseStyle(form.get("style"));
    if ("error" in parsed) return Response.json({ error: parsed.error }, { status: 400 });
    profile = {
      id, name, createdAt: new Date().toISOString(),
      sourceType: "upload", sourceRef: file.name, status: "processing",
      ...(parsed.style ? { style: parsed.style } : {}),
    };
    await fs.mkdir(profileDir(id), { recursive: true });
    await pipeline(Readable.fromWeb(file.stream() as never), createWriteStream(path.join(profileDir(id), "source.mp4")));
  } else {
    const body = (await request.json().catch(() => null)) as { name?: string; url?: string; start?: number; duration?: number; maxFootFloatCm?: number; style?: string } | null;
    const name = body?.name?.trim();
    const url = body?.url?.trim();
    if (!name || !url) return Response.json({ error: "name and url are required" }, { status: 400 });
    if (!/^https?:\/\//.test(url)) return Response.json({ error: "url must be http(s)" }, { status: 400 });
    const parsed = parseStyle(body?.style);
    if ("error" in parsed) return Response.json({ error: parsed.error }, { status: 400 });
    opts = {
      start: Number.isFinite(body?.start) ? Number(body?.start) : undefined,
      duration: Number.isFinite(body?.duration) ? Number(body?.duration) : undefined,
      maxFootFloatCm: Number.isFinite(body?.maxFootFloatCm) ? Number(body?.maxFootFloatCm) : undefined,
    };
    profile = {
      id, name, createdAt: new Date().toISOString(),
      sourceType: "youtube", sourceRef: url, status: "processing",
      ...(parsed.style ? { style: parsed.style } : {}),
    };
  }

  await saveProfile(profile);
  startProfileExtraction(profile, opts);
  return Response.json(profile, { status: 201 });
}
