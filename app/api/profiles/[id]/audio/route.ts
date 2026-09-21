// The profile clip's audio, streamed so the profiles page can play a profile's motion with
// the voice it was extracted from. Range requests are honoured (the HTMLAudioElement needs
// them for seeking); `bytes=start-`, `bytes=start-end` and `bytes=-suffix` are all accepted.

import fs from "fs/promises";
import { createReadStream } from "fs";
import path from "path";
import { Readable } from "stream";
import { getProfile, profileDir } from "@/lib/storage";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const profile = await getProfile(id);
  if (!profile) return Response.json({ error: "not found" }, { status: 404 });

  const file = path.join(profileDir(id), "audio.wav");
  let size: number;
  try {
    size = (await fs.stat(file)).size;
  } catch {
    return Response.json({ error: "no audio" }, { status: 404 });
  }

  const headers: Record<string, string> = {
    "Content-Type": "audio/wav",
    "Accept-Ranges": "bytes",
    "Cache-Control": "no-store",
  };

  const range = /^bytes=(\d*)-(\d*)$/.exec(request.headers.get("range")?.trim() ?? "");
  if (range && (range[1] || range[2])) {
    let start: number;
    let end: number;
    if (range[1] === "") {
      // Suffix form: the last N bytes.
      const n = Number(range[2]);
      start = Math.max(0, size - n);
      end = size - 1;
    } else {
      start = Number(range[1]);
      end = range[2] === "" ? size - 1 : Math.min(Number(range[2]), size - 1);
    }
    if (start >= size || start > end) {
      return new Response(null, { status: 416, headers: { "Content-Range": `bytes */${size}` } });
    }
    headers["Content-Range"] = `bytes ${start}-${end}/${size}`;
    headers["Content-Length"] = String(end - start + 1);
    const body = Readable.toWeb(createReadStream(file, { start, end })) as ReadableStream;
    return new Response(body, { status: 206, headers });
  }

  headers["Content-Length"] = String(size);
  const body = Readable.toWeb(createReadStream(file)) as ReadableStream;
  return new Response(body, { status: 200, headers });
}
