"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ProfilePreview, { type PreviewPlayback } from "./ProfilePreview";
import type { MotionProfile, MotionSegment } from "@/lib/types";

function timecode(s: number): string {
  const m = Math.floor(s / 60);
  const sec = (s - m * 60).toFixed(1).padStart(4, "0");
  return `${m}:${sec}`;
}

/** Sentence-level gesture segments from segment.py (public profile-<id>.segments.json).
 *  Clicking a row plays just that sentence — motion and audio together; clicking it again stops. */
function SegmentList({ profileId, active, onSelect }: {
  profileId: string;
  active: number | null;
  onSelect: (segment: MotionSegment | null) => void;
}) {
  const [segments, setSegments] = useState<MotionSegment[] | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    fetch(`/motion/profile-${profileId}.segments.json`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d: { segments: MotionSegment[] }) => setSegments(d.segments))
      .catch(() => setSegments(null));
  }, [profileId]);

  if (!segments) return null;
  return (
    <div className="text-xs text-gray-600">
      <button onClick={() => setOpen(!open)} className="hover:underline">
        {segments.length} gesture segment{segments.length === 1 ? "" : "s"} {open ? "▾" : "▸"}
      </button>
      {open && (
        <ul className="mt-1 space-y-1 max-h-40 overflow-y-auto">
          {segments.map((s) => (
            <li key={s.i}>
              <button onClick={() => onSelect(active === s.i ? null : s)}
                className={`text-left rounded px-1 -mx-1 hover:bg-gray-100 ${active === s.i ? "bg-amber-100 text-gray-900 font-medium" : ""}`}>
                <span className="text-gray-400 tabular-nums">{timecode(s.startS)}–{timecode(s.endS)}</span>{" "}
                {s.text}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function ProfilePanel() {
  const [profiles, setProfiles] = useState<MotionProfile[]>([]);
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [start, setStart] = useState("");
  const [duration, setDuration] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  /** Per-profile playback state shared by the preview (play button) and the segment list. */
  const [playback, setPlayback] = useState<Record<string, PreviewPlayback | null>>({});

  const refresh = useCallback(() => {
    fetch("/api/profiles", { cache: "no-store" }).then((r) => r.json()).then(setProfiles).catch(() => {});
  }, []);
  useEffect(refresh, [refresh]);

  // Extraction runs server-side in the background; poll while anything is still processing.
  useEffect(() => {
    if (!profiles.some((p) => p.status === "processing")) return;
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [profiles, refresh]);

  const submit = async () => {
    setBusy(true); setError(null);
    try {
      let r: Response;
      if (file) {
        const form = new FormData();
        form.set("name", name);
        form.set("file", file);
        r = await fetch("/api/profiles", { method: "POST", body: form });
      } else {
        r = await fetch("/api/profiles", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name, url,
            start: start ? Number(start) : undefined,
            duration: duration ? Number(duration) : undefined,
          }),
        });
      }
      const data = await r.json();
      if (!r.ok) throw new Error(data.error ?? `HTTP ${r.status}`);
      setName(""); setUrl(""); setStart(""); setDuration(""); setFile(null);
      if (fileRef.current) fileRef.current.value = "";
      refresh();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };

  const remove = async (p: MotionProfile) => {
    if (!window.confirm(`Delete “${p.name}”? The clip, the extracted motion and its arena candidacy all go.`)) return;
    await fetch(`/api/profiles/${p.id}`, { method: "DELETE" });
    refresh();
  };

  const canSubmit = !busy && !!name.trim() && (!!file || !!url.trim());

  return (
    <div className="space-y-8">
      <section className="bg-white border border-gray-200 rounded-xl p-5 space-y-3">
        <h2 className="text-sm font-semibold text-gray-700">Add a profile</h2>
        <div className="flex flex-wrap items-center gap-3">
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name (e.g. Amy Cuddy TED)"
            className="border rounded-lg px-3 py-2 text-sm flex-1 min-w-48" />
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="YouTube URL" disabled={!!file}
            className="border rounded-lg px-3 py-2 text-sm flex-1 min-w-64 disabled:opacity-50" />
          <input value={start} onChange={(e) => setStart(e.target.value)} placeholder="start s" disabled={!!file}
            className="border rounded-lg px-3 py-2 text-sm w-24 disabled:opacity-50" />
          <input value={duration} onChange={(e) => setDuration(e.target.value)} placeholder="duration s" disabled={!!file}
            className="border rounded-lg px-3 py-2 text-sm w-24 disabled:opacity-50" />
        </div>
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <span className="text-gray-500">or</span>
          <input ref={fileRef} type="file" accept="video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="text-sm" />
          <button onClick={submit} disabled={!canSubmit}
            className="px-4 py-2 rounded-lg bg-gray-900 text-white disabled:opacity-50">
            {busy ? "adding…" : "Extract motion"}
          </button>
          {error && <span className="text-sm text-red-600">{error}</span>}
        </div>
        <p className="text-xs text-gray-400">
          A static-camera clip of one person, ~30 s is plenty. Extraction runs on the GPU one job at a time; a new profile appears in the arena once it is ready.
        </p>
      </section>

      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {profiles.map((p) => {
          const pb = playback[p.id] ?? null;
          const activeSegment = pb?.kind === "segment" ? pb.segment.i : null;
          return (
          <div key={p.id} className="bg-white border border-gray-200 rounded-xl p-4 space-y-3">
            <div>
              <div className="font-semibold text-sm">{p.name}</div>
              <div className="text-xs text-gray-500 truncate" title={p.sourceRef}>
                {p.sourceType === "youtube" ? "YouTube" : "upload"} · {p.sourceRef}
              </div>
              <div className="text-xs text-gray-400">{new Date(p.createdAt).toLocaleString()}</div>
            </div>

            {p.status === "processing" && (
              <div className="text-sm text-gray-500 animate-pulse py-8 text-center">extracting motion…</div>
            )}
            {p.status === "failed" && (
              <div className="text-sm text-red-600 break-words">{p.error ?? "extraction failed"}</div>
            )}
            {p.status === "ready" && p.trackPath && (
              <>
                <ProfilePreview trackPath={p.trackPath} audioPath={`/api/profiles/${p.id}/audio`}
                  playback={pb} onPlayback={(next) => setPlayback((m) => ({ ...m, [p.id]: next }))} />
                {p.stats && (
                  <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs text-gray-600">
                    <dt>worst limb</dt><dd className="text-right">{p.stats.worstLimbDeg.toFixed(2)}°</dd>
                    <dt>foot float</dt><dd className="text-right">{p.stats.footFloatCm.toFixed(1)} cm</dd>
                    <dt>foot skate</dt><dd className="text-right">{p.stats.footSkateCmPerFrame.toFixed(2)} cm/frame</dd>
                    <dt>diversity</dt><dd className="text-right">{p.stats.diversity.toFixed(2)}</dd>
                  </dl>
                )}
                {p.hasSegments && (
                  <SegmentList profileId={p.id} active={activeSegment}
                    onSelect={(s) => setPlayback((m) => ({ ...m, [p.id]: s ? { kind: "segment", segment: s } : null }))} />
                )}
              </>
            )}

            <div className="flex justify-end">
              <button onClick={() => remove(p)} className="text-xs text-red-600 hover:underline">Delete</button>
            </div>
          </div>
          );
        })}
        {!profiles.length && (
          <div className="text-sm text-gray-400 col-span-full py-4">No profiles yet. Add a clip of someone whose body language you admire.</div>
        )}
      </section>
    </div>
  );
}
