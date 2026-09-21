"use client";

// Side-by-side imitation eval (docs/protocol.md §5): the profile's source clip on the left,
// the extracted track on the avatar on the right, one transport driving both. The video is
// the single master clock; the avatar's track action follows it (hard-corrected when the
// drift exceeds 80 ms). Sound comes only from the video, and its loudness drives the mouth —
// the same lip-sync the arena uses. Below: sentence chips that seek both sides, and the
// three-dimension star rating whose submissions go to /api/profiles/[id]/eval.

import { useCallback, useEffect, useRef, useState } from "react";
import { AvatarStage, speechFace, type AvatarState, type MotionTrack } from "@/player";
import { useLoudness } from "./useLoudness";
import type { MotionProfile, MotionSegment } from "@/lib/types";

const AVATAR = "/assets/model-clips.glb";
/** Avatar may lag the video this long before it is hard-corrected, seconds. */
const MAX_DRIFT = 0.08;

function timecode(s: number): string {
  const m = Math.floor(s / 60);
  const sec = (s - m * 60).toFixed(1).padStart(4, "0");
  return `${m}:${sec}`;
}

type EvalStats = { count: number; means: { likeness: number | null; timing: number | null; naturalness: number | null } };

/** One row of five clickable stars for a rating dimension. */
function StarRow({ label, hint, value, onChange }: {
  label: string;
  hint: string;
  value: number;
  onChange: (v: number) => void;
}) {
  const [hover, setHover] = useState(0);
  const shown = hover || value;
  return (
    <div className="flex items-center gap-4">
      <div className="w-32 shrink-0">
        <div className="text-sm font-medium text-gray-800">{label}</div>
        <div className="text-xs text-gray-400">{hint}</div>
      </div>
      <div className="flex" onMouseLeave={() => setHover(0)}>
        {[1, 2, 3, 4, 5].map((n) => (
          <button key={n} type="button" title={`${n}`}
            onMouseEnter={() => setHover(n)} onFocus={() => setHover(n)} onClick={() => onChange(n)}
            className={`px-0.5 text-2xl leading-none transition-colors ${shown >= n ? "text-amber-400" : "text-gray-300 hover:text-amber-200"}`}>
            ★
          </button>
        ))}
      </div>
      <span className="text-xs text-gray-400 tabular-nums w-4">{value || ""}</span>
    </div>
  );
}

export default function ProfileEval({ profile }: { profile: MotionProfile }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<AvatarStage | null>(null);
  const trackDurRef = useRef(0);
  const driftRef = useRef<number | null>(null);
  const lastTimeRef = useRef(0);

  const [status, setStatus] = useState<AvatarState>("loading");
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(profile.durationS ?? 0);
  const [rate, setRate] = useState<1 | 0.5>(1);
  const [segments, setSegments] = useState<MotionSegment[] | null>(null);
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);
  const level = useLoudness(videoEl);

  const [likeness, setLikeness] = useState(0);
  const [timing, setTiming] = useState(0);
  const [naturalness, setNaturalness] = useState(0);
  const [note, setNote] = useState("");
  const [stats, setStats] = useState<EvalStats | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The stage: load the track, let the speaking crossfade settle, then freeze on frame 0
  // until the transport says play.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !profile.trackPath) return;
    let cancelled = false;
    let stage: AvatarStage | null = null;
    let freezeTimer = 0;
    fetch(profile.trackPath)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`no track at ${profile.trackPath}`))))
      .then((track: MotionTrack) => {
        if (cancelled) return;
        trackDurRef.current = track.frames / track.fps;
        stage = new AvatarStage({
          canvas, ground: true, track,
          onState: (s) => {
            setStatus(s);
            if (s === "ready" && stage) {
              stage.setMotion("speaking");
              freezeTimer = window.setTimeout(() => {
                stage?.setPaused(true);
                stage?.seekTrack(0);
              }, 500);
            }
          },
        });
        stageRef.current = stage;
        stage.enter(AVATAR);
      })
      .catch(() => { if (!cancelled) setStatus("failed"); });
    return () => {
      cancelled = true;
      window.clearTimeout(freezeTimer);
      stageRef.current = null;
      stage?.loseContext();
    };
  }, [profile.trackPath]);

  // The video: the master clock. Play/pause on it moves the avatar with it.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    setVideoEl(video);
    const onMeta = () => setDuration(video.duration);
    const onPlay = () => { setPlaying(true); stageRef.current?.setPaused(false); };
    const onPause = () => { setPlaying(false); stageRef.current?.setPaused(true); };
    video.addEventListener("loadedmetadata", onMeta);
    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);
    return () => {
      video.removeEventListener("loadedmetadata", onMeta);
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPause);
      video.pause();
      setVideoEl(null);
    };
  }, []);

  // Playback rate: 1x normally, 0.5x for frame-by-frame inspection. Both clocks scale.
  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = rate;
    stageRef.current?.setTrackRate(rate);
  }, [rate]);

  // The sync loop: every frame the avatar's track clock follows the video; a drift over
  // MAX_DRIFT (loop-aware) is hard-corrected. Also the mouth and the smooth time display.
  useEffect(() => {
    const start = performance.now();
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const video = videoRef.current, stage = stageRef.current, trackDur = trackDurRef.current;
      if (!video || !stage) return;
      if (!video.paused && trackDur > 0) {
        const target = video.currentTime % trackDur;
        const at = stage.trackTime;
        if (at != null) {
          const d = Math.abs(at - target);
          const drift = Math.min(d, trackDur - d);
          driftRef.current = drift;
          if (drift > MAX_DRIFT) stage.seekTrack(video.currentTime);
        }
      }
      stage.setFace(speechFace(video.paused ? 0 : level(), (performance.now() - start) / 1000));
      if (Math.abs(video.currentTime - lastTimeRef.current) > 0.05) {
        lastTimeRef.current = video.currentTime;
        setTime(video.currentTime);
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [level]);

  // Sentence chips (public segments JSON; embeddings are server-side only).
  useEffect(() => {
    if (!profile.hasSegments) return;
    fetch(`/motion/profile-${profile.id}.segments.json`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d: { segments: MotionSegment[] }) => setSegments(d.segments))
      .catch(() => setSegments(null));
  }, [profile.id, profile.hasSegments]);

  // Existing eval stats.
  const loadStats = useCallback(() => {
    fetch(`/api/profiles/${profile.id}/eval`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setStats)
      .catch(() => setStats(null));
  }, [profile.id]);
  useEffect(loadStats, [loadStats]);

  // A measuring handle, like the arena's __arena / the preview's __profiles, so sync can be
  // inspected instead of guessed at from screenshots.
  useEffect(() => {
    if (process.env.NODE_ENV === "production") return;
    const w = window as unknown as { __eval?: Record<string, unknown> };
    w.__eval = {
      get video() { return videoRef.current; },
      get stage() { return stageRef.current; },
      get drift() { return driftRef.current; },
    };
  }, []);

  const seek = useCallback((t: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = t;
    stageRef.current?.seekTrack(t);
    lastTimeRef.current = t;
    setTime(t);
  }, []);

  const toggle = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) void video.play().catch(() => {});
    else video.pause();
  };

  const pickSegment = (s: MotionSegment) => {
    seek(s.startS);
    const video = videoRef.current;
    if (video?.paused) void video.play().catch(() => {});
  };

  const submit = async () => {
    setBusy(true); setError(null);
    try {
      const r = await fetch(`/api/profiles/${profile.id}/eval`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ likeness, timing, naturalness, note: note.trim() || undefined }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error ?? `HTTP ${r.status}`);
      setLikeness(0); setTiming(0); setNaturalness(0); setNote("");
      loadStats();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };

  const activeSegment = segments?.find((s) => time >= s.startS && time < s.endS)?.i ?? null;
  const fmt = (v: number | null) => (v == null ? "—" : v.toFixed(1));

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="relative rounded-xl overflow-hidden border border-gray-200 bg-black h-[40vh] lg:h-[68vh]">
          <video ref={videoRef} src={`/api/profiles/${profile.id}/video`} loop playsInline
            className="w-full h-full object-contain" />
          <span className="absolute top-2 left-2 text-xs text-white/80 bg-black/50 rounded px-2 py-0.5">原视频</span>
        </div>
        <div className="relative rounded-xl overflow-hidden border border-gray-200 bg-[#0d1117] h-[40vh] lg:h-[68vh]">
          <canvas ref={canvasRef} className="w-full h-full block" />
          <span className="absolute top-2 left-2 text-xs text-white/80 bg-black/50 rounded px-2 py-0.5">数字人</span>
          {status !== "ready" && (
            <div className="absolute inset-0 grid place-items-center text-center text-sm text-white/70 px-6">
              {status === "loading" ? "loading motion track…" : "the track failed to load"}
            </div>
          )}
        </div>
      </div>

      <div className="flex items-center gap-3 bg-white border border-gray-200 rounded-xl px-4 py-3">
        <button onClick={toggle} title={playing ? "Pause" : "Play"}
          className="w-9 h-9 rounded-full bg-gray-900 text-white grid place-items-center text-sm shrink-0 disabled:opacity-50"
          disabled={status !== "ready"}>
          {playing ? "⏸" : "▶"}
        </button>
        <input type="range" min={0} max={duration || 0} step={0.01} value={Math.min(time, duration || 0)}
          onChange={(e) => seek(Number(e.target.value))}
          className="flex-1 accent-gray-900" aria-label="Seek" />
        <span className="text-xs text-gray-500 tabular-nums shrink-0">{timecode(time)} / {timecode(duration)}</span>
        <button onClick={() => setRate(rate === 1 ? 0.5 : 1)} title="Playback rate (0.5x for frame-by-frame checks)"
          className={`text-xs rounded-lg border px-2 py-1 shrink-0 tabular-nums ${rate === 1 ? "border-gray-200 text-gray-500" : "border-amber-300 bg-amber-50 text-amber-700"}`}>
          {rate === 1 ? "1x" : "0.5x"}
        </button>
      </div>

      {segments && (
        <div className="flex gap-2 overflow-x-auto pb-1">
          {segments.map((s) => (
            <button key={s.i} onClick={() => pickSegment(s)} title={s.text}
              className={`shrink-0 rounded-full border px-3 py-1 text-xs tabular-nums whitespace-nowrap transition-colors ${
                activeSegment === s.i
                  ? "border-amber-300 bg-amber-100 text-gray-900 font-medium"
                  : "border-gray-200 bg-white text-gray-600 hover:bg-gray-50"}`}>
              #{s.i + 1} {timecode(s.startS)}–{timecode(s.endS)}
            </button>
          ))}
        </div>
      )}

      <section className="bg-white border border-gray-200 rounded-xl p-5 space-y-4">
        <div className="flex items-baseline justify-between flex-wrap gap-2">
          <h2 className="text-sm font-semibold text-gray-700">像不像？ Rate the imitation</h2>
          {stats && (
            <p className="text-xs text-gray-500">
              共 {stats.count} 次评测 · 均分 {fmt(stats.means.likeness)} / {fmt(stats.means.timing)} / {fmt(stats.means.naturalness)}
            </p>
          )}
        </div>
        <div className="space-y-2">
          <StarRow label="像不像本人" hint="likeness" value={likeness} onChange={setLikeness} />
          <StarRow label="节奏同步" hint="timing" value={timing} onChange={setTiming} />
          <StarRow label="自然度" hint="naturalness" value={naturalness} onChange={setNaturalness} />
        </div>
        <textarea value={note} onChange={(e) => setNote(e.target.value)} rows={2}
          placeholder="备注（可选）：哪里像、哪里不像…"
          className="w-full border rounded-lg px-3 py-2 text-sm" />
        <div className="flex items-center gap-3">
          <button onClick={submit} disabled={busy || !likeness || !timing || !naturalness}
            className="px-4 py-2 rounded-lg bg-gray-900 text-white text-sm disabled:opacity-50">
            {busy ? "submitting…" : "提交评分"}
          </button>
          {error && <span className="text-sm text-red-600">{error}</span>}
        </div>
      </section>
    </div>
  );
}
