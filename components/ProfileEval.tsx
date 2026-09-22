"use client";

// Triptych imitation eval (docs/protocol.md §5): the profile's source clip on the left, the
// extracted SMPL-X motion as a bare skeleton in the middle, the retargeted track on the
// avatar on the right, one transport driving all three. The video is the single master
// clock; the avatar's track action follows it (hard-corrected when the drift exceeds
// 80 ms) and the skeleton redraws the frame at floor(currentTime * fps). Sound comes only
// from the video, and its loudness drives the mouth — the same lip-sync the arena uses.
// The middle column splits "哪里不像" into two engineering stages: video↔skeleton judges
// the extraction (blame: "extract"), skeleton↔avatar judges the presentation
// (blame: "retarget"); the rating panel asks for that attribution when likeness is ≤ 3.
//
// Two modes. The default "逐段评测" walks the rater through the sentence segments one by
// one: the current segment loops on both sides until the rater scores it, a submission
// (POST with `segment`) auto-advances to the first unrated segment, and chips show which
// segments are done (✓ plus the latest score). "整段对比" is the original whole-clip loop
// whose submissions carry no segment. GET stats keep the two apart.

import { useCallback, useEffect, useRef, useState } from "react";
import { AvatarStage, speechFace, type AvatarState, type MotionTrack } from "@/player";
import SkeletonView, { type SkeletonViewHandle } from "./SkeletonView";
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

type Blame = "extract" | "retarget" | "unsure";
type BlameCounts = Record<Blame, number>;
type DimMeans = { likeness: number | null; timing: number | null; naturalness: number | null };
type EvalStats = {
  /** Whole-clip evals (no segment): count and means. */
  count: number;
  means: DimMeans;
  /** Blame tallies over the whole-clip evals (present when the server serves them). */
  blames?: BlameCounts;
  /** Indexed by segment i: submission count, and the latest submission's scores. */
  perSegment: { count: number; means: DimMeans; blames?: BlameCounts }[];
};

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
  const skeletonRef = useRef<SkeletonViewHandle>(null);
  const trackDurRef = useRef(0);
  const driftRef = useRef<number | null>(null);
  const lastTimeRef = useRef(0);
  /** The [startS, endS] window the per-segment mode loops inside; null in whole-clip mode. */
  const segLoopRef = useRef<{ start: number; end: number } | null>(null);
  const segmentsRef = useRef<MotionSegment[] | null>(null);
  const statsRef = useRef<EvalStats | null>(null);
  const currentRef = useRef<number | null>(null);
  const modeRef = useRef<"segments" | "whole">("segments");

  const [status, setStatus] = useState<AvatarState>("loading");
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(profile.durationS ?? 0);
  const [rate, setRate] = useState<1 | 0.5>(1);
  const [segments, setSegments] = useState<MotionSegment[] | null>(null);
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);
  const level = useLoudness(videoEl);

  const [mode, setMode] = useState<"segments" | "whole">("segments");
  /** Segment being evaluated in per-segment mode; null before auto-start / after finishing. */
  const [current, setCurrent] = useState<number | null>(null);

  const [likeness, setLikeness] = useState(0);
  const [timing, setTiming] = useState(0);
  const [naturalness, setNaturalness] = useState(0);
  const [blame, setBlame] = useState<Blame>("unsure");
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

  const seek = useCallback((t: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = t;
    stageRef.current?.seekTrack(t);
    lastTimeRef.current = t;
    setTime(t);
  }, []);

  // Jump to segment i for per-segment eval: seek both sides to its start, loop it, play.
  // A rated segment pre-fills the stars with its latest scores so re-rating starts from them.
  const goToSegment = useCallback((i: number) => {
    const seg = segmentsRef.current?.find((s) => s.i === i);
    if (!seg) return;
    currentRef.current = i;
    setCurrent(i);
    segLoopRef.current = { start: seg.startS, end: seg.endS };
    seek(seg.startS);
    const latest = statsRef.current?.perSegment[i];
    if (latest && latest.count > 0) {
      setLikeness(Math.round(latest.means.likeness ?? 0));
      setTiming(Math.round(latest.means.timing ?? 0));
      setNaturalness(Math.round(latest.means.naturalness ?? 0));
    } else {
      setLikeness(0); setTiming(0); setNaturalness(0);
    }
    setNote("");
    setBlame("unsure");
    const video = videoRef.current;
    if (video?.paused) void video.play().catch(() => {});
  }, [seek]);

  // Per-segment mode auto-starts on the first unrated segment once segments and stats are in.
  // Runs from the fetch callbacks below (not an effect) so it never cascades renders.
  const maybeAutoStart = useCallback(() => {
    if (modeRef.current !== "segments" || currentRef.current !== null) return;
    const segs = segmentsRef.current, st = statsRef.current;
    if (!segs || !st) return;
    const rated = new Set(st.perSegment.flatMap((s, i) => (s.count > 0 ? [i] : [])));
    const next = segs.find((s) => !rated.has(s.i));
    if (next) goToSegment(next.i);
  }, [goToSegment]);

  // The sync loop: every frame the avatar's track clock follows the video; a drift over
  // MAX_DRIFT (loop-aware) is hard-corrected. In per-segment mode the video wraps back to
  // the segment start when it runs past the end, so the segment loops until the rater acts.
  // Also the mouth and the smooth time display.
  useEffect(() => {
    const start = performance.now();
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const video = videoRef.current, stage = stageRef.current, trackDur = trackDurRef.current;
      if (!video) return;
      // The skeleton column has no clock of its own; it redraws off the master clock.
      skeletonRef.current?.setTime(video.currentTime);
      if (!stage) return;
      if (!video.paused && trackDur > 0) {
        const loop = segLoopRef.current;
        if (loop && (video.currentTime >= loop.end || video.currentTime < loop.start - 0.5)) {
          video.currentTime = loop.start;
          stage.seekTrack(loop.start);
          lastTimeRef.current = loop.start;
        }
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
      .then((d: { segments: MotionSegment[] }) => {
        segmentsRef.current = d.segments;
        setSegments(d.segments);
        maybeAutoStart();
      })
      .catch(() => setSegments(null));
  }, [profile.id, profile.hasSegments, maybeAutoStart]);

  // Existing eval stats (whole-clip count/means plus perSegment).
  const loadStats = useCallback(() => {
    fetch(`/api/profiles/${profile.id}/eval`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d: EvalStats) => {
        statsRef.current = d;
        setStats(d);
        maybeAutoStart();
      })
      .catch(() => setStats(null));
  }, [profile.id, maybeAutoStart]);
  useEffect(loadStats, [loadStats]);

  // A measuring handle, like the arena's __arena / the preview's __profiles, so sync can be
  // inspected instead of guessed at from screenshots.
  useEffect(() => {
    if (process.env.NODE_ENV === "production") return;
    const w = window as unknown as { __eval?: Record<string, unknown> };
    w.__eval = {
      get video() { return videoRef.current; },
      get stage() { return stageRef.current; },
      get skeleton() { return skeletonRef.current; },
      get drift() { return driftRef.current; },
      get currentSegment() { return segLoopRef.current; },
    };
  }, []);

  const total = segments?.length ?? 0;
  const perSegment = stats?.perSegment ?? [];
  const ratedCount = perSegment.filter((s) => s.count > 0).length;
  const allDone = total > 0 && ratedCount >= total;

  const switchMode = (m: "segments" | "whole") => {
    modeRef.current = m;
    setMode(m);
    if (m === "whole") {
      segLoopRef.current = null;
    } else if (current === null) {
      const st = statsRef.current;
      const rated = new Set((st?.perSegment ?? []).flatMap((s, i) => (s.count > 0 ? [i] : [])));
      const next = segments?.find((s) => !rated.has(s.i))?.i ?? segments?.[0]?.i;
      if (next != null) goToSegment(next);
    } else {
      goToSegment(current);
    }
  };

  const toggle = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) void video.play().catch(() => {});
    else video.pause();
  };

  const pickSegment = (s: MotionSegment) => {
    if (mode === "segments") { goToSegment(s.i); return; }
    seek(s.startS);
    const video = videoRef.current;
    if (video?.paused) void video.play().catch(() => {});
  };

  const submit = async () => {
    setBusy(true); setError(null);
    const seg = mode === "segments" ? current : null;
    // Blame is only collected (and only meaningful) when likeness says "not alike" (≤ 3).
    const withBlame = likeness >= 1 && likeness <= 3;
    try {
      const r = await fetch(`/api/profiles/${profile.id}/eval`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          likeness, timing, naturalness, note: note.trim() || undefined,
          ...(withBlame ? { blame } : {}),
          ...(seg !== null ? { segment: seg } : {}),
        }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error ?? `HTTP ${r.status}`);
      setLikeness(0); setTiming(0); setNaturalness(0); setBlame("unsure"); setNote("");
      loadStats();
      if (seg !== null && segments) {
        // Advance to the first unrated segment (counting this one as just rated);
        // when none is left the flow is done and the completion line shows.
        const rated = new Set(perSegment.flatMap((s, i) => (s.count > 0 ? [i] : [])));
        rated.add(seg);
        const next = segments.find((s) => !rated.has(s.i));
        if (next) goToSegment(next.i);
        else { currentRef.current = null; setCurrent(null); }
      }
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };

  const currentSeg = segments?.find((s) => s.i === current) ?? null;
  const activeSegment = mode === "segments"
    ? current
    : segments?.find((s) => time >= s.startS && time < s.endS)?.i ?? null;
  const fmt = (v: number | null) => (v == null ? "—" : v.toFixed(1));
  const segScore = (i: number): number | null => {
    const m = perSegment[i]?.means;
    if (!m || perSegment[i].count === 0) return null;
    return ((m.likeness ?? 0) + (m.timing ?? 0) + (m.naturalness ?? 0)) / 3;
  };

  // In per-segment mode the slider spans the current segment only; otherwise the whole clip.
  const sliderMin = mode === "segments" && currentSeg ? currentSeg.startS : 0;
  const sliderMax = mode === "segments" && currentSeg ? currentSeg.endS : duration || 0;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex rounded-lg bg-gray-100 p-1 text-sm">
          {([["segments", "逐段评测"], ["whole", "整段对比"]] as const).map(([m, label]) => (
            <button key={m} onClick={() => switchMode(m)}
              className={`px-3 py-1 rounded-md transition-colors ${mode === m ? "bg-white shadow-sm font-medium text-gray-900" : "text-gray-500 hover:text-gray-700"}`}>
              {label}
            </button>
          ))}
        </div>
        {mode === "segments" && total > 0 && (
          <span className="text-xs text-gray-500">
            {currentSeg ? `第 ${currentSeg.i + 1}/${total} 段` : `共 ${total} 段`} · 已评 {ratedCount}
            {allDone && <span className="ml-2 text-emerald-600">已评完 {ratedCount}/{total}，可重评任意段</span>}
          </span>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="relative rounded-xl overflow-hidden border border-gray-200 bg-black h-[40vh] lg:h-[60vh]">
          <video ref={videoRef} src={`/api/profiles/${profile.id}/video`} loop playsInline
            className="w-full h-full object-contain" />
          <span className="absolute top-2 left-2 text-xs text-white/80 bg-black/50 rounded px-2 py-0.5">原视频</span>
        </div>
        <div className="relative rounded-xl overflow-hidden border border-gray-200 bg-[#0d1117] h-[40vh] lg:h-[60vh]">
          <SkeletonView ref={skeletonRef} src={`/motion/profile-${profile.id}.joints3d.json`} />
          <span className="absolute top-2 left-2 text-xs text-white/80 bg-black/50 rounded px-2 py-0.5">骨架（抽取结果）</span>
          <span className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[11px] text-white/60 bg-black/50 rounded px-2 py-0.5 whitespace-nowrap">
            视频↔骨架 = 抽得对不对；骨架↔数字人 = 跟没跟上
          </span>
        </div>
        <div className="relative rounded-xl overflow-hidden border border-gray-200 bg-[#0d1117] h-[40vh] lg:h-[60vh]">
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
        <input type="range" min={sliderMin} max={sliderMax} step={0.01}
          value={Math.min(Math.max(time, sliderMin), sliderMax)}
          onChange={(e) => seek(Number(e.target.value))}
          className="flex-1 accent-gray-900" aria-label="Seek" />
        <span className="text-xs text-gray-500 tabular-nums shrink-0">
          {timecode(Math.max(time - sliderMin, 0))} / {timecode(Math.max(sliderMax - sliderMin, 0))}
        </span>
        <button onClick={() => setRate(rate === 1 ? 0.5 : 1)} title="Playback rate (0.5x for frame-by-frame checks)"
          className={`text-xs rounded-lg border px-2 py-1 shrink-0 tabular-nums ${rate === 1 ? "border-gray-200 text-gray-500" : "border-amber-300 bg-amber-50 text-amber-700"}`}>
          {rate === 1 ? "1x" : "0.5x"}
        </button>
      </div>

      {segments && mode === "segments" && (
        <div className="flex items-center gap-2">
          <div className="flex-1 h-1.5 rounded-full bg-gray-100 overflow-hidden" role="progressbar"
            aria-valuenow={ratedCount} aria-valuemin={0} aria-valuemax={total}>
            <div className="h-full bg-emerald-500 rounded-full transition-all"
              style={{ width: `${total ? (ratedCount / total) * 100 : 0}%` }} />
          </div>
          <span className="text-xs text-gray-500 tabular-nums shrink-0">{ratedCount}/{total}</span>
        </div>
      )}

      {segments && (
        <div className="flex gap-2 overflow-x-auto pb-1">
          {segments.map((s) => {
            const score = segScore(s.i);
            return (
              <button key={s.i} onClick={() => pickSegment(s)} title={s.text}
                className={`shrink-0 rounded-full border px-3 py-1 text-xs tabular-nums whitespace-nowrap transition-colors ${
                  activeSegment === s.i
                    ? "border-amber-300 bg-amber-100 text-gray-900 font-medium"
                    : score !== null
                      ? "border-emerald-300 bg-emerald-50 text-emerald-800 hover:bg-emerald-100"
                      : "border-gray-200 bg-white text-gray-600 hover:bg-gray-50"}`}>
                #{s.i + 1}{score !== null ? ` ✓${score.toFixed(1)}` : ` ${timecode(s.startS)}–${timecode(s.endS)}`}
              </button>
            );
          })}
        </div>
      )}

      <section className="bg-white border border-gray-200 rounded-xl p-5 space-y-4">
        <div className="flex items-baseline justify-between flex-wrap gap-2">
          <h2 className="text-sm font-semibold text-gray-700">
            像不像？ Rate the imitation
            {mode === "segments" && currentSeg && (
              <span className="ml-2 font-normal text-gray-500">第 {currentSeg.i + 1} 段</span>
            )}
          </h2>
          {stats && mode === "whole" && (
            <p className="text-xs text-gray-500">
              整段共 {stats.count} 次评测 · 均分 {fmt(stats.means.likeness)} / {fmt(stats.means.timing)} / {fmt(stats.means.naturalness)}
            </p>
          )}
          {stats && mode === "segments" && currentSeg && perSegment[currentSeg.i]?.count > 0 && (
            <p className="text-xs text-gray-500">
              本段已评 {perSegment[currentSeg.i].count} 次 · 最新 {fmt(perSegment[currentSeg.i].means.likeness)} / {fmt(perSegment[currentSeg.i].means.timing)} / {fmt(perSegment[currentSeg.i].means.naturalness)}
            </p>
          )}
        </div>
        {mode === "segments" && currentSeg && (
          <blockquote className="text-sm text-gray-600 border-l-2 border-amber-300 pl-3">
            “{currentSeg.text}” <span className="text-xs text-gray-400">{timecode(currentSeg.startS)}–{timecode(currentSeg.endS)}</span>
          </blockquote>
        )}
        <div className="space-y-2">
          <StarRow label="像不像本人" hint="likeness" value={likeness} onChange={setLikeness} />
          <StarRow label="节奏同步" hint="timing" value={timing} onChange={setTiming} />
          <StarRow label="自然度" hint="naturalness" value={naturalness} onChange={setNaturalness} />
        </div>
        {likeness >= 1 && likeness <= 3 && (
          <fieldset className="flex items-center gap-4 flex-wrap border border-amber-200 bg-amber-50 rounded-lg px-3 py-2">
            <legend className="text-xs text-gray-500 px-1">不像主要出在哪？</legend>
            {([
              ["extract", "抽错了（视频↔骨架就不像）"],
              ["retarget", "数字人没跟上（骨架对，呈现不像）"],
              ["unsure", "说不好"],
            ] as [Blame, string][]).map(([value, label]) => (
              <label key={value} className="flex items-center gap-1.5 text-sm text-gray-700 cursor-pointer">
                <input type="radio" name="blame" checked={blame === value} onChange={() => setBlame(value)}
                  className="accent-gray-900" />
                {label}
              </label>
            ))}
          </fieldset>
        )}
        <textarea value={note} onChange={(e) => setNote(e.target.value)} rows={2}
          placeholder="备注（可选）：哪里像、哪里不像…"
          className="w-full border rounded-lg px-3 py-2 text-sm" />
        <div className="flex items-center gap-3">
          <button onClick={submit} disabled={busy || !likeness || !timing || !naturalness || (mode === "segments" && current === null)}
            className="px-4 py-2 rounded-lg bg-gray-900 text-white text-sm disabled:opacity-50">
            {busy ? "submitting…" : mode === "segments" && !allDone ? "提交，下一段 →" : "提交评分"}
          </button>
          {error && <span className="text-sm text-red-600">{error}</span>}
        </div>
      </section>
    </div>
  );
}
