"use client";

// A profile's track, played on the avatar the way the arena plays a generated candidate —
// now with the clip's own audio under it (GET /api/profiles/<id>/audio). One button plays the
// whole take in sync (the track loops, so the audio loops too), and the panel's sentence list
// can ask for a single segment: the track is cut to that frame range and the audio plays
// startS..endS, then everything stops and settles back to idle. While audio plays, its
// loudness drives the mouth (speechFace), the same lip-sync the arena uses.

import { useEffect, useRef, useState } from "react";
import { AvatarStage, speechFace, type AvatarState, type MotionTrack } from "@/player";
import { assembleTrack } from "@/lib/motion-segments";
import { useLoudness } from "./useLoudness";
import type { MotionSegment } from "@/lib/types";

const AVATAR = "/assets/model-clips.glb";

/** What the preview is playing: the whole take, or one sentence segment. Null means stopped. */
export type PreviewPlayback = { kind: "full" } | { kind: "segment"; segment: MotionSegment };

function timecode(s: number): string {
  const m = Math.floor(s / 60);
  const sec = (s - m * 60).toFixed(0).padStart(2, "0");
  return `${m}:${sec}`;
}

export default function ProfilePreview({ trackPath, audioPath, playback, onPlayback }: {
  trackPath: string;
  audioPath: string;
  playback: PreviewPlayback | null;
  onPlayback: (playback: PreviewPlayback | null) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<AvatarStage | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const trackRef = useRef<MotionTrack | null>(null);
  /** True when playback was cleared by pausing the full take, so play resumes where it left off. */
  const pausedFullRef = useRef(false);
  const [status, setStatus] = useState<AvatarState>("loading");
  const [durationS, setDurationS] = useState<number | null>(null);
  const [audioEl, setAudioEl] = useState<HTMLAudioElement | null>(null);
  const level = useLoudness(audioEl);

  const playbackRef = useRef(playback);
  const onPlaybackRef = useRef(onPlayback);
  useEffect(() => { playbackRef.current = playback; onPlaybackRef.current = onPlayback; });

  // The stage: fetch the retargeted rotations and hand them to the shared player as the
  // speaking clip. Idle until the user presses play.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let cancelled = false;
    let stage: AvatarStage | null = null;
    fetch(trackPath)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`no track at ${trackPath}`))))
      .then((track: MotionTrack) => {
        if (cancelled) return;
        trackRef.current = track;
        setDurationS((d) => d ?? track.frames / track.fps);
        stage = new AvatarStage({
          canvas, ground: true, track,
          onState: (s) => setStatus(s),
        });
        stageRef.current = stage;
        stage.enter(AVATAR);
      })
      .catch(() => { if (!cancelled) setStatus("failed"); });
    return () => { cancelled = true; stageRef.current = null; stage?.loseContext(); };
  }, [trackPath]);

  // The audio: one element per profile. A segment ends when its endS is reached; the take
  // loops, so it never ends on its own.
  useEffect(() => {
    const audio = new Audio(audioPath);
    audio.preload = "metadata";
    audioRef.current = audio;
    const onMeta = () => setDurationS(audio.duration);
    const stopSegment = () => {
      if (playbackRef.current?.kind === "segment") onPlaybackRef.current(null);
    };
    const onTime = () => {
      const pb = playbackRef.current;
      if (pb?.kind === "segment" && audio.currentTime >= pb.segment.endS) stopSegment();
    };
    audio.addEventListener("loadedmetadata", onMeta);
    audio.addEventListener("timeupdate", onTime);
    audio.addEventListener("ended", stopSegment);
    return () => {
      audio.removeEventListener("loadedmetadata", onMeta);
      audio.removeEventListener("timeupdate", onTime);
      audio.removeEventListener("ended", stopSegment);
      audio.pause();
      audioRef.current = null;
    };
  }, [audioPath]);
  useEffect(() => { setAudioEl(audioRef.current); }, [audioPath]);

  // The mouth: every frame the face follows the audio's loudness; with nothing playing the
  // level is 0 and the jaw closes (blinks keep running on time alone, as in the arena).
  useEffect(() => {
    const start = performance.now();
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const stage = stageRef.current;
      if (!stage) return;
      stage.setFace(speechFace(playbackRef.current ? level() : 0, (performance.now() - start) / 1000));
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [level]);

  // A measuring handle, like the arena's __arena, so lip-sync can be inspected instead of
  // guessed at from screenshots.
  useEffect(() => {
    if (process.env.NODE_ENV === "production") return;
    const w = window as unknown as { __profiles?: Record<string, unknown> };
    w.__profiles = w.__profiles ?? {};
    w.__profiles[trackPath] = { get stage() { return stageRef.current; }, level };
  }, [trackPath, level]);

  // Playback: the avatar and the audio start and stop together. Pausing the full take freezes
  // the mixer mid-gesture; anything else settling to null returns the avatar to idle.
  useEffect(() => {
    const stage = stageRef.current, audio = audioRef.current, track = trackRef.current;
    if (!stage || !audio) return;
    if (!playback) {
      audio.pause();
      if (pausedFullRef.current) stage.setPaused(true);
      else { stage.setPaused(false); stage.setMotion("idle"); }
      return;
    }
    stage.setPaused(false);
    if (playback.kind === "full") {
      if (track) stage.setTrack(track);
      audio.loop = true;
      if (!pausedFullRef.current) audio.currentTime = 0;
    } else {
      const s = playback.segment;
      // A single pick has no seam to blend, so assembleTrack is a plain frame-range slice.
      if (track) stage.setTrack(assembleTrack(track, [{ startFrame: s.startFrame, endFrame: s.endFrame }]));
      audio.loop = false;
      audio.currentTime = s.startS;
    }
    pausedFullRef.current = false;
    stage.setMotion("speaking");
    void audio.play().catch(() => {});
  }, [playback]);

  const toggleFull = () => {
    if (playback?.kind === "full") { pausedFullRef.current = true; onPlayback(null); }
    else { pausedFullRef.current = false; onPlayback({ kind: "full" }); }
  };

  const playingFull = playback?.kind === "full";
  return (
    <div className="relative rounded-xl overflow-hidden border border-gray-200 bg-[#0d1117]" style={{ aspectRatio: "3 / 4" }}>
      <canvas ref={canvasRef} className="w-full h-full block" />
      {status !== "ready" && (
        <div className="absolute inset-0 grid place-items-center text-center text-sm text-white/70 px-6">
          {status === "loading" ? "loading motion track…" : "the track failed to load"}
        </div>
      )}
      {status === "ready" && (
        <div className="absolute bottom-2 left-2 flex items-center gap-2">
          <button onClick={toggleFull} title={playingFull ? "Pause" : "Play with audio"}
            className="w-8 h-8 rounded-full bg-white/15 hover:bg-white/25 grid place-items-center text-sm text-white">
            {playingFull ? "⏸" : "▶"}
          </button>
          {durationS != null && <span className="text-xs text-white/70 tabular-nums">{timecode(durationS)}</span>}
        </div>
      )}
    </div>
  );
}
