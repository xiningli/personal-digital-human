"use client";

// One candidate, drawn by the shared player in `player/`. This file is a React wrapper and
// nothing else: the avatar, the clip playback, foot grounding and the mouth all live in the
// player, which personal-site imports from this same checkout.

import { useEffect, useRef, useState } from "react";
import { AvatarStage, speechFace, type AvatarState, type MotionTrack } from "@/player";
import { profileTrackPath, trackPath } from "@/lib/policies";
import { assembleTrack, splitSentences, type SegmentMatch } from "@/lib/motion-segments";
import type { MotionPolicy } from "@/lib/types";

interface Props {
  policy: MotionPolicy;
  avatar: string;
  /** The round's audio, which a generated track is keyed to. */
  audioPath: string;
  /** Speech loudness 0..1 right now; 0 when nothing plays. */
  level: () => number;
  /** Whether the shared audio is playing. */
  playing: boolean;
  seed: number;
  label: string;
  /** What the digital human says this round; profile candidates gesture per sentence. */
  text?: string;
  onClips?: (names: string[]) => void;
}

/**
 * Assembled tracks, memoised by profile and text: two candidates of the same profile in one
 * round share one match call, and re-rendering a round never re-embeds the same text.
 */
const assembledCache = new Map<string, Promise<MotionTrack | null>>();

/**
 * Pick a gesture segment per sentence and splice them into one track. Returns null — the
 * caller then plays the whole track — when the profile has no segments or the match route
 * cannot help: per-sentence gestures are an upgrade, never a requirement.
 */
async function assembleForText(profileId: string, text: string, track: MotionTrack): Promise<MotionTrack | null> {
  const sentences = splitSentences(text);
  if (!sentences.length) return null;
  const segRes = await fetch(`/motion/profile-${profileId}.segments.json`);
  if (!segRes.ok) return null;
  const segFile = await segRes.json();
  if (!Array.isArray(segFile?.segments) || !segFile.segments.length) return null;
  const matchRes = await fetch(`/api/profiles/${profileId}/match`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ texts: sentences }),
  });
  if (!matchRes.ok) return null;
  const { matches } = (await matchRes.json()) as { matches: (SegmentMatch | null)[] };
  const picks = (matches ?? []).filter((m): m is SegmentMatch => !!m)
    .map((m) => ({ startFrame: m.startFrame, endFrame: m.endFrame }));
  if (!picks.length) return null;
  return assembleTrack(track, picks);
}

function assembledTrack(profileId: string, text: string, track: MotionTrack): Promise<MotionTrack | null> {
  const key = `${profileId}\n${text}`;
  let p = assembledCache.get(key);
  if (!p) {
    p = assembleForText(profileId, text, track).catch(() => null);
    assembledCache.set(key, p);
  }
  return p;
}

export default function MotionStage({ policy, avatar, audioPath, level, playing, seed, label, text, onClips }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<AvatarStage | null>(null);
  const [loadState, setLoadState] = useState<AvatarState>("loading");
  const status = loadState;
  const generated = policy.source === "generated";
  // Generated and profile candidates both play a recorded track; only where it lives differs.
  const url = generated ? trackPath(audioPath, policy.model ?? "")
    : policy.source === "profile" ? profileTrackPath(policy.profileId ?? "")
    : null;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let cancelled = false;
    let stage: AvatarStage | null = null;
    let raf = 0;
    // A generated or profile candidate needs its track before the rig can be built, because
    // the track becomes the speaking clip.
    const build = (track?: MotionTrack) => {
      if (cancelled || !canvas) return;
      stage = new AvatarStage({
        canvas, ground: true, onState: setLoadState, onClips, track,
        clipFor: policy.clip ? { speaking: [policy.clip] } : undefined,
      });
      stageRef.current = stage;
      stage.enter(avatar);
      const start = performance.now() + (seed % 1000);
      const tick = () => { raf = requestAnimationFrame(tick); stage?.setFace(speechFace(level(), (performance.now() - start) / 1000)); };
      raf = requestAnimationFrame(tick);
    };
    if (url) {
      fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`no track at ${url}`))))
        .then(async (t: MotionTrack) => {
          // A profile candidate with a round text swaps the whole-track loop for gestures
          // matched per sentence; any failure quietly keeps the whole track.
          if (policy.source === "profile" && policy.profileId && text?.trim()) {
            build((await assembledTrack(policy.profileId, text, t)) ?? t);
          } else {
            build(t);
          }
        })
        .catch(() => { if (!cancelled) setLoadState("failed"); });
    } else {
      build();
    }
    return () => { cancelled = true; cancelAnimationFrame(raf); stage?.loseContext(); stageRef.current = null; };
  }, [policy.id, policy.clip, policy.source, policy.profileId, url, avatar, seed, text]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { stageRef.current?.setMotion(playing ? "speaking" : "idle"); }, [playing]);

  // A measuring handle, so a round that looks wrong can be inspected instead of guessed at.
  useEffect(() => {
    if (process.env.NODE_ENV === "production") return;
    const w = window as unknown as { __arena?: Record<string, unknown> };
    w.__arena = w.__arena ?? {};
    w.__arena[label] = { get stage() { return stageRef.current; }, policy };
  }, [label, policy]);

  return (
    <div className="relative rounded-xl overflow-hidden border border-gray-200 bg-[#0d1117]" style={{ aspectRatio: "3 / 4" }}>
      <canvas ref={canvasRef} className="w-full h-full block" />
      <div className="absolute top-2 left-3 text-white/90 font-mono text-lg font-semibold">{label}</div>
      {status !== "ready" && (
        <div className="absolute inset-0 grid place-items-center text-center text-sm text-white/70 px-6">
          {status === "loading" ? (url ? "loading motion track…" : "loading avatar…")
            : generated ? "no generated track for this clip (motion/build_tracks.py)"
            : policy.source === "profile" ? "no track for this profile (is it still ready?)"
            : "avatar failed to load (bash scripts/import-avatar.sh)"}
        </div>
      )}
    </div>
  );
}
