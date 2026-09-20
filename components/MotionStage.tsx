"use client";

// One candidate, drawn by the shared player in `player/`. This file is a React wrapper and
// nothing else: the avatar, the clip playback, foot grounding and the mouth all live in the
// player, which personal-site imports from this same checkout.

import { useEffect, useRef, useState } from "react";
import { AvatarStage, speechFace, type AvatarState } from "@/player";
import type { MotionPolicy } from "@/lib/types";

interface Props {
  policy: MotionPolicy;
  avatar: string;
  /** Speech loudness 0..1 right now; 0 when nothing plays. */
  level: () => number;
  /** Whether the shared audio is playing. */
  playing: boolean;
  seed: number;
  label: string;
  onClips?: (names: string[]) => void;
}

export default function MotionStage({ policy, avatar, level, playing, seed, label, onClips }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<AvatarStage | null>(null);
  const [loadState, setLoadState] = useState<AvatarState>("loading");
  // No generated track is wired in yet. Derived from the prop rather than stored, so the
  // effect never sets state synchronously.
  const unsupported = policy.source === "generated";
  const status = unsupported ? "unsupported" : loadState;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || unsupported) return;
    const stage = new AvatarStage({
      canvas, ground: true, onState: setLoadState, onClips,
      clipFor: policy.clip ? { speaking: [policy.clip] } : undefined,
    });
    stageRef.current = stage;
    stage.enter(avatar);
    let raf = 0;
    const start = performance.now() + (seed % 1000);
    const tick = () => { raf = requestAnimationFrame(tick); stage.setFace(speechFace(level(), (performance.now() - start) / 1000)); };
    raf = requestAnimationFrame(tick);
    return () => { cancelAnimationFrame(raf); stage.loseContext(); stageRef.current = null; };
  }, [policy.id, policy.clip, unsupported, avatar, seed]); // eslint-disable-line react-hooks/exhaustive-deps

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
          {status === "loading" ? "loading avatar…"
            : status === "unsupported" ? "generated motion is not wired in yet"
            : "avatar failed to load (bash scripts/import-avatar.sh)"}
        </div>
      )}
    </div>
  );
}
