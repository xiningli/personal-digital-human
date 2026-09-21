"use client";

// Dev-only harness: one avatar playing a generated/extracted MotionTrack as its speaking
// clip, exposed as window.__stage for frame-by-frame checks. ?track=<name> picks the JSON
// under /assets/; without it the track-less page still loads (mocap clips).

import { useEffect, useRef } from "react";
import { AvatarStage, type MotionTrack } from "@/player";

export default function PlayerCheck() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const name = new URLSearchParams(window.location.search).get("track");
    let stage: AvatarStage | null = null;
    const build = (track?: MotionTrack) => {
      if (!ref.current) return;
      stage = new AvatarStage({ canvas: ref.current, ground: true, track });
      (window as unknown as { __stage: AvatarStage }).__stage = stage;
      stage.enter("/assets/model-clips.glb");
    };
    if (name) {
      fetch(`/motion/${name}.track.json`).then((r) => r.json()).then((t) => build(t));
    } else build();
    return () => stage?.loseContext();
  }, []);
  return (
    <div style={{ width: 600, height: 800 }}>
      <canvas ref={ref} style={{ width: "100%", height: "100%" }} />
    </div>
  );
}
