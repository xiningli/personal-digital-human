"use client";

// One candidate on the avatar: the policy's clip underneath (if any), the procedural rules
// on top, the mouth driven by the shared audio's loudness. Two of these side by side, fed by
// the same <audio>, are a round.

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { BEAT_S, EASE_S, HEAD_BONES, POSE_BONES, gestures, pose, speechFace, wantsBeat, type Beat, type Pose, type Triple } from "@/lib/pose";
import type { Mood, MotionPolicy } from "@/lib/types";

const ARM_DOWN = 1.25;
const FADE = 0.35;

interface Props {
  policy: MotionPolicy;
  avatar: string;
  /** Speech loudness 0..1 right now; 0 when nothing plays. */
  level: () => number;
  /** Whether the shared audio is playing: speaking mood, else idle. */
  playing: boolean;
  seed: number;
  label: string;
  onClips?: (names: string[]) => void;
}

function relax(model: THREE.Object3D) {
  const set = (name: string, x: number) => { const b = model.getObjectByName(name); if (b) b.rotation.x = x; };
  set("LeftArm", ARM_DOWN); set("RightArm", ARM_DOWN); set("LeftForeArm", 0.22); set("RightForeArm", 0.22);
}

export default function MotionStage({ policy, avatar, level, playing, seed, label, onClips }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "failed">("loading");
  const playingRef = useRef(playing);
  const policyRef = useRef(policy);
  // Refs are written in effects, not during render, so the animation loop reads the latest
  // props without rebuilding the rig on every change.
  useEffect(() => { playingRef.current = playing; }, [playing]);
  useEffect(() => { policyRef.current = policy; }, [policy]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let disposed = false;
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 40);
    camera.position.set(0, 1.34, 4.35);
    camera.lookAt(0, 1.02, 0);
    scene.add(new THREE.AmbientLight(0xd5f0ff, 0.85));
    const key = new THREE.DirectionalLight(0xd6f5ff, 1.5); key.position.set(-2, 4, 5); scene.add(key);
    const warm = new THREE.DirectionalLight(0xd7c2a0, 1.1); warm.position.set(3, 1, 2); scene.add(warm);
    const rim = new THREE.DirectionalLight(0x9fd8e8, 1.4); rim.position.set(0, 2.5, -4); scene.add(rim);

    const morphs: { influences: number[]; index: Record<string, number> }[] = [];
    const bones = new Map<keyof Pose, { bone: THREE.Object3D; rest: Triple; offset: Triple }>();
    let mixer: THREE.AnimationMixer | null = null;
    let action: THREE.AnimationAction | null = null;
    let idleAction: THREE.AnimationAction | null = null;
    let clipAction: THREE.AnimationAction | null = null;
    let mood: Mood = "idle";
    let time = seed % 1000;
    let last = 0;
    let lastBeat = -Infinity;
    let nextArm: "left" | "right" = "right";
    let beats: Beat[] = [];
    let raf = 0;
    const clock = new THREE.Clock();

    const play = (next: THREE.AnimationAction | null) => {
      if (!next || next === action) return;
      next.reset(); next.play();
      if (action) action.crossFadeTo(next, FADE, false);
      action = next;
    };
    const resize = () => {
      const { width, height } = canvas.getBoundingClientRect();
      if (!width || !height) return;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    new GLTFLoader().loadAsync(avatar).then((gltf) => {
      if (disposed) return;
      const root = gltf.scene;
      root.traverse((node) => {
        const mesh = node as THREE.Mesh;
        if (mesh.isMesh && mesh.morphTargetDictionary && mesh.morphTargetInfluences) morphs.push({ influences: mesh.morphTargetInfluences, index: mesh.morphTargetDictionary });
      });
      onClips?.(gltf.animations.map((a) => a.name));
      const p = policyRef.current;
      if (gltf.animations.length && p.clip !== null) {
        mixer = new THREE.AnimationMixer(root);
        const find = (names: string[]) => names.map((n) => gltf.animations.find((a) => a.name === n)).find(Boolean);
        const idle = find(["Breathing Idle", "Idle"]);
        const clip = find([p.clip]);
        if (idle) idleAction = mixer.clipAction(idle).setLoop(THREE.LoopRepeat, Infinity);
        if (clip) clipAction = mixer.clipAction(clip).setLoop(THREE.LoopRepeat, Infinity);
        play(idleAction ?? clipAction);
      } else {
        relax(root);
      }
      for (const name of POSE_BONES) {
        const bone = root.getObjectByName(name);
        if (bone) bones.set(name, { bone, rest: [bone.rotation.x, bone.rotation.y, bone.rotation.z], offset: [0, 0, 0] });
      }
      scene.add(root);
      setStatus("ready");
      resize();
      const tick = () => {
        raf = requestAnimationFrame(tick);
        const delta = Math.min(0.1, clock.getDelta());
        time += delta;
        const p = policyRef.current;
        const want: Mood = playingRef.current ? "speaking" : "idle";
        if (want !== mood) {
          mood = want;
          if (mixer) play(mood === "speaking" ? clipAction ?? idleAction : idleAction ?? clipAction);
        }
        const lv = mood === "speaking" ? level() : 0;
        if (wantsBeat(lv, last, time - lastBeat, p)) { beats.push({ arm: nextArm, at: time }); nextArm = nextArm === "left" ? "right" : "left"; lastBeat = time; }
        last = lv;
        if (mixer) mixer.update(delta);
        else { root.position.y = Math.sin(time * 0.9) * 0.006; root.rotation.y = Math.sin(time * 0.31) * 0.035; }
        beats = beats.filter((b) => time - b.at <= BEAT_S);
        const target = pose(mood, time, lv, p);
        const extra = gestures(time, Infinity, beats, p);
        const k = 1 - Math.exp(-delta / (EASE_S[mood] * p.ease));
        for (const name of mixer ? HEAD_BONES : POSE_BONES) {
          const entry = bones.get(name);
          if (!entry) continue;
          const wantR = target[name];
          const g = extra[name] ?? [0, 0, 0];
          for (let i = 0; i < 3; i++) entry.offset[i] += (wantR[i] - entry.offset[i]) * k;
          const base = mixer ? entry.bone.rotation : { x: entry.rest[0], y: entry.rest[1], z: entry.rest[2] };
          entry.bone.rotation.set(base.x + entry.offset[0] + g[0], base.y + entry.offset[1] + g[1], base.z + entry.offset[2] + g[2]);
        }
        const face = speechFace(lv, time);
        for (const s of morphs) for (const [name, value] of Object.entries(face)) { const i = s.index[name]; if (i !== undefined) s.influences[i] = value; }
        renderer.render(scene, camera);
      };
      raf = requestAnimationFrame(tick);
    }).catch(() => { if (!disposed) setStatus("failed"); });

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      observer.disconnect();
      renderer.dispose();
    };
    // The stage is rebuilt when the policy or the avatar changes: a candidate is one rig setup.
  }, [policy.id, avatar, seed]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="relative rounded-xl overflow-hidden border border-gray-200 bg-[#0d1117]" style={{ aspectRatio: "3 / 4" }}>
      <canvas ref={canvasRef} className="w-full h-full block" />
      <div className="absolute top-2 left-3 text-white/90 font-mono text-lg font-semibold">{label}</div>
      {status !== "ready" && (
        <div className="absolute inset-0 grid place-items-center text-sm text-white/70">
          {status === "loading" ? "loading avatar…" : "avatar failed to load (bash scripts/import-avatar.sh)"}
        </div>
      )}
    </div>
  );
}
