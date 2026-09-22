"use client";

// The eval page's middle column (docs/protocol.md §5): the profile's SMPL-X motion drawn
// as a bare stick figure, sitting between the source video and the retargeted avatar so a
// low likeness score can be blamed on the right stage — video vs skeleton judges the
// extraction, skeleton vs avatar judges the presentation. A stripped-down AvatarStage
// (own renderer, ResizeObserver, identical camera framing) that renders one frame of a
// joints3d.json (motion/export_joints3d.py) on demand; it has no clock of its own, the
// parent's rAF calls setTime() off the video's master clock, so 0.5x, seek and the
// per-segment loop all follow for free.

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as THREE from "three";

/** What motion/export_joints3d.py writes: SMPL-X FK positions, y-up, stage-aligned. */
export interface Joints3D {
  fps: number;
  frames: number;
  names: string[];
  /** frames[f][j] = [x, y, z], flattened frame-major, 55 joints. */
  joints: number[];
}

export interface SkeletonViewHandle {
  /** Draw the frame at `seconds` (floor(t * fps), clamped). */
  setTime: (seconds: number) => void;
  /** The frame index last drawn; null before the first draw. */
  readonly frameIndex: number | null;
}

const JOINTS = 55;

// Bone links by SMPL-X joint index (names are in JOINT_NAMES in export_joints3d.py):
// the 22-joint body tree, plus a rough fan from each wrist to its five finger bases —
// GVHMR extracts no finger motion, so the fingers only mark where the hand is.
const BONES: [number, number][] = [
  [0, 1], [0, 2], [0, 3],           // pelvis -> hips, spine1
  [1, 4], [2, 5],                   // hips -> knees
  [3, 6], [6, 9],                   // spine chain
  [4, 7], [5, 8],                   // knees -> ankles
  [7, 10], [8, 11],                 // ankles -> feet
  [9, 12], [12, 15],                // spine3 -> neck -> head
  [9, 13], [9, 14],                 // spine3 -> collars
  [13, 16], [14, 17],               // collars -> shoulders
  [16, 18], [17, 19],               // shoulders -> elbows
  [18, 20], [19, 21],               // elbows -> wrists
  [15, 23], [15, 24],               // head -> eyes (a facing cue)
  [20, 25], [20, 28], [20, 31], [20, 34], [20, 37],   // left wrist fan
  [21, 40], [21, 43], [21, 46], [21, 49], [21, 52],   // right wrist fan
];

const SkeletonView = forwardRef<SkeletonViewHandle, { src: string }>(function SkeletonView({ src }, ref) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dataRef = useRef<Joints3D | null>(null);
  const frameRef = useRef<number | null>(null);
  const drawRef = useRef<(frame: number) => void>(() => {});
  const [status, setStatus] = useState<"loading" | "ready" | "failed">("loading");

  useImperativeHandle(ref, () => ({
    setTime(seconds: number) {
      const d = dataRef.current;
      if (!d || !Number.isFinite(seconds)) return;
      const frame = Math.min(Math.max(Math.floor(seconds * d.fps), 0), d.frames - 1);
      if (frame !== frameRef.current) drawRef.current(frame);
    },
    get frameIndex() { return frameRef.current; },
  }), []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let cancelled = false;
    let lost = false;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 40);
    // Identical framing to AvatarStage so the three columns compose the same shot.
    camera.position.set(0, 1.5, 3.46);
    camera.lookAt(0, 0.89, 0);

    const grid = new THREE.GridHelper(4, 8, 0x334155, 0x1e293b);
    scene.add(grid);

    const boneGeom = new THREE.BufferGeometry();
    boneGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(BONES.length * 2 * 3), 3));
    const bones = new THREE.LineSegments(boneGeom, new THREE.LineBasicMaterial({ color: 0x7dd3fc }));
    const jointGeom = new THREE.BufferGeometry();
    jointGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(JOINTS * 3), 3));
    const joints = new THREE.Points(jointGeom, new THREE.PointsMaterial({ color: 0xfbbf24, size: 5, sizeAttenuation: false }));
    scene.add(bones, joints);

    const draw = (frame: number) => {
      const d = dataRef.current;
      if (!d || lost) return;
      frameRef.current = frame;
      const base = frame * JOINTS * 3;
      const bp = boneGeom.attributes.position as THREE.BufferAttribute;
      for (let s = 0; s < BONES.length; s++) {
        for (let e = 0; e < 2; e++) {
          const j = base + BONES[s][e] * 3, v = (s * 2 + e) * 3;
          bp.array[v] = d.joints[j]; bp.array[v + 1] = d.joints[j + 1]; bp.array[v + 2] = d.joints[j + 2];
        }
      }
      bp.needsUpdate = true;
      const jp = jointGeom.attributes.position as THREE.BufferAttribute;
      (jp.array as Float32Array).set(d.joints.slice(base, base + JOINTS * 3));
      jp.needsUpdate = true;
      renderer.render(scene, camera);
    };
    drawRef.current = draw;

    const resize = () => {
      const { width, height } = canvas.getBoundingClientRect();
      if (!width || !height || lost) return;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      if (frameRef.current !== null) draw(frameRef.current);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    fetch(src, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d: Joints3D) => {
        if (cancelled) return;
        dataRef.current = d;
        setStatus("ready");
        resize();
        draw(0);
      })
      .catch(() => { if (!cancelled) setStatus("failed"); });

    return () => {
      cancelled = true;
      lost = true;
      observer.disconnect();
      boneGeom.dispose();
      jointGeom.dispose();
      renderer.dispose();
      dataRef.current = null;
      frameRef.current = null;
    };
  }, [src]);

  return (
    <>
      <canvas ref={canvasRef} className="w-full h-full block" />
      {status !== "ready" && (
        <div className="absolute inset-0 grid place-items-center text-center text-sm text-white/70 px-6">
          {status === "loading" ? "loading skeleton…" : "骨架数据未生成（joints3d.json 缺失）"}
        </div>
      )}
    </>
  );
});

export default SkeletonView;
