/**
 * The digital human's player: one implementation, two consumers.
 *
 * This repository owns the digital human the way personal-voice-clone-studio owns the voice.
 * The motion arena here ranks what the body does; `personal-site` shows the result. Before
 * 2026-09-19 each of them carried its own copy of this file, drifted apart, and a fix to one
 * had to be mirrored by hand into the other. Now there is one copy and `personal-site`
 * imports it from this sibling checkout.
 *
 * The stage owns a scene, a camera and lights, and either borrows a renderer (personal-site
 * shares one with the mirror it draws the world into) or makes its own (the arena gives each
 * candidate a canvas). It plays motion capture baked into the asset; generated tracks from a
 * model are the next source and land here, not in either consumer.
 *
 * Nothing here poses the body by rule. The hand-written body language was deleted on
 * 2026-09-19; what survived is foot grounding, which is a correctness fix every motion source
 * needs, and the mouth, which is lip-sync rather than movement.
 */
import * as THREE from 'three';
import { groundOffset } from './ground';

export type AvatarState = 'loading' | 'ready' | 'failed';

/** What the body is doing. Each names the motion-capture clips it may use, first found wins. */
export type MotionState = 'idle' | 'speaking' | 'affirmative' | 'negative' | 'greeting';

export const CLIP_FOR: Record<MotionState, string[]> = {
  idle: ['Breathing Idle', 'Idle'],
  // "Telling A Secret" is a conspiratorial hunch with the weight on one foot; it read as a
  // floating body and is wrong for answering a visitor, so it is not offered here.
  speaking: ['Talking', 'Lengthy Head Nod'],
  affirmative: ['Head Nod Yes', 'Thoughtful Head Nod', 'Lengthy Head Nod'],
  negative: ['Shrugging'],
  greeting: ['Waving'],
};

/**
 * Motion produced by a model and retargeted onto this rig: local rotations per bone per
 * frame, xyzw, flattened frame-major. `motion/retarget.py` writes it.
 */
export interface MotionTrack {
  fps: number;
  frames: number;
  bones: string[];
  quats: number[];
}

export interface AvatarOptions {
  canvas: HTMLCanvasElement;
  /** Borrow a renderer, or leave it out and the stage makes one for this canvas. */
  renderer?: THREE.WebGLRenderer;
  reduced?: () => boolean;
  onState?: (state: AvatarState) => void;
  onClips?: (names: string[]) => void;
  /** Draw a shadow-catching floor. Off where the avatar already stands in a scene with one. */
  ground?: boolean;
  /** Pin a state to particular clips, so the arena can play one named capture per candidate. */
  clipFor?: Partial<Record<MotionState, string[]>>;
  /** Generated motion to speak with, instead of a capture. */
  track?: MotionTrack;
}

interface MorphSurface { influences: number[]; index: Record<string, number> }

const FADE = .35;
// ~172 deg/s: slower than a real fast head-shake can peak at, so a legitimately quick gesture
// is softened rather than left snapping; found by measurement (a first pass at 25 rad/s, four
// times looser, still let the "Talking" clip's bad keyframe through: 0.24 rad max over 3 s of
// steady-state playback at 3, against up to 2.94 rad before either fix existed).
const MAX_RAD_PER_S = 3;
const TOES = ['LeftToeBase', 'RightToeBase'];

/**
 * A quaternion and its negation represent the same rotation, but SLERP is not sign-invariant:
 * interpolating between two keyframes that landed on opposite signs of that double cover spins
 * the bone the long way around in one frame, then snaps back the next. Found on the asset's
 * own "Talking" clip (measured: the head bone turned 2.38 rad, 136 degrees, in a single 60 fps
 * frame, twice a second) and is exactly what reads as the rig seizing. Neither the retargeting
 * script nor whatever authored the mocap enforced sign continuity between keyframes, and
 * three.js's QuaternionKeyframeTrack does not do it either, so every clip is walked once here,
 * right after it is built or loaded, flipping a keyframe's sign whenever it opposes the one
 * before it.
 */
function fixQuaternionContinuity(clip: THREE.AnimationClip): void {
  for (const track of clip.tracks) {
    if (!(track instanceof THREE.QuaternionKeyframeTrack)) continue;
    const v = track.values;
    for (let f = 1; f * 4 < v.length; f++) {
      const p = (f - 1) * 4, c = f * 4;
      const dot = v[p] * v[c] + v[p + 1] * v[c + 1] + v[p + 2] * v[c + 2] + v[p + 3] * v[c + 3];
      if (dot < 0) { v[c] = -v[c]; v[c + 1] = -v[c + 1]; v[c + 2] = -v[c + 2]; v[c + 3] = -v[c + 3]; }
    }
  }
}

/** Build a three.js clip from a retargeted track: one quaternion curve per bone. */
export function trackToClip(track: MotionTrack, name = 'generated'): THREE.AnimationClip {
  const { fps, frames, bones, quats } = track;
  const times = new Float32Array(frames);
  for (let f = 0; f < frames; f++) times[f] = f / fps;
  const curves: THREE.KeyframeTrack[] = [];
  for (let b = 0; b < bones.length; b++) {
    const values = new Float32Array(frames * 4);
    for (let f = 0; f < frames; f++) {
      const src = (f * bones.length + b) * 4;
      values.set([quats[src], quats[src + 1], quats[src + 2], quats[src + 3]], f * 4);
    }
    curves.push(new THREE.QuaternionKeyframeTrack(`${bones[b]}.quaternion`, times, values));
  }
  const clip = new THREE.AnimationClip(name, frames / fps, curves);
  fixQuaternionContinuity(clip);
  return clip;
}

export class AvatarStage {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(32, 1, .1, 40);
  state: AvatarState | 'idle' = 'idle';
  clips = 0;
  motion: MotionState | null = null;
  face: Record<string, number> = {};
  root: THREE.Group | null = null;
  private renderer: THREE.WebGLRenderer;
  private ownsRenderer: boolean;
  private toes: THREE.Object3D[] = [];
  private foot = new THREE.Vector3();
  private time = 0;
  // A safety net behind fixQuaternionContinuity: a bad keyframe (not a sign flip, a
  // genuinely-too-large delta between two adjacent keyframes — measured on the asset's own
  // "Talking" clip: 173 degrees in under 0.08 s, physically impossible for a real head) still
  // reads as the rig seizing. Capping the angle any bone is allowed to turn per second, however
  // it got that pose, is cheap and source-agnostic: it never touches a clip that behaves.
  private bones: THREE.Object3D[] = [];
  private prevQuat = new Map<THREE.Object3D, THREE.Quaternion>();
  private mixer: THREE.AnimationMixer | null = null;
  private actions = new Map<MotionState, THREE.AnimationAction>();
  private action: THREE.AnimationAction | null = null;
  private morphs: MorphSurface[] = [];
  private clock = new THREE.Clock();
  private raf = 0;
  private loading: Promise<void> | null = null;
  private lost = false;
  private observer: ResizeObserver;

  constructor(private options: AvatarOptions) {
    this.ownsRenderer = !options.renderer;
    this.renderer = options.renderer ?? new THREE.WebGLRenderer({ canvas: options.canvas, antialias: true, alpha: true });
    if (this.ownsRenderer) this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    this.scene.add(new THREE.AmbientLight(0xd5f0ff, .85));
    const key = new THREE.DirectionalLight(0xd6f5ff, 1.5); key.position.set(-2, 4, 5); this.scene.add(key);
    const warm = new THREE.DirectionalLight(0xd7c2a0, 1.1); warm.position.set(3, 1, 2); this.scene.add(warm);
    const rim = new THREE.DirectionalLight(0x9fd8e8, 1.4); rim.position.set(0, 2.5, -4); this.scene.add(rim);
    if (options.ground) this.addGroundShadow();
    this.camera.position.set(0, 1.34, 4.35);
    this.camera.lookAt(0, 1.02, 0);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(options.canvas);
  }

  /**
   * A shadow-only floor lit from almost overhead. The key light sits at (-2, 4, 5), so the
   * shadow it throws of the hips lands 0.58 to the right and 1.45 behind the feet: correct,
   * and useless as a contact cue, which is half of why the figure reads as floating.
   */
  private addGroundShadow() {
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    const contact = new THREE.DirectionalLight(0xffffff, .18);
    contact.position.set(.25, 6, .6);
    contact.castShadow = true;
    contact.shadow.mapSize.set(1024, 1024);
    contact.shadow.camera.top = 2; contact.shadow.camera.bottom = -2;
    contact.shadow.camera.left = -1.5; contact.shadow.camera.right = 1.5;
    contact.shadow.camera.near = 1; contact.shadow.camera.far = 9;
    contact.shadow.bias = -.002;
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(24, 24), new THREE.ShadowMaterial({ opacity: .5 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(contact, contact.target, floor);
  }

  /** Show the avatar; `urls` are tried in order, so a richer asset can sit in front of a plainer one. */
  enter(...urls: string[]) {
    this.resize();
    if (this.root) this.start();
    else if (!this.loading) this.loading = this.load(urls);
  }
  leave() { this.stop(); }
  loseContext() {
    this.lost = true; this.stop(); this.observer.disconnect();
    if (this.ownsRenderer) this.renderer.dispose();
  }

  setMotion(next: MotionState) {
    this.motion = next;
    this.play(this.actions.get(next));
  }

  private play(action: THREE.AnimationAction | undefined) {
    if (!action || action === this.action) return;
    action.reset(); action.play();
    if (this.action) this.action.crossFadeTo(action, FADE, false);
    this.action = action;
    if (this.options.reduced?.()) this.still();
  }

  setFace(targets: Record<string, number>) {
    this.face = { ...this.face, ...targets };
    if (this.options.reduced?.()) this.still();
  }
  clearFace() {
    this.face = {};
    for (const surface of this.morphs) for (const name of Object.keys(surface.index)) surface.influences[surface.index[name]] = 0;
    if (this.options.reduced?.()) this.render();
  }

  private still() { this.mixer?.update(0); this.ground(); this.applyFace(); this.render(); }

  /**
   * Put the lower foot on the floor. The clips were retargeted with the hips' translation
   * locked at its bind value, so any leg flexion lifts both feet and an asymmetric stance lifts
   * one more than the other. Measured on the Talking clip: soles 3.7 cm and 4.9 cm up.
   */
  /**
   * The safety net described where MAX_RAD_PER_S is declared: after the mixer has posed every
   * bone for this frame, pull back any bone that turned faster than that from wherever it
   * landed towards where it actually was last frame, so a bad keyframe reads as a very fast
   * beat instead of a snap. Applies whatever produced the pose — a capture, a generated track,
   * or a crossfade between two of either.
   */
  private clampRotationSpeed(delta: number) {
    if (delta <= 0) return;
    const maxAngle = MAX_RAD_PER_S * delta;
    for (const bone of this.bones) {
      const prev = this.prevQuat.get(bone);
      if (prev) {
        const angle = prev.angleTo(bone.quaternion);
        if (angle > maxAngle) bone.quaternion.copy(prev.slerp(bone.quaternion, maxAngle / angle));
        prev.copy(bone.quaternion);
      } else {
        this.prevQuat.set(bone, bone.quaternion.clone());
      }
    }
  }

  private ground() {
    if (!this.root || !this.toes.length) return;
    this.root.position.y = 0;
    this.root.updateWorldMatrix(true, true);
    this.root.position.y = groundOffset(this.toes.map(t => { t.getWorldPosition(this.foot); return this.foot.y; }));
  }

  private async load(urls: string[]) {
    this.state = 'loading'; this.options.onState?.('loading');
    try {
      const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
      const loader = new GLTFLoader();
      let gltf: Awaited<ReturnType<typeof loader.loadAsync>> | null = null;
      for (const [i, url] of urls.entries()) {
        try { gltf = await loader.loadAsync(url); break; } catch (error) { if (i === urls.length - 1) throw error; }
      }
      if (!gltf) throw new Error('no avatar asset');
      if (this.lost) return;
      this.root = gltf.scene;
      this.root.traverse(node => {
        const mesh = node as THREE.Mesh;
        if (mesh.isMesh && this.options.ground) mesh.castShadow = true;
        if (mesh.isMesh && mesh.morphTargetDictionary && mesh.morphTargetInfluences) {
          this.morphs.push({ influences: mesh.morphTargetInfluences, index: mesh.morphTargetDictionary });
        }
      });
      this.options.onClips?.(gltf.animations.map(a => a.name));
      for (const clip of gltf.animations) fixQuaternionContinuity(clip);
      if (gltf.animations.length || this.options.track) {
        this.mixer = new THREE.AnimationMixer(this.root);
        const find = (names: string[]) => names.map(name => gltf!.animations.find(c => c.name === name)).find(Boolean);
        const map = { ...CLIP_FOR, ...this.options.clipFor };
        for (const [state, names] of Object.entries(map) as [MotionState, string[]][]) {
          const clip = find(names);
          if (clip) this.actions.set(state, this.mixer.clipAction(clip).setLoop(THREE.LoopRepeat, Infinity));
        }
        this.clips = gltf.animations.length;
      }
      if (this.options.track) {
        // Generated motion becomes an ordinary AnimationClip, so it crossfades, loops and
        // shares the mixer with the captures instead of needing a second playback path.
        const clip = trackToClip(this.options.track);
        this.actions.set('speaking', this.mixer!.clipAction(clip).setLoop(THREE.LoopRepeat, Infinity));
      }
      if (this.mixer) this.play(this.actions.get('idle') ?? this.actions.get('speaking'));
      for (const n of TOES) { const b = this.root.getObjectByName(n); if (b) this.toes.push(b); }
      this.root.traverse(node => { if ((node as THREE.Bone).isBone) this.bones.push(node); });
      // Pose once before the first render, so clampRotationSpeed's baseline is the loaded
      // character's own first animated pose rather than nothing: without this, the very first
      // frame is unclamped by construction (there is no "last frame" to compare against yet),
      // which is exactly where a transition into a bad clip would otherwise still get through.
      this.mixer?.update(0);
      for (const bone of this.bones) this.prevQuat.set(bone, bone.quaternion.clone());
      this.scene.add(this.root);
      this.state = 'ready'; this.options.onState?.('ready');
      this.start();
    } catch {
      this.loading = null;
      this.state = 'failed'; this.options.onState?.('failed');
    }
  }

  private start() {
    this.stop();
    this.clock.getDelta();
    if (this.options.reduced?.()) { this.still(); return; }
    const tick = () => {
      this.raf = requestAnimationFrame(tick);
      const delta = Math.min(.1, this.clock.getDelta());
      this.mixer?.update(delta);
      this.clampRotationSpeed(delta);
      this.time += delta;
      this.ground();
      this.applyFace();
      this.render();
    };
    this.raf = requestAnimationFrame(tick);
  }
  private stop() { if (this.raf) cancelAnimationFrame(this.raf); this.raf = 0; }

  private applyFace() {
    for (const surface of this.morphs) {
      for (const [name, value] of Object.entries(this.face)) {
        const index = surface.index[name];
        if (index !== undefined) surface.influences[index] = value;
      }
    }
  }

  resize() {
    const { width, height } = this.options.canvas.getBoundingClientRect();
    if (!width || !height || this.lost) return;
    if (this.ownsRenderer) this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.render();
  }
  render() { if (!this.lost) this.renderer.render(this.scene, this.camera); }
}
