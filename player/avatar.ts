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
 * needs, and the mouth, which is lip-sync rather than movement. A bad keyframe in the asset
 * (2026-09-20, the rig visibly seizing) was fixed once in the asset itself, by
 * `scripts/clean-avatar-animations.mjs` — not patched around here on every playback; see that
 * script for what was wrong and how it is corrected.
 */
import * as THREE from 'three';
import { groundOffset } from './ground';

export type AvatarState = 'loading' | 'ready' | 'failed';

/** What the body is doing. Each names the motion-capture clips it may use, first found wins. */
export type MotionState = 'idle' | 'speaking' | 'affirmative' | 'negative' | 'greeting';

/** The one material the MetaPerson export gives the clothes; its texture is a flat garment atlas. */
export const OUTFIT_MATERIAL = 'outfit';
/**
 * The chest of the shirt's front panel in that atlas, measured from the mesh on 2026-09-25: the
 * front-facing vertices between 1.12 and 1.42 m up and within 14 cm of the centerline map here.
 * The pocket is a separate island drawn over it, so a print sits partly behind the pocket.
 */
export const PRINT_AREA = { u: [0.077, 0.278], v: [0.136, 0.316] } as const;

export const CLIP_FOR: Record<MotionState, string[]> = {
  idle: ['Breathing Idle', 'Idle'],
  // Speaking uses every listed clip the asset has, played one after another instead of
  // looping one capture: a single clip on LoopRepeat visibly restarts every few seconds, and
  // even with a seamless loop the repetition reads as mechanical. "Telling A Secret" is a
  // conspiratorial hunch with the weight on one foot; it read as a floating body and is wrong
  // for answering a visitor, so it is not offered here.
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
const TOES = ['LeftToeBase', 'RightToeBase'];

/** States that play their clip once and then settle back to idle, instead of looping it. */
const ONE_SHOT: ReadonlySet<MotionState> = new Set(['affirmative', 'negative', 'greeting']);

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
  return new THREE.AnimationClip(name, frames / fps, curves);
}

export class AvatarStage {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(32, 1, .1, 40);
  state: AvatarState | 'idle' = 'idle';
  clips = 0;
  motion: MotionState | null = null;
  face: Record<string, number> = {};
  root: THREE.Group | null = null;
  /** What is printed on the shirt, kept so it survives a load that finishes after it was asked for. */
  private print: CanvasImageSource | null = null;
  private outfit: { material: THREE.MeshStandardMaterial; base: THREE.Texture; printed: THREE.CanvasTexture | null } | null = null;
  private renderer: THREE.WebGLRenderer;
  private ownsRenderer: boolean;
  private toes: THREE.Object3D[] = [];
  private foot = new THREE.Vector3();
  private time = 0;
  private mixer: THREE.AnimationMixer | null = null;
  private actions = new Map<MotionState, THREE.AnimationAction>();
  private action: THREE.AnimationAction | null = null;
  /** The action built from `options.track` or the last `setTrack` call, for uncaching on swap. */
  private generated: THREE.AnimationAction | null = null;
  /** Every speaking clip the asset has, played in turn (empty when speaking is pinned or generated). */
  private speaking: THREE.AnimationAction[] = [];
  private speakIndex = 0;
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
    this.camera.position.set(0, 1.5, 3.46);
    this.camera.lookAt(0, 0.89, 0);
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
    // Same-state calls arrive per audio chunk while a line is still playing; restarting the
    // gesture on each one would stutter, so only an actual state change touches playback.
    if (next === this.motion) return;
    this.motion = next;
    // A generated track installed by setTrack outranks the asset's speaking clips; without one
    // they play in turn, and a state with neither falls back to whatever clip it has.
    this.play(next === 'speaking' ? this.generated ?? this.speaking[this.speakIndex] ?? this.actions.get(next) : this.actions.get(next));
  }

  /**
   * Swap the generated speaking track — e.g. the profiles page cuts the full profile track to
   * one sentence's frame range, then puts the full track back. The swap is a hard cut rather
   * than a crossfade: the point is to jump to the sentence's first frame.
   */
  setTrack(track: MotionTrack | null) {
    const prev = this.generated;
    if (prev) {
      prev.stop();
      this.mixer?.uncacheClip(prev.getClip());
      this.actions.delete('speaking');
      this.generated = null;
      if (this.action === prev) this.action = null;
    }
    if (track && this.mixer) {
      const action = this.mixer.clipAction(trackToClip(track)).setLoop(THREE.LoopRepeat, Infinity);
      this.actions.set('speaking', action);
      this.generated = action;
      if (this.motion === 'speaking') this.play(action);
    }
  }

  /** Freeze or resume the animation clock, so a preview can pause in step with its audio. */
  setPaused(paused: boolean) {
    if (paused) this.stop();
    else if (this.state === 'ready' && !this.lost && !this.options.reduced?.()) this.start();
  }

  /**
   * The generated speaking track's clock, in seconds — for a consumer that drives sync
   * itself. The eval page (/profiles/[id]) locks the avatar to a video's currentTime: the
   * video is the master clock and this is the follower. Null when there is no track.
   */
  get trackTime(): number | null {
    return this.generated ? this.generated.time : null;
  }

  /**
   * Jump the generated track to `seconds` (wrapped into its loop) and repaint, even while
   * paused — a scrub while stopped must still show the frame it lands on.
   */
  seekTrack(seconds: number) {
    const action = this.generated;
    if (!action) return;
    const duration = action.getClip().duration;
    action.time = ((seconds % duration) + duration) % duration;
    if (!this.raf) this.still();
  }

  /** Slow the track's clock for frame-by-frame inspection (the eval page's 0.5x mode). */
  setTrackRate(rate: number) {
    if (this.generated) this.generated.timeScale = rate;
  }

  private play(action: THREE.AnimationAction | undefined) {
    if (!action || action === this.action) return;
    action.reset(); action.play();
    if (this.action) this.action.crossFadeTo(action, FADE, false);
    this.action = action;
    if (this.options.reduced?.()) this.still();
  }

  /**
   * A one-shot clip ended. Speaking moves on to the next speaking clip (cycling, so an answer
   * of any length keeps moving without repeating one capture); anything else settles to idle.
   */
  private onClipFinished(action: THREE.AnimationAction) {
    if (action !== this.action) return;
    if (this.motion === 'speaking' && this.speaking.length > 1) {
      this.speakIndex = (this.speakIndex + 1) % this.speaking.length;
      this.play(this.speaking[this.speakIndex]);
    } else if (this.motion && this.motion !== 'idle') {
      this.motion = 'idle';
      this.play(this.actions.get('idle'));
    }
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

  /**
   * Prints an image on the chest of the shirt, or takes it off with null. The source is fitted
   * into PRINT_AREA of the outfit's own texture (a flat garment atlas, so nothing is warped) and
   * composited over it once, on a canvas; the material then samples that canvas instead.
   */
  setPrint(source: CanvasImageSource | null) {
    this.print = source;
    this.applyPrint();
  }
  private applyPrint() {
    const outfit = this.outfit;
    if (!outfit) return;
    const { material, base } = outfit;
    outfit.printed?.dispose(); outfit.printed = null;
    material.map = base;
    const image = base.image as CanvasImageSource & { width?: number; height?: number } | undefined;
    if (this.print && image?.width && image?.height) {
      const canvas = document.createElement('canvas');
      canvas.width = image.width; canvas.height = image.height;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(image, 0, 0);
        const area = { x: PRINT_AREA.u[0] * canvas.width, y: PRINT_AREA.v[0] * canvas.height, w: (PRINT_AREA.u[1] - PRINT_AREA.u[0]) * canvas.width, h: (PRINT_AREA.v[1] - PRINT_AREA.v[0]) * canvas.height };
        const src = this.print as { width?: number; height?: number };
        const sw = Number(src.width) || area.w, sh = Number(src.height) || area.h;
        const scale = Math.min(area.w / sw, area.h / sh);
        const w = sw * scale, h = sh * scale;
        ctx.drawImage(this.print, area.x + (area.w - w) / 2, area.y + (area.h - h) / 2, w, h);
        const printed = new THREE.CanvasTexture(canvas);
        printed.flipY = base.flipY; printed.colorSpace = base.colorSpace;
        printed.wrapS = base.wrapS; printed.wrapT = base.wrapT; printed.anisotropy = base.anisotropy;
        outfit.printed = printed;
        material.map = printed;
      }
    }
    material.needsUpdate = true;
    if (this.options.reduced?.()) this.render();
  }

  private still() { this.mixer?.update(0); this.ground(); this.applyFace(); this.render(); }

  /**
   * Put the lower foot on the floor. The clips were retargeted with the hips' translation
   * locked at its bind value, so any leg flexion lifts both feet and an asymmetric stance lifts
   * one more than the other. Measured on the Talking clip: soles 3.7 cm and 4.9 cm up.
   */
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
        const material = mesh.isMesh ? mesh.material as THREE.MeshStandardMaterial : null;
        if (material?.name === OUTFIT_MATERIAL && material.map && !this.outfit) this.outfit = { material, base: material.map, printed: null };
      });
      this.applyPrint();
      this.options.onClips?.(gltf.animations.map(a => a.name));
      if (gltf.animations.length || this.options.track) {
        this.mixer = new THREE.AnimationMixer(this.root);
        const find = (names: string[]) => names.map(name => gltf!.animations.find(c => c.name === name)).find(Boolean);
        const map = { ...CLIP_FOR, ...this.options.clipFor };
        for (const [state, names] of Object.entries(map) as [MotionState, string[]][]) {
          // Speaking plays each listed clip once, in turn, so a long answer doesn't restate the
          // same capture on a loop. A pinned or generated speaking clip (the arena) keeps
          // looping instead: a round is judged against its own audio, which can be any length.
          if (state === 'speaking' && !this.options.clipFor?.speaking && !this.options.track) {
            const clips = names.map(name => gltf!.animations.find(c => c.name === name)).filter(Boolean) as THREE.AnimationClip[];
            if (clips.length > 1) {
              this.speaking = clips.map(clip => {
                const a = this.mixer!.clipAction(clip).setLoop(THREE.LoopOnce, 1);
                a.clampWhenFinished = true;
                return a;
              });
              continue;
            }
          }
          const clip = find(names);
          if (!clip) continue;
          const action = this.mixer.clipAction(clip);
          if (ONE_SHOT.has(state)) { action.setLoop(THREE.LoopOnce, 1); action.clampWhenFinished = true; }
          else action.setLoop(THREE.LoopRepeat, Infinity);
          this.actions.set(state, action);
        }
        this.mixer.addEventListener('finished', (e) => this.onClipFinished(e.action));
        this.clips = gltf.animations.length;
      }
      if (this.options.track) {
        // Generated motion becomes an ordinary AnimationClip, so it crossfades, loops and
        // shares the mixer with the captures instead of needing a second playback path.
        const clip = trackToClip(this.options.track);
        const action = this.mixer!.clipAction(clip).setLoop(THREE.LoopRepeat, Infinity);
        this.actions.set('speaking', action);
        this.generated = action;
      }
      if (this.mixer) this.play(this.actions.get('idle') ?? this.speaking[this.speakIndex] ?? this.actions.get('speaking'));
      for (const n of TOES) { const b = this.root.getObjectByName(n); if (b) this.toes.push(b); }
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
