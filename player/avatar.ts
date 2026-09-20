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
}

interface MorphSurface { influences: number[]; index: Record<string, number> }

const FADE = .35;
const TOES = ['LeftToeBase', 'RightToeBase'];

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
      if (gltf.animations.length) {
        this.mixer = new THREE.AnimationMixer(this.root);
        const find = (names: string[]) => names.map(name => gltf!.animations.find(c => c.name === name)).find(Boolean);
        const map = { ...CLIP_FOR, ...this.options.clipFor };
        for (const [state, names] of Object.entries(map) as [MotionState, string[]][]) {
          const clip = find(names);
          if (clip) this.actions.set(state, this.mixer.clipAction(clip).setLoop(THREE.LoopRepeat, Infinity));
        }
        this.clips = gltf.animations.length;
        this.play(this.actions.get('idle'));
      }
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
