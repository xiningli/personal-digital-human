#!/usr/bin/env node
/**
 * Put every capture clip in the asset on the same feet: write motion/ground_track.py's solve
 * back into the glb. Runs after fix-avatar-motion.mjs (import-avatar.sh).
 *
 * Why (measured 2026-09-28, motion/physics_audit.py): the Mixamo clips on this rig all stand
 * crouched and wide -- 15-34 deg of knee flexion, ankles 34-43 cm apart, the Breathing Idle toes
 * turned out ~40 deg -- a retargeting leftover, not the captures' intent, and the "gorilla" the
 * owner kept rejecting. The speaking track (a video-extracted profile) stood somewhere else
 * again, so every crossfade between idle and speaking dragged the feet to a new place.
 * ground_track.py solves each clip's legs from the ground up onto one natural stance (ankles
 * under the hip joints, toes out 10 deg, knees soft), with the pelvis carrying the weight shift;
 * this script writes the result: the Hips translation plus the rotation of every bone the solve
 * produced, on the clip's own 30 fps grid (fix-avatar-motion.mjs already resampled onto it).
 *
 * Idempotent: a grounded clip is marked in the animation's extras and skipped next time (the
 * solve keeps a fraction of the source's weight shift, so a second pass would shrink it again).
 *
 * Usage: node scripts/ground-avatar-clips.mjs <in.glb> [out.glb]
 */
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { NodeIO } from '@gltf-transform/core';
import { prune } from '@gltf-transform/functions';

const here = path.dirname(fileURLToPath(import.meta.url));
const MOTION = path.join(here, '..', 'motion');
const PYTHON = path.join(MOTION, '.venv', 'bin', 'python');
const FPS = 30;

async function main() {
  const [inPath, outPath = inPath] = process.argv.slice(2);
  if (!inPath) { console.error('usage: ground-avatar-clips.mjs <in.glb> [out.glb]'); process.exit(1); }
  const io = new NodeIO();
  const doc = await io.read(inPath);
  const nodes = new Map(doc.getRoot().listNodes().map((n) => [n.getName(), n]));
  const work = mkdtempSync(path.join(tmpdir(), 'ground-clips-'));
  try {
    for (const anim of doc.getRoot().listAnimations()) {
      const name = anim.getName();
      if (anim.getExtras()?.grounded) { console.log(`${name}: already grounded, skipped`); continue; }
      const out = path.join(work, `${name.replace(/\W+/g, '_')}.json`);
      const log = execFileSync(PYTHON, [path.join(MOTION, 'ground_track.py'), `clip:${name}`, '--avatar', path.resolve(inPath), '--out', out], { encoding: 'utf8' });
      const track = JSON.parse(readFileSync(out, 'utf8'));
      const F = track.frames, B = track.bones.length;
      const grid = new Float32Array(F);
      for (let f = 0; f < F; f++) grid[f] = f / FPS;
      const timesAcc = doc.createAccessor(`${name}-ground-grid`).setType('SCALAR').setArray(grid);

      const channel = (node, targetPath) => anim.listChannels().find((c) => c.getTargetNode() === node && c.getTargetPath() === targetPath);
      const write = (node, targetPath, values, type) => {
        const acc = doc.createAccessor(`${name}-${node.getName()}-${targetPath}-ground`).setType(type).setArray(values);
        const ch = channel(node, targetPath);
        if (ch) { ch.getSampler().setInput(timesAcc).setOutput(acc).setInterpolation('LINEAR'); return; }
        const sampler = doc.createAnimationSampler().setInput(timesAcc).setOutput(acc).setInterpolation('LINEAR');
        anim.addSampler(sampler).addChannel(doc.createAnimationChannel().setTargetNode(node).setTargetPath(targetPath).setSampler(sampler));
      };
      for (let b = 0; b < B; b++) {
        const node = nodes.get(track.bones[b]);
        if (!node) continue;
        const values = new Float32Array(F * 4);
        for (let f = 0; f < F; f++) for (let k = 0; k < 4; k++) values[f * 4 + k] = track.quats[(f * B + b) * 4 + k];
        write(node, 'rotation', values, 'VEC4');
      }
      write(nodes.get('Hips'), 'translation', Float32Array.from(track.hips), 'VEC3');
      // Constant channels that outlast the grid would freeze a tail into three.js's clip duration.
      for (const ch of anim.listChannels()) {
        const input = ch.getSampler().getInput();
        if (input === timesAcc) continue;
        const times = input.getArray();
        if (times[times.length - 1] > grid[F - 1] + 1e-4) {
          const comps = ch.getTargetPath() === 'rotation' ? 4 : 3;
          const vals = ch.getSampler().getOutput().getArray();
          const two = new Float32Array(comps * 2);
          two.set(vals.slice(0, comps), 0); two.set(vals.slice(0, comps), comps);
          ch.getSampler()
            .setInput(doc.createAccessor(`${name}-const-t`).setType('SCALAR').setArray(new Float32Array([0, grid[F - 1]])))
            .setOutput(doc.createAccessor(`${name}-const-v`).setType(comps === 4 ? 'VEC4' : 'VEC3').setArray(two));
        }
      }
      anim.setExtras({ ...anim.getExtras(), grounded: track.grounded });
      console.log(`${name}: ${log.trim().split('\n').filter((l) => l.includes('stance')).join(' ')}`);
    }
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
  await doc.transform(prune());
  await io.write(outPath, doc);
  console.log(`wrote ${outPath}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
