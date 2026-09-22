// The motion-profile pipeline (docs/protocol.md §5): video -> GVHMR SMPL-X -> retargeted
// track the player can serve. Every step is an existing tool under motion/; this file only
// sequences them as child processes and records the outcome on the profile.
//
// One job at a time, through the module-level `queue`: the GPU is shared with the voice
// server, and two GVHMR runs would not fit in it. Never parallelize this.

import { spawn } from "node:child_process";
import fs from "fs/promises";
import path from "path";
import { getProfile, getProfiles, profileDir, saveProfile, PUBLIC_MOTION_DIR } from "./storage";
import { profileTrackPath } from "./policies";
import type { MotionPolicy, MotionProfile } from "./types";

const ROOT = process.cwd();
const EXTRACT = path.join(ROOT, "motion", "extract");
const MOTION = path.join(ROOT, "motion");
const AVATAR = path.join(ROOT, "public", "assets", "model-clips.glb");
const STEP_TIMEOUT_MS = 10 * 60 * 1000;

let queue: Promise<unknown> = Promise.resolve();

/** Ready profiles as arena candidates (data-driven, so they live here next to storage). */
export async function profilePolicies(): Promise<MotionPolicy[]> {
  const profiles = await getProfiles();
  return profiles
    .filter((p) => p.status === "ready")
    .map((p) => ({ id: `profile:${p.id}`, name: `${p.name} (video profile)`, source: "profile", clip: null, profileId: p.id }));
}

interface StepResult { stdout: string; stderr: string }

function runStep(log: fs.FileHandle, cmd: string, args: string[], cwd: string): Promise<StepResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { cwd });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`${path.basename(cmd)} timed out after ${STEP_TIMEOUT_MS / 60000} min`));
    }, STEP_TIMEOUT_MS);
    child.stdout.on("data", (d) => { stdout += d; void log.write(d); });
    child.stderr.on("data", (d) => { stderr += d; void log.write(d); });
    child.on("error", (e) => { clearTimeout(timer); reject(e); });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) resolve({ stdout, stderr });
      else {
        // gate scripts (verify_retarget, metrics, check_track) print their FAIL lines to
        // stdout, so fall back to it when stderr is empty or the reason is lost.
        const detail = (stderr.trim() || stdout.trim()).split("\n").slice(-5).join(" ").slice(0, 300);
        reject(new Error(`${path.basename(cmd)} exited ${code}: ${detail}`));
      }
    });
  });
}

function num(re: RegExp, text: string): number | null {
  const m = re.exec(text);
  return m ? Number(m[1]) : null;
}

async function extract(profile: MotionProfile, opts: { start?: number; duration?: number; maxFootFloatCm?: number }): Promise<void> {
  const dir = profileDir(profile.id);
  const source = path.join(dir, "source.mp4");
  const npz = path.join(dir, "motion.npz");
  const audio = path.join(dir, "audio.wav");
  const track = path.join(PUBLIC_MOTION_DIR, `profile-${profile.id}.track.json`);
  const log = await fs.open(path.join(dir, "run.log"), "a");
  try {
    if (profile.sourceType === "youtube") {
      // fetch.sh prints the trimmed clip's path as its last stdout line.
      const args = [path.join(EXTRACT, "fetch.sh"), profile.sourceRef];
      if (opts.start !== undefined) args.push(String(opts.start));
      if (opts.duration !== undefined) args.push(String(opts.duration));
      const { stdout } = await runStep(log, "bash", args, EXTRACT);
      const fetched = stdout.trim().split("\n").pop()?.trim();
      if (!fetched) throw new Error("fetch.sh printed no output path");
      await fs.copyFile(fetched, source);
    }

    // extract.py keys GVHMR's preprocess cache by the video filename stem, so passing every
    // profile's "source.mp4" shares one cache directory — a stale cache from a shorter clip
    // poisons the next run (frame-count mismatch). Give each profile a unique-stem copy.
    const workVideo = path.join(EXTRACT, "videos", `profile-${profile.id}.mp4`);
    await fs.copyFile(source, workVideo);
    await runStep(log, path.join(EXTRACT, ".venv", "bin", "python"), [path.join(EXTRACT, "extract.py"), workVideo, npz], EXTRACT);
    await runStep(log, "ffmpeg", ["-y", "-hide_banner", "-loglevel", "error", "-i", source, "-vn", "-ar", "16000", "-ac", "1", audio], dir);
    await runStep(log, path.join(MOTION, ".venv", "bin", "python"), [path.join(MOTION, "retarget.py"), npz, AVATAR, track], MOTION);

    const verify = await runStep(log, path.join(MOTION, ".venv", "bin", "python"), [path.join(MOTION, "verify_retarget.py"), npz, track, "--check"], MOTION);
    const worstLimbDeg = num(/worst limb-direction error:\s*([\d.]+)°/, verify.stdout);
    if (worstLimbDeg === null) throw new Error("verify_retarget printed no worst-limb number");

    // Elegance post-pass: recenter the arms/hips onto the Talking capture's mean pose and
    // converge the GVHMR-estimated legs toward rest (see refine_track.py's docstring for the
    // measurements behind it). Additive, like segments: the raw track is backed up first and
    // a refine failure only lands in the log, never fails the profile.
    try {
      await fs.copyFile(track, path.join(dir, "track.unrefined.json"));
      await runStep(log, path.join(MOTION, ".venv", "bin", "python"), [path.join(MOTION, "refine_track.py"), track], MOTION);
    } catch (e) {
      void log.write(`refine_track.py failed (profile still ready): ${e instanceof Error ? e.message : String(e)}\n`);
    }

    // Quality gate on the final playback artifact: every check above fired on the
    // upstream npz, and refine then rewrote the track. check_track.py FKs the refined
    // track.json the player actually serves — clipping against the calibrated surfaces,
    // foot float/skate, angular velocity — and a failure here fails the profile. It
    // runs after refine (the last step that mutates the track), so it is also what
    // judges the unrefined track when refine itself only logged its failure.
    await runStep(log, path.join(MOTION, ".venv", "bin", "python"), [path.join(MOTION, "check_track.py"), AVATAR, track, "--check"], MOTION);

    const metricsArgs = [path.join(MOTION, "metrics.py"), "--check", npz, audio];
    if (opts.maxFootFloatCm !== undefined) metricsArgs.push("--foot-float-mean-max-cm", String(opts.maxFootFloatCm));
    const metrics = await runStep(log, path.join(MOTION, ".venv", "bin", "python"), metricsArgs, MOTION);
    const durationS = num(/\(([\d.]+)s at 30 fps\)/, metrics.stdout);
    const diversity = num(/diversity\s+([\d.]+)/, metrics.stdout);
    const footFloatCm = num(/foot float mean\s+([\d.]+) cm/, metrics.stdout);
    const footSkateCmPerFrame = num(/foot skate mean\s+([\d.]+) cm\/frame/, metrics.stdout);
    if (diversity === null || footFloatCm === null || footSkateCmPerFrame === null) throw new Error("metrics printed no stats");

    // Sentence segments + embeddings are additive: a clip with no recognizable speech
    // (or a whisper/model failure) still becomes a ready profile, just without segments.
    let hasSegments = false;
    try {
      await runStep(log, path.join(EXTRACT, ".venv", "bin", "python"), [
        path.join(EXTRACT, "segment.py"),
        audio,
        track,
        path.join(dir, "segments.json"),
        path.join(PUBLIC_MOTION_DIR, `profile-${profile.id}.segments.json`),
      ], EXTRACT);
      hasSegments = true;
    } catch (e) {
      void log.write(`segment.py failed (profile still ready): ${e instanceof Error ? e.message : String(e)}\n`);
    }

    // Skeleton data for the eval page's middle column (docs/protocol.md §5): SMPL-X FK
    // joint positions off the same npz, served next to the track. Additive like segments —
    // a failure only lands in the log, never fails the profile.
    try {
      await runStep(log, path.join(MOTION, ".venv", "bin", "python"), [
        path.join(MOTION, "export_joints3d.py"),
        npz,
        path.join(PUBLIC_MOTION_DIR, `profile-${profile.id}.joints3d.json`),
      ], MOTION);
    } catch (e) {
      void log.write(`export_joints3d.py failed (profile still ready): ${e instanceof Error ? e.message : String(e)}\n`);
    }

    const done = await getProfile(profile.id);
    if (!done) return; // deleted while extracting
    await saveProfile({
      ...done,
      status: "ready",
      error: undefined,
      trackPath: profileTrackPath(profile.id),
      durationS: durationS ?? undefined,
      stats: { worstLimbDeg, footFloatCm, footSkateCmPerFrame, diversity },
      hasSegments,
    });
  } finally {
    await log.close();
  }
}

/**
 * Kick off extraction for a processing profile and return immediately. Serialized through
 * `queue`, so a profile uploaded while another is running simply waits its turn.
 */
export function startProfileExtraction(profile: MotionProfile, opts: { start?: number; duration?: number; maxFootFloatCm?: number } = {}): void {
  queue = queue.then(async () => {
    try {
      await extract(profile, opts);
    } catch (e) {
      const current = await getProfile(profile.id);
      if (current) {
        await saveProfile({ ...current, status: "failed", error: e instanceof Error ? e.message : String(e) });
      }
    }
  });
}
