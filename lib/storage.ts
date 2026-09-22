import fs from "fs/promises";
import path from "path";
import type { ArenaRound, ArenaVote, MotionProfile, ProfileEval } from "./types";

const ROOT = process.cwd();
export const DATA_DIR = path.join(ROOT, "data");
const ROUNDS_FILE = path.join(DATA_DIR, "arena-rounds.json");
const VOTES_FILE = path.join(DATA_DIR, "arena-votes.jsonl");
const EVALS_FILE = path.join(DATA_DIR, "profile-evals.jsonl");
export const AUDIO_DIR = path.join(ROOT, "public", "audio", "arena");
export const PROFILES_DIR = path.join(DATA_DIR, "profiles");
export const PUBLIC_MOTION_DIR = path.join(ROOT, "public", "motion");

async function ensureDirs() {
  await fs.mkdir(DATA_DIR, { recursive: true });
  await fs.mkdir(AUDIO_DIR, { recursive: true });
}

async function readJson<T>(file: string, fallback: T): Promise<T> {
  try { return JSON.parse(await fs.readFile(file, "utf-8")) as T; } catch { return fallback; }
}

export async function getArenaRounds(): Promise<ArenaRound[]> {
  await ensureDirs();
  return readJson<ArenaRound[]>(ROUNDS_FILE, []);
}

export async function getArenaRound(id: string): Promise<ArenaRound | null> {
  return (await getArenaRounds()).find((r) => r.id === id) ?? null;
}

export async function saveArenaRound(round: ArenaRound): Promise<void> {
  const rounds = await getArenaRounds();
  const i = rounds.findIndex((r) => r.id === round.id);
  if (i >= 0) rounds[i] = round; else rounds.push(round);
  await fs.writeFile(ROUNDS_FILE, JSON.stringify(rounds, null, 2), "utf-8");
}

export async function deleteArenaRound(id: string): Promise<boolean> {
  const rounds = await getArenaRounds();
  const kept = rounds.filter((r) => r.id !== id);
  if (kept.length === rounds.length) return false;
  await fs.writeFile(ROUNDS_FILE, JSON.stringify(kept, null, 2), "utf-8");
  return true;
}

export async function getArenaVotes(): Promise<ArenaVote[]> {
  await ensureDirs();
  try {
    const text = await fs.readFile(VOTES_FILE, "utf-8");
    return text.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l) as ArenaVote);
  } catch { return []; }
}

export async function appendArenaVote(vote: ArenaVote): Promise<void> {
  await ensureDirs();
  await fs.appendFile(VOTES_FILE, JSON.stringify(vote) + "\n", "utf-8");
}

// Profile evals (lib/types.ts): human faithfulness scores from the triptych eval page,
// one JSON line per submission.

export async function getProfileEvals(profileId?: string): Promise<ProfileEval[]> {
  await ensureDirs();
  let evals: ProfileEval[] = [];
  try {
    const text = await fs.readFile(EVALS_FILE, "utf-8");
    evals = text.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l) as ProfileEval);
  } catch { return []; }
  return profileId ? evals.filter((e) => e.profileId === profileId) : evals;
}

/** Number of sentence segments in the served public/profile-<id>.segments.json; null when absent. */
export async function getProfileSegmentCount(id: string): Promise<number | null> {
  const d = await readJson<{ segments?: unknown[] } | null>(
    path.join(PUBLIC_MOTION_DIR, `profile-${id}.segments.json`), null,
  );
  return Array.isArray(d?.segments) ? d.segments.length : null;
}

export async function appendProfileEval(ev: ProfileEval): Promise<void> {
  await ensureDirs();
  await fs.appendFile(EVALS_FILE, JSON.stringify(ev) + "\n", "utf-8");
}

// Motion profiles (lib/types.ts): one directory per profile with the source clip, the
// extracted npz, the audio and the meta record; the served track lives under public/motion/.

export function profileDir(id: string): string {
  return path.join(PROFILES_DIR, id);
}

export async function getProfiles(): Promise<MotionProfile[]> {
  let ids: string[] = [];
  try { ids = await fs.readdir(PROFILES_DIR); } catch { return []; }
  const profiles: MotionProfile[] = [];
  for (const id of ids) {
    const meta = await readJson<MotionProfile | null>(path.join(PROFILES_DIR, id, "meta.json"), null);
    if (meta) profiles.push(meta);
  }
  return profiles.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export async function getProfile(id: string): Promise<MotionProfile | null> {
  return readJson<MotionProfile | null>(path.join(PROFILES_DIR, id, "meta.json"), null);
}

export async function saveProfile(profile: MotionProfile): Promise<void> {
  const dir = profileDir(profile.id);
  await fs.mkdir(dir, { recursive: true });
  await fs.writeFile(path.join(dir, "meta.json"), JSON.stringify(profile, null, 2), "utf-8");
}

export async function deleteProfile(id: string): Promise<boolean> {
  const profile = await getProfile(id);
  if (!profile) return false;
  await fs.rm(profileDir(id), { recursive: true, force: true });
  await fs.rm(path.join(PUBLIC_MOTION_DIR, `profile-${id}.track.json`), { force: true });
  await fs.rm(path.join(PUBLIC_MOTION_DIR, `profile-${id}.segments.json`), { force: true });
  return true;
}
