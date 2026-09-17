import fs from "fs/promises";
import path from "path";
import type { ArenaRound, ArenaVote } from "./types";

const ROOT = process.cwd();
export const DATA_DIR = path.join(ROOT, "data");
const ROUNDS_FILE = path.join(DATA_DIR, "arena-rounds.json");
const VOTES_FILE = path.join(DATA_DIR, "arena-votes.jsonl");
export const AUDIO_DIR = path.join(ROOT, "public", "audio", "arena");

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
