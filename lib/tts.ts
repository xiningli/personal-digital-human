// Speech for arena lines comes from Personal Voice Clone Studio's service (the same
// checkpoint, profile and per-emotion instruction the digital human uses), cached under
// public/audio/arena/ by a hash of everything that shaped it. Nothing here is generated
// twice, and a repeat trial replays the same file.

import crypto from "crypto";
import fs from "fs/promises";
import path from "path";
import { AUDIO_DIR } from "./storage";

export const STUDIO_URL = (process.env.STUDIO_TTS_URL ?? "http://127.0.0.1:8010").replace(/\/$/, "");
export const STUDIO_ROOT = process.env.STUDIO_ROOT ?? path.join(process.env.HOME ?? "", "personal-voice-clone-studio");

/** The digital human's delivery table (personal-site backend/tts.py, ADR-027). */
export const EMOTION_INSTRUCT: Record<string, string> = {
  neutral: "Speak naturally and clearly, in a relaxed conversational voice, like answering a guest in person.",
  warm: "Speak in a warm, friendly, gentle voice, smiling, like welcoming a friend into your home.",
  happy: "Speak in a bright, happy voice, smiling, with lively energy and light emphasis.",
  excited: "Speak with real excitement and quick energy, eager to share something you love.",
  serious: "Speak in a calm, serious, thoughtful voice, measured and sincere.",
  calm: "Speak slowly and calmly, soft and steady, reassuring.",
  curious: "Speak with curiosity and interest, thinking aloud, rising a little on the questions.",
  surprised: "Speak with pleasant surprise, a little breathless, delighted by what you just heard.",
  tender: "Speak softly and tenderly, close and caring, unhurried.",
};
export const EMOTION_FOR_CATEGORY: Record<string, string> = { greeting: "warm", answer: "curious", explaining: "neutral", thinking: "curious", delight: "happy" };

interface Profile { id: string; emotion?: string; language?: string; promptAudioPath?: string; promptText?: string }
interface ServedProfile { id: string; emotion?: string; wav: string; promptText?: string }

/**
 * The neutral reference clip as a path on the machine that runs the voice service. The
 * service says which clips it has (`GET /v1/profiles`, present on the studio and on every
 * exported bundle); the studio's local data file is the fallback for an older service.
 */
export async function neutralProfile(): Promise<{ wav: string; text: string }> {
  try {
    const r = await fetch(`${STUDIO_URL}/v1/profiles`, { cache: "no-store" });
    if (r.ok) {
      const served = (await r.json()) as ServedProfile[];
      const p = served.find((x) => x.emotion === "neutral") ?? served[0];
      if (p?.wav) return { wav: p.wav, text: p.promptText ?? "" };
    }
  } catch { /* fall through to the local file */ }
  const file = path.join(STUDIO_ROOT, "data", "voice-profiles.json");
  const profiles = JSON.parse(await fs.readFile(file, "utf-8")) as Profile[];
  const p = profiles.find((x) => x.emotion === "neutral" && x.promptAudioPath) ?? profiles.find((x) => x.promptAudioPath);
  if (!p?.promptAudioPath) throw new Error("the voice service has no prepared profile");
  return { wav: path.join(STUDIO_ROOT, "public", p.promptAudioPath.replace(/^\/+/, "")), text: p.promptText ?? "" };
}

export async function studioHealth(): Promise<{ reachable: boolean; ready?: boolean; model?: string }> {
  try {
    const r = await fetch(`${STUDIO_URL}/health`, { cache: "no-store" });
    const j = (await r.json()) as { ready?: boolean; model?: string };
    return { reachable: true, ready: j.ready, model: j.model };
  } catch { return { reachable: false }; }
}

/** Synthesize (or reuse) one line; returns the public path and the checkpoint that spoke it. */
export async function speak(text: string, emotion: string, seed = 7): Promise<{ audioPath: string; voiceModel: string; emotion: string }> {
  const { wav, text: promptText } = await neutralProfile();
  const instruct = EMOTION_INSTRUCT[emotion] ?? EMOTION_INSTRUCT.neutral;
  const health = await studioHealth();
  if (!health.reachable || !health.ready) throw new Error(`the voice service at ${STUDIO_URL} is not ready`);
  const voiceModel = health.model ?? "unknown";
  const key = crypto.createHash("sha1").update([voiceModel, wav, instruct, String(seed), text].join("|")).digest("hex").slice(0, 16);
  const file = path.join(AUDIO_DIR, `${key}.wav`);
  const audioPath = `/audio/arena/${key}.wav`;
  try { await fs.access(file); return { audioPath, voiceModel, emotion }; } catch { /* synthesize below */ }
  const r = await fetch(`${STUDIO_URL}/v1/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, reference_audio: wav, prompt_text: promptText, instruct, speed: 1.0, mode: "auto", seed, trim_breath: true }),
  });
  if (!r.ok) throw new Error(`the voice studio answered ${r.status}`);
  await fs.mkdir(AUDIO_DIR, { recursive: true });
  await fs.writeFile(file, Buffer.from(await r.arrayBuffer()));
  return { audioPath, voiceModel, emotion };
}
