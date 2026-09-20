"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import MotionStage from "./MotionStage";
import Leaderboard from "./Leaderboard";
import { LINE_BANK } from "@/lib/policies";
import type { ArenaRound, ArenaVote, CandidateRating } from "@/lib/types";

const SCALES: { key: keyof CandidateRating; label: string; hint: string }[] = [
  { key: "naturalness", label: "Natural", hint: "1 stiff or robotic … 5 a person talking" },
  { key: "fit", label: "Fits the words", hint: "1 unrelated to what is said … 5 moves with the phrasing" },
  { key: "likeness", label: "Like me", hint: "1 not how I move … 5 that is me" },
];

/** Loudness of a playing <audio>, 0..1, scaled by its own loudest window (the site's rule). */
function useLoudness(audio: HTMLAudioElement | null) {
  const analyser = useRef<AnalyserNode | null>(null);
  const buffer = useRef<Float32Array<ArrayBuffer> | null>(null);
  const peak = useRef(0.001);
  const smoothed = useRef(0);
  const lastAt = useRef(0);
  useEffect(() => {
    if (!audio) return;
    const ctx = new AudioContext();
    const source = ctx.createMediaElementSource(audio);
    const node = ctx.createAnalyser();
    node.fftSize = 1024;
    source.connect(node); node.connect(ctx.destination);
    analyser.current = node;
    buffer.current = new Float32Array(node.fftSize);
    const resume = () => { void ctx.resume(); };
    audio.addEventListener("play", resume);
    return () => { audio.removeEventListener("play", resume); void ctx.close(); analyser.current = null; };
  }, [audio]);
  return useCallback(() => {
    const node = analyser.current, buf = buffer.current;
    if (!node || !buf) return 0;
    node.getFloatTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    const rms = Math.sqrt(sum / buf.length);
    if (rms > peak.current) peak.current = rms;
    const target = Math.min(1, rms * Math.min(6, 0.4 / peak.current));
    const now = performance.now();
    const dt = Math.min(0.1, (now - lastAt.current) / 1000); lastAt.current = now;
    const k = 1 - Math.exp(-dt / (target > smoothed.current ? 0.045 : 0.11));
    smoothed.current += (target - smoothed.current) * k;
    return smoothed.current;
  }, []);
}

export default function ArenaPanel() {
  const [category, setCategory] = useState("any");
  const [round, setRound] = useState<ArenaRound | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [winner, setWinner] = useState<string | null | undefined>(undefined);
  const [ratings, setRatings] = useState<Record<string, Partial<CandidateRating>>>({});
  const [notes, setNotes] = useState("");
  const [candidateNotes, setCandidateNotes] = useState<Record<string, string>>({});
  const [history, setHistory] = useState<{ rounds: ArenaRound[]; votes: ArenaVote[] }>({ rounds: [], votes: [] });
  const [refresh, setRefresh] = useState(0);
  const [clips, setClips] = useState<string[] | null>(null);
  const [copied, setCopied] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [audioEl, setAudioEl] = useState<HTMLAudioElement | null>(null);
  const level = useLoudness(audioEl);

  useEffect(() => { setAudioEl(audioRef.current); }, [round?.id]);
  useEffect(() => {
    Promise.all([fetch("/api/arena/rounds").then((r) => r.json()), fetch("/api/arena/votes").then((r) => r.json())])
      .then(([rounds, votes]) => setHistory({ rounds, votes })).catch(() => {});
  }, [refresh]);

  /**
   * Show a round and put its id in the address bar, so a round can be linked to, reloaded and
   * quoted when something about it looks wrong. replaceState rather than a route change: the
   * canvases must not be torn down and rebuilt just because the URL gained an id.
   */
  const show = useCallback((r: ArenaRound | null) => {
    setRound(r); setWinner(undefined); setRatings({}); setNotes(""); setCandidateNotes({}); setPlaying(false);
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    if (r) url.searchParams.set("round", r.id); else url.searchParams.delete("round");
    window.history.replaceState(null, "", url.toString());
  }, []);

  // A link opened with ?round=<id> restores that round instead of an empty page.
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("round");
    if (!id) return;
    fetch(`/api/arena/rounds/${encodeURIComponent(id)}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("that round is not in this arena"))))
      .then((r: ArenaRound) => { setRound(r); setPlaying(false); })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const draw = async () => {
    setBusy(true); setError(null);
    try {
      const r = await fetch("/api/arena/random", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ category }) });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error ?? `HTTP ${r.status}`);
      show(data);
      setRefresh((n) => n + 1);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };

  const submit = async () => {
    if (!round || winner === undefined) return;
    setBusy(true); setError(null);
    try {
      const clean: Record<string, CandidateRating> = {};
      for (const [id, r] of Object.entries(ratings)) if (r.naturalness && r.fit && r.likeness) clean[id] = r as CandidateRating;
      const r = await fetch("/api/arena/votes", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ roundId: round.id, winnerId: winner, ratings: clean, notes, candidateNotes }) });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error ?? `HTTP ${r.status}`);
      setRefresh((n) => n + 1);
      await draw();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); setBusy(false); }
  };

  const voted = new Set(history.votes.map((v) => v.roundId));
  const recent = [...history.rounds].sort((a, b) => b.createdAt.localeCompare(a.createdAt)).slice(0, 12);

  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center gap-3">
        <select value={category} onChange={(e) => setCategory(e.target.value)} className="border rounded-lg px-3 py-2 bg-white">
          <option value="any">Any category</option>
          {LINE_BANK.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
        </select>
        <button onClick={draw} disabled={busy} className="px-4 py-2 rounded-lg bg-gray-900 text-white disabled:opacity-50">🎲 New round</button>
        {clips && <span className="text-xs text-gray-500">{clips.length ? `${clips.length} clips in the avatar` : "clipless avatar: procedural body only"}</span>}
        {error && <span className="text-sm text-red-600">{error}</span>}
      </section>

      {round && (
        <section className="space-y-4">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-sm text-gray-500">
            <span className="uppercase tracking-wide text-xs">{round.trial === "repeat" ? "repeat trial" : "test trial"}</span>
            <span>{LINE_BANK.find((c) => c.id === round.category)?.label ?? round.category}</span>
            <span>· voice {round.emotion}</span>
            <span className="font-mono text-xs text-gray-400">· {round.id.slice(0, 8)}</span>
            <button
              onClick={() => { navigator.clipboard?.writeText(window.location.href).then(() => { setCopied(true); window.setTimeout(() => setCopied(false), 1500); }).catch(() => {}); }}
              title={typeof window === "undefined" ? "" : window.location.href}
              className="text-xs underline decoration-dotted hover:text-gray-700"
            >
              {copied ? "link copied" : "copy link"}
            </button>
          </div>
          <p className="text-lg leading-relaxed">“{round.text}”</p>
          <audio ref={audioRef} key={round.id} src={round.audioPath} controls className="w-full"
            onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} />
          <div className="grid grid-cols-2 gap-4">
            {round.candidates.map((c) => (
              <div key={c.id} className="space-y-3">
                <MotionStage policy={c.policy} avatar={round.avatar} audioPath={round.audioPath} level={level} playing={playing} seed={round.seed} label={c.label} onClips={setClips} />
                <button onClick={() => setWinner(c.id)}
                  className={`w-full px-3 py-2 rounded-lg border ${winner === c.id ? "bg-gray-900 text-white border-gray-900" : "bg-white hover:bg-gray-50"}`}>
                  {c.label} moves better
                </button>
                <div className="grid grid-cols-3 gap-2">
                  {SCALES.map((s) => (
                    <label key={s.key} className="text-xs text-gray-600" title={s.hint}>
                      {s.label}
                      <select value={ratings[c.id]?.[s.key] ?? ""} onChange={(e) => setRatings((r) => ({ ...r, [c.id]: { ...r[c.id], [s.key]: Number(e.target.value) || undefined } }))}
                        className="mt-1 w-full border rounded px-2 py-1 bg-white text-sm">
                        <option value="">–</option>
                        {[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{n}</option>)}
                      </select>
                    </label>
                  ))}
                </div>
                <input value={candidateNotes[c.id] ?? ""} onChange={(e) => setCandidateNotes((n) => ({ ...n, [c.id]: e.target.value }))}
                  placeholder={`Note on ${c.label}: "hands: too busy; want them still on the short words"`} className="w-full border rounded px-2 py-1 text-sm" />
                <details className="text-xs text-gray-400"><summary>policy</summary><code>{c.policy.id}</code></details>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button onClick={() => setWinner(null)} className={`px-3 py-2 rounded-lg border ${winner === null ? "bg-gray-900 text-white border-gray-900" : "bg-white hover:bg-gray-50"}`}>No preference</button>
            <input value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Overall note (optional)" className="flex-1 min-w-60 border rounded px-3 py-2 text-sm" />
            <button onClick={submit} disabled={busy || winner === undefined} className="px-4 py-2 rounded-lg bg-emerald-700 text-white disabled:opacity-50">Submit vote → next</button>
          </div>
        </section>
      )}

      <Leaderboard refresh={refresh} />

      <section>
        <h2 className="text-sm font-semibold text-gray-700 mb-2">Recent rounds</h2>
        <ul className="divide-y text-sm">
          {recent.map((r) => (
            <li key={r.id} className="py-2 flex items-center gap-3">
              <span className={`w-2 h-2 rounded-full ${voted.has(r.id) ? "bg-emerald-500" : "bg-gray-300"}`} />
              <span className="text-gray-400 text-xs">{r.trial}</span>
              <span className="font-mono text-gray-400 text-xs">{r.id.slice(0, 8)}</span>
              <span className="truncate flex-1">{r.text}</span>
              <button className="text-xs text-gray-500 hover:underline" onClick={() => show(r)}>open</button>
            </li>
          ))}
          {!recent.length && <li className="py-2 text-gray-400">No rounds yet. Press 🎲 New round.</li>}
        </ul>
      </section>
    </div>
  );
}
