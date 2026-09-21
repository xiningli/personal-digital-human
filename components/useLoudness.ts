"use client";

import { useCallback, useEffect, useRef } from "react";

/**
 * Loudness of a playing <audio>, 0..1, scaled by its own loudest window (the site's rule).
 * Works on any HTMLAudioElement, in the DOM or held in a ref (`new Audio(...)`); the
 * AudioContext is closed when the element changes or the component unmounts.
 */
export function useLoudness(audio: HTMLAudioElement | null) {
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
