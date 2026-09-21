"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const nav = [
  { href: "/arena", label: "Motion Arena" },
  { href: "/profiles", label: "Profiles" },
  { href: "/api/report", label: "Report" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [voice, setVoice] = useState<{ reachable: boolean; ready?: boolean; model?: string } | null>(null);
  useEffect(() => {
    let cancelled = false;
    const poll = () => fetch("/api/tts", { cache: "no-store" }).then((r) => r.json()).then((d) => { if (!cancelled) setVoice(d); }).catch(() => { if (!cancelled) setVoice({ reachable: false }); });
    poll();
    const t = setInterval(poll, 10000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);
  const dot = !voice ? "bg-gray-400" : !voice.reachable ? "bg-red-500" : voice.ready ? "bg-green-500" : "bg-yellow-400";
  const status = !voice ? "checking voice…" : !voice.reachable ? "voice studio offline" : voice.ready ? `voice: ${voice.model?.split("/").pop()}` : "voice studio: model idle";
  return (
    <aside className="w-60 shrink-0 border-r border-gray-200 bg-white p-5 flex flex-col gap-6">
      <div>
        <div className="font-semibold">Personal Digital Human</div>
        <div className="text-xs text-gray-500">preference learning for body language</div>
      </div>
      <nav className="flex flex-col gap-1">
        {nav.map((n) => (
          <Link key={n.href} href={n.href} className={`px-3 py-2 rounded-lg text-sm ${pathname === n.href ? "bg-gray-900 text-white" : "hover:bg-gray-100"}`}>{n.label}</Link>
        ))}
      </nav>
      <div className="mt-auto text-xs text-gray-500 flex items-center gap-2"><span className={`w-2 h-2 rounded-full ${dot}`} />{status}</div>
    </aside>
  );
}
