import Link from "next/link";
import { notFound } from "next/navigation";
import ProfileEval from "@/components/ProfileEval";
import { getProfile } from "@/lib/storage";

export const dynamic = "force-dynamic";

export default async function ProfileEvalPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const profile = await getProfile(id);
  if (!profile || profile.status !== "ready" || !profile.trackPath) notFound();

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <Link href="/profiles" className="text-sm text-gray-500 hover:underline">← Back to profiles</Link>
        <h1 className="text-2xl font-semibold">{profile.name}</h1>
        <p className="text-sm text-gray-600">
          {profile.sourceType === "youtube" ? (
            <a href={profile.sourceRef} target="_blank" rel="noreferrer" className="hover:underline break-all">
              YouTube · {profile.sourceRef}
            </a>
          ) : (
            <>upload · {profile.sourceRef}</>
          )}
          {profile.durationS != null && <> · {profile.durationS.toFixed(0)} s</>}
        </p>
        {profile.stats && (
          <dl className="grid grid-cols-2 sm:grid-cols-4 gap-x-6 gap-y-1 text-xs text-gray-600 max-w-xl">
            <dt>worst limb</dt><dd className="text-right">{profile.stats.worstLimbDeg.toFixed(2)}°</dd>
            <dt>foot float</dt><dd className="text-right">{profile.stats.footFloatCm.toFixed(1)} cm</dd>
            <dt>foot skate</dt><dd className="text-right">{profile.stats.footSkateCmPerFrame.toFixed(2)} cm/frame</dd>
            <dt>diversity</dt><dd className="text-right">{profile.stats.diversity.toFixed(2)}</dd>
          </dl>
        )}
        <p className="text-xs text-gray-400">
          原视频与抽取的动作并排同步播放。看视频、听声音，然后给数字人的模仿打分。
        </p>
      </header>
      <ProfileEval profile={profile} />
    </div>
  );
}
