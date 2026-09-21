import ProfilePanel from "@/components/ProfilePanel";

export default function ProfilesPage() {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Motion Profiles</h1>
        <p className="text-sm text-gray-600 mt-1">
          A clip of someone whose body language you admire becomes a motion profile: GVHMR extracts how they move, and the result joins the arena&apos;s candidate pool to be ranked against mocap and generated motion.
        </p>
      </header>
      <ProfilePanel />
    </div>
  );
}
