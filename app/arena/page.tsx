import ArenaPanel from "@/components/ArenaPanel";

export default function ArenaPage() {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Motion Arena</h1>
        <p className="text-sm text-gray-600 mt-1">
          One line in your voice, two ways of moving to it. Play it, watch both, pick the one that moves more like you. Ratings and notes are optional; the choice is the data.
        </p>
      </header>
      <ArenaPanel />
    </div>
  );
}
