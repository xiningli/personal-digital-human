import { deleteArenaRound, getArenaRound } from "@/lib/storage";
export async function GET(_: Request, { params }: { params: Promise<{ id: string }> }) {
  const round = await getArenaRound((await params).id);
  return round ? Response.json(round) : Response.json({ error: "Round not found" }, { status: 404 });
}
export async function DELETE(_: Request, { params }: { params: Promise<{ id: string }> }) {
  return Response.json({ deleted: await deleteArenaRound((await params).id) });
}
