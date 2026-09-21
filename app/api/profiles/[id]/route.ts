import { deleteProfile, getProfile } from "@/lib/storage";

export const dynamic = "force-dynamic";

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const profile = await getProfile(id);
  if (!profile) return Response.json({ error: "not found" }, { status: 404 });
  return Response.json(profile);
}

/** Remove the profile: its data directory, its served track, and with it its arena candidacy. */
export async function DELETE(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  if (!(await deleteProfile(id))) return Response.json({ error: "not found" }, { status: 404 });
  return Response.json({ ok: true });
}
