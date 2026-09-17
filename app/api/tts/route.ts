import { studioHealth } from "@/lib/tts";
export const dynamic = "force-dynamic";
export async function GET() { return Response.json(await studioHealth()); }
