import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const session = await auth();
  if (!session?.user) {
    return NextResponse.json({ error: "SESSION_EXPIRED" }, { status: 401 });
  }

  const id = req.nextUrl.searchParams.get("id");
  if (!id) return NextResponse.json({ error: "Missing id" }, { status: 400 });

  const apiKey = process.env.BACKEND_API_KEY;
  if (!apiKey) return NextResponse.json({ error: "Not configured" }, { status: 500 });

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const upstream = await fetch(`${apiUrl}/api/sessions/${encodeURIComponent(id)}`, {
    headers: { "X-API-Key": apiKey },
  });

  if (!upstream.ok) {
    return NextResponse.json({ error: `Backend ${upstream.status}` }, { status: upstream.status });
  }

  const data = await upstream.json();
  return NextResponse.json(data);
}
