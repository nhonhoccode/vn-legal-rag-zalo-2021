import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/auth";

// Force Node.js runtime + dynamic (no caching) — required for SSE pass-through.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Proxy POST /api/chat → backend POST /api/ask
 *
 * - Forward JWT-signed bearer token (NextAuth tự sign với NEXTAUTH_SECRET).
 * - Backend (FastAPI) decode với JWT_SECRET (cùng giá trị) → auth thành công.
 * - Stream response back nếu user gọi stream=true.
 */
export async function POST(req: NextRequest) {
  const session = await auth();
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // For server-to-server call, use a service token (API key) — không dùng JWT thật từ NextAuth
  // (NextAuth JWT là encrypted JWE, FastAPI không decode được).
  // Pattern an toàn: server-side proxy với API key duy nhất, không expose ra client.
  const apiKey = process.env.BACKEND_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: "BACKEND_API_KEY not configured" },
      { status: 500 }
    );
  }

  const isStream = body?.stream === true;

  const upstream = await fetch(`${apiUrl}/api/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": apiKey,
      // Forward user identity in custom header (backend có thể log per-user)
      "X-User-Id": session.user.id ?? session.user.email ?? "anonymous",
      "X-User-Email": session.user.email ?? "",
    },
    body: JSON.stringify(body),
  });

  if (!upstream.ok && !isStream) {
    const text = await upstream.text();
    return NextResponse.json(
      { error: `Backend error ${upstream.status}: ${text.slice(0, 500)}` },
      { status: upstream.status }
    );
  }

  if (isStream) {
    return new NextResponse(upstream.body, {
      status: upstream.status,
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        // no-transform = không cho Cloudflare/proxy buffer hoặc compress
        "Cache-Control": "no-cache, no-store, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
        "Content-Encoding": "identity",
        // CDN bypass hints
        "CDN-Cache-Control": "no-store",
        "Cloudflare-CDN-Cache-Control": "no-store",
      },
    });
  }

  const data = await upstream.json();
  return NextResponse.json(data);
}
