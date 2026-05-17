export { auth as middleware } from "@/auth";

export const config = {
  // Protect tất cả routes EXCEPT login, OAuth callback, static files.
  matcher: ["/((?!api/auth|login|_next/static|_next/image|favicon.ico).*)"],
};
