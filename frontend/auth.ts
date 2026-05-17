import NextAuth from "next-auth";
import Google from "next-auth/providers/google";
import GitHub from "next-auth/providers/github";

/**
 * NextAuth v5 config — JWT strategy.
 *
 * NEXTAUTH_SECRET phải GIỐNG JWT_SECRET ở backend/.env.
 * Backend (FastAPI) decode token bằng same secret + HS256.
 */
export const { auth, handlers, signIn, signOut } = NextAuth({
  // Cần TRUE khi deploy behind reverse proxy (Cloudflare Tunnel, nginx, etc.)
  // Otherwise: "UntrustedHost: Host must be trusted".
  trustHost: true,
  providers: [
    Google({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
    GitHub({
      clientId: process.env.GITHUB_CLIENT_ID!,
      clientSecret: process.env.GITHUB_CLIENT_SECRET!,
    }),
  ],
  session: { strategy: "jwt" },
  pages: {
    signIn: "/login",
  },
  callbacks: {
    async jwt({ token, user }) {
      // First sign-in: user object có. Sau đó chỉ token.
      if (user) {
        token.sub = user.id ?? token.sub;
        token.email = user.email ?? token.email;
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user && token.sub) {
        session.user.id = token.sub as string;
      }
      return session;
    },
  },
});
