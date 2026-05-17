import { auth, signOut } from "@/auth";
import ChatClient from "@/components/chat-client";

export default async function HomePage() {
  const session = await auth();
  const userEmail = session?.user?.email ?? "user";
  const userName = session?.user?.name ?? userEmail;
  const userImage = session?.user?.image;

  return (
    <main className="flex min-h-screen flex-col bg-background bg-paper">
      <header className="border-b-2 border-ink bg-card">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-4 py-3">
          <div>
            <h1 className="font-serif text-xl font-semibold tracking-tight">
              Pháp Luật <span className="text-primary">Việt Nam</span>
            </h1>
            <p className="text-xs text-muted-foreground">
              Tra cứu qua AI · Lao động · Hình sự · Doanh nghiệp
            </p>
          </div>
          <div className="flex items-center gap-3">
            {userImage && (
              <img
                src={userImage}
                alt={userName}
                className="h-8 w-8 rounded-full"
                style={{ border: "1.5px solid rgb(var(--ink))" }}
              />
            )}
            <span className="hidden text-sm font-medium text-foreground sm:inline">
              {userName}
            </span>
            <form
              action={async () => {
                "use server";
                await signOut({ redirectTo: "/login" });
              }}
            >
              <button
                type="submit"
                className="neubrut rounded-lg bg-card px-3 py-1.5 text-xs font-semibold transition hover:bg-muted"
              >
                Logout
              </button>
            </form>
          </div>
        </div>
      </header>

      <ChatClient />

      <footer className="border-t border-border/20 bg-card py-3 text-center text-xs text-muted-foreground">
        <span className="font-serif italic">Pháp luật VN qua RAG</span>
        {" · "}backend{" "}
        <code className="rounded bg-muted px-1.5 py-0.5 font-mono">cx/gpt-5.5</code>
        {" · "}
        {new Date().getFullYear()}
      </footer>
    </main>
  );
}
