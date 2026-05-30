import { auth } from "@/auth";
import ChatClient from "@/components/chat-client";

export default async function HomePage() {
  const session = await auth();
  const userEmail = session?.user?.email ?? "";
  const rawName = session?.user?.name;
  // name → first part of email → "Khách" — never show raw "user"
  const userName = rawName ?? (userEmail ? userEmail.split("@")[0] : "Khách");
  const userImage = session?.user?.image;

  return (
    <main className="flex min-h-screen flex-col bg-background">
      <ChatClient
        userName={userName}
        userEmail={userEmail}
        userImage={userImage ?? undefined}
      />
    </main>
  );
}
