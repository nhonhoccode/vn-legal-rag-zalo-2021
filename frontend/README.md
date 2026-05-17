# Frontend (Next.js 15)

> **Phase 1**: chỉ skeleton folder — Next.js app sẽ được init đầy đủ ở **Phase 7b** trong [ROADMAP.md](../docs/ROADMAP.md).

## Sẽ được tạo ở Phase 7b

```bash
cd project_AI/frontend
npx create-next-app@latest . --typescript --tailwind --app --no-src-dir
npm install next-auth@beta @auth/core
npx shadcn@latest init
npx shadcn@latest add button card input avatar dropdown-menu
npm install ai @ai-sdk/google @ai-sdk/openai
```

## Stack

- Next.js 15 (App Router)
- TypeScript
- Tailwind CSS
- shadcn/ui
- NextAuth.js v5 (Google + GitHub OAuth)
- Vercel AI SDK (streaming LLM response)

## Dev

```bash
cp .env.example .env.local
# Điền NEXTAUTH_SECRET, GOOGLE_*, GITHUB_*
npm install
npm run dev
```

Default chạy ở `http://localhost:3000`.
