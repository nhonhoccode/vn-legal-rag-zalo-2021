"use client";

import { useState, useRef, useEffect, FormEvent, useCallback } from "react";
import {
  Send, Loader2, Sparkles, MessageSquare, Copy, Check,
  Download, Zap, ZapOff, Moon, Sun, Globe, User, History,
  Share2, X, ChevronDown, ChevronUp,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { signOut } from "next-auth/react";
import { type Lang, t, getStoredLang, setStoredLang } from "@/lib/i18n";
import { saveSession, loadHistory, clearHistory, formatRelative } from "@/lib/history";

type Citation = {
  text_index: number;
  law_id: string;
  law_title: string;
  article_id: string;
  khoan_id: string | null;
  matched: boolean;
};

type Source = {
  chunk_id: string;
  score: number;
  law_id: string;
  law_title: string;
  article_id: string;
  domain: string;
  text: string;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  sources?: Source[];
  loading?: boolean;
  streaming?: boolean;
  rewritten?: string | null;
  timestamp?: number;
  stage?: "retrieval" | "reranking" | "generation" | null;
  errorKind?: "timeout" | "network" | "other" | null;
  retryQuery?: string;
  followUpQuestions?: string[];
};

const STAGE_LABEL: Record<string, string> = {
  retrieval: "🔍 Đang tìm văn bản liên quan...",
  reranking: "📊 Đang xếp hạng kết quả...",
  generation: "✍️ Đang tổng hợp câu trả lời...",
};

const DOMAINS = [
  { id: "all", label: "Tất cả" },
  { id: "labor", label: "Lao động" },
  { id: "criminal", label: "Hình sự" },
  { id: "business", label: "Doanh nghiệp" },
] as const;

const SUGGESTED_QUESTIONS = [
  // ─── LAO ĐỘNG ───
  { domain: "labor", icon: "💼", title: "Sa thải trái luật",
    question: "Người lao động bị sa thải trái pháp luật được bồi thường gì?" },
  { domain: "labor", icon: "📋", title: "Hợp đồng",
    question: "Hợp đồng lao động là gì? Có những loại nào?" },
  { domain: "labor", icon: "💰", title: "Lương tối thiểu",
    question: "Mức lương tối thiểu vùng năm 2024 là bao nhiêu?" },
  { domain: "labor", icon: "🏖️", title: "Nghỉ phép",
    question: "Người lao động được nghỉ phép năm bao nhiêu ngày?" },
  { domain: "labor", icon: "⏰", title: "Làm thêm giờ",
    question: "Quy định về làm thêm giờ và tiền lương làm thêm giờ?" },
  { domain: "labor", icon: "🤰", title: "Thai sản",
    question: "Lao động nữ được nghỉ thai sản bao lâu? Mức hưởng thế nào?" },
  { domain: "labor", icon: "🚪", title: "Đơn phương chấm dứt",
    question: "Khi nào người lao động được đơn phương chấm dứt hợp đồng?" },
  { domain: "labor", icon: "🛡️", title: "BHXH",
    question: "Quyền lợi bảo hiểm xã hội của người lao động gồm những gì?" },

  // ─── HÌNH SỰ ───
  { domain: "criminal", icon: "⚖️", title: "Trộm cắp",
    question: "Tội trộm cắp tài sản bị xử phạt như thế nào?" },
  { domain: "criminal", icon: "🛡️", title: "Tội phạm",
    question: "Khái niệm tội phạm theo Bộ luật Hình sự là gì?" },
  { domain: "criminal", icon: "🚗", title: "Vi phạm giao thông",
    question: "Mức phạt khi điều khiển xe máy quá tốc độ là bao nhiêu?" },
  { domain: "criminal", icon: "🍺", title: "Nồng độ cồn",
    question: "Lái xe có nồng độ cồn bị xử phạt thế nào?" },
  { domain: "criminal", icon: "🥊", title: "Cố ý gây thương tích",
    question: "Tội cố ý gây thương tích bị phạt bao nhiêu năm tù?" },
  { domain: "criminal", icon: "💊", title: "Ma túy",
    question: "Tội tàng trữ trái phép chất ma túy có mức án ra sao?" },
  { domain: "criminal", icon: "💸", title: "Lừa đảo",
    question: "Tội lừa đảo chiếm đoạt tài sản bị xử lý thế nào?" },
  { domain: "criminal", icon: "👮", title: "Án treo",
    question: "Điều kiện để được hưởng án treo là gì?" },

  // ─── DOANH NGHIỆP & ĐẦU TƯ ───
  { domain: "business", icon: "🏢", title: "Quyền doanh nghiệp",
    question: "Doanh nghiệp có những quyền gì khi hoạt động kinh doanh?" },
  { domain: "business", icon: "💰", title: "Ngành cấm đầu tư",
    question: "Những ngành nghề nào bị cấm đầu tư kinh doanh?" },
  { domain: "business", icon: "🏗️", title: "Thành lập DN",
    question: "Thủ tục thành lập công ty TNHH gồm những bước nào?" },
  { domain: "business", icon: "📊", title: "Vốn điều lệ",
    question: "Vốn điều lệ tối thiểu để thành lập doanh nghiệp là bao nhiêu?" },
  { domain: "business", icon: "📝", title: "Hợp đồng kinh doanh",
    question: "Hợp đồng mua bán hàng hóa cần có những điều khoản nào?" },
  { domain: "business", icon: "🛑", title: "Phá sản",
    question: "Doanh nghiệp khi nào được coi là mất khả năng thanh toán?" },
  { domain: "business", icon: "🤝", title: "M&A",
    question: "Thủ tục sáp nhập doanh nghiệp được quy định như thế nào?" },
  { domain: "business", icon: "🌐", title: "Đầu tư nước ngoài",
    question: "Nhà đầu tư nước ngoài cần điều kiện gì để đầu tư tại Việt Nam?" },
];

const FOLLOW_UP_HINTS = [
  "Giải thích cụ thể hơn",
  "Còn trường hợp khác thì sao?",
  "Mức phạt cụ thể là bao nhiêu?",
  "Tôi nên làm gì tiếp theo?",
];

interface ChatClientProps {
  userName?: string;
  userEmail?: string;
  userImage?: string;
}

export default function ChatClient({ userName = "Khách", userEmail = "", userImage }: ChatClientProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [useStreaming, setUseStreaming] = useState(true);
  const [lang, setLang] = useState<Lang>("vi");
  const [isDark, setIsDark] = useState(false);
  const [showDashboard, setShowDashboard] = useState(false);
  const [shareModal, setShareModal] = useState<string | null>(null); // URL string when open
  const [historyEntries, setHistoryEntries] = useState<ReturnType<typeof loadHistory>>([]);
  const [loadingSession, setLoadingSession] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Init theme + lang from localStorage
  useEffect(() => {
    setLang(getStoredLang());
    setIsDark(document.documentElement.classList.contains("dark"));
    setHistoryEntries(loadHistory());
  }, []);

  // Item 5: reset state on bfcache restore
  useEffect(() => {
    const handlePageShow = (e: PageTransitionEvent) => {
      if (e.persisted) {
        setMessages([]);
        setSessionId("");
        setIsLoading(false);
      }
    };
    window.addEventListener("pageshow", handlePageShow);
    return () => window.removeEventListener("pageshow", handlePageShow);
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  function toggleTheme() {
    const next = !isDark;
    setIsDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("vn-legal-theme", next ? "dark" : "light");
  }

  function toggleLang() {
    const next: Lang = lang === "vi" ? "en" : "vi";
    setLang(next);
    setStoredLang(next);
  }

  // Item 13: share link
  function shareConversation() {
    if (!messages.length) return;
    const payload = messages.map((m) => ({ r: m.role[0], c: m.content.slice(0, 400) }));
    try {
      const json = JSON.stringify(payload);
      // UTF-8 safe + URL-safe base64: encodeURI → unescape → btoa → replace +/= → URL safe
      const b64 = btoa(unescape(encodeURIComponent(json)));
      const urlSafe = b64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=/g, "");
      const url = `${window.location.origin}${window.location.pathname}?c=${urlSafe}`;
      setShareModal(url);
    } catch {
      setShareModal(null);
    }
  }

  // Load old session from history (click on history entry)
  async function loadOldSession(sid: string) {
    setLoadingSession(true);
    setShowDashboard(false);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`/api/session-proxy?id=${encodeURIComponent(sid)}`);
      if (!res.ok) {
        // Fallback: just set session ID so next query continues the conversation
        setSessionId(sid);
        setMessages([]);
        return;
      }
      const data = await res.json();
      const loaded: Message[] = (data.turns ?? []).map((t: { role: string; content: string }) => ({
        id: crypto.randomUUID(),
        role: t.role as "user" | "assistant",
        content: t.content,
        timestamp: Date.now(),
      }));
      setMessages(loaded);
      setSessionId(sid);
    } catch {
      setSessionId(sid);
      setMessages([]);
    } finally {
      setLoadingSession(false);
    }
  }

  // Load shared conversation from URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const c = params.get("c");
    if (!c) return;
    try {
      // Reverse URL-safe base64 → standard base64 → decode UTF-8
      const standard = c.replace(/-/g, "+").replace(/_/g, "/");
      const padded = standard + "=".repeat((4 - (standard.length % 4)) % 4);
      const decoded = JSON.parse(decodeURIComponent(escape(atob(padded))));
      const loaded: Message[] = decoded.map((m: { r: string; c: string }) => ({
        id: crypto.randomUUID(),
        role: m.r === "u" ? "user" : "assistant",
        content: m.c,
        timestamp: Date.now(),
      }));
      setMessages(loaded);
      window.history.replaceState({}, "", window.location.pathname);
    } catch { /* ignore invalid share links */ }
  }, []);

  async function sendQuery(query: string) {
    if (!query.trim() || isLoading) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
      timestamp: Date.now(),
    };
    const assistantMsg: Message = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      loading: true,
      timestamp: Date.now(),
    };
    setMessages((m) => [...m, userMsg, assistantMsg]);
    setInput("");
    setIsLoading(true);

    // Save to localStorage history
    if (!sessionId) {
      const newSid = crypto.randomUUID().slice(0, 8);
      saveSession(newSid, query);
      setHistoryEntries(loadHistory());
    } else {
      saveSession(sessionId, query);
      setHistoryEntries(loadHistory());
    }

    try {
      if (useStreaming) {
        try {
          await sendStreaming(query, assistantMsg.id);
        } catch (streamErr) {
          // Item 9: auto-retry once on stream interrupt
          console.warn("Stream failed, retrying once:", streamErr);
          setMessages((m) => m.map((msg) =>
            msg.id === assistantMsg.id
              ? { ...msg, content: t(lang, "retrying"), loading: true, streaming: false }
              : msg
          ));
          await sendStreaming(query, assistantMsg.id);
        }
      } else {
        await sendNonStreaming(query, assistantMsg.id);
      }
    } catch (err) {
      console.error(err);
      const msgStr = String(err);
      const kind: Message["errorKind"] = msgStr.includes("TIMEOUT") || msgStr.includes("524") || msgStr.includes("502") || msgStr.includes("504")
        ? "timeout"
        : msgStr.includes("Failed to fetch") || msgStr.includes("NetworkError")
        ? "network"
        : "other";
      const friendly = kind === "timeout"
        ? `⏱️ ${t(lang, "errorTimeout")}`
        : kind === "network"
        ? `🌐 ${t(lang, "errorNetwork")}`
        : `${t(lang, "errorGeneric")}: ${msgStr}`;
      setMessages((m) =>
        m.map((msg) =>
          msg.id === assistantMsg.id
            ? { ...msg, content: friendly, loading: false, streaming: false, stage: null, errorKind: kind, retryQuery: query }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }

  async function sendNonStreaming(query: string, assistantMsgId: string) {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 5, session_id: sessionId, stream: false }),
    });

    if (!res.ok) {
      const isTimeout = res.status === 524 || res.status === 502 || res.status === 504;
      if (res.status === 401) {
        const body = await res.text();
        if (body.includes("SESSION_EXPIRED")) {
          window.location.href = "/login";
          return;
        }
      }
      throw new Error(isTimeout ? `TIMEOUT (HTTP ${res.status})` : `HTTP ${res.status}: ${(await res.text()).slice(0, 100)}`);
    }
    const data = await res.json();
    if (data.session_id && !sessionId) setSessionId(data.session_id);

    setMessages((m) =>
      m.map((msg) =>
        msg.id === assistantMsgId
          ? {
              ...msg,
              content: data.answer ?? "(empty response)",
              citations: data.citations,
              sources: data.sources,
              rewritten: data.standalone_query,
              loading: false,
            }
          : msg
      )
    );
  }

  async function sendStreaming(query: string, assistantMsgId: string) {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 5, session_id: sessionId, stream: true }),
    });

    if (!res.ok || !res.body) {
      const isTimeout = res.status === 524 || res.status === 502 || res.status === 504;
      if (res.status === 401) {
        const body = await res.text();
        if (body.includes("SESSION_EXPIRED")) {
          window.location.href = "/login";
          return;
        }
      }
      throw new Error(isTimeout ? `TIMEOUT (HTTP ${res.status})` : `HTTP ${res.status}: ${(await res.text()).slice(0, 100)}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let accumText = "";

    // KHÔNG set loading=false ở đây — đợi progress event đầu tiên
    // (tránh empty bubble khi giữa thời điểm response arrive và first progress event)

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") continue;
        try {
          const event = JSON.parse(payload);
          if (event.type === "ready") {
            // Stream alive confirmation — không đổi UI, đợi progress event đầu tiên
            continue;
          } else if (event.type === "progress") {
            const stage = event.stage as Message["stage"];
            setMessages((m) =>
              m.map((msg) => (msg.id === assistantMsgId ? { ...msg, stage, loading: false, streaming: true } : msg))
            );
          } else if (event.type === "sources") {
            const sources = event.sources as Source[];
            setMessages((m) =>
              m.map((msg) => (msg.id === assistantMsgId ? { ...msg, sources } : msg))
            );
          } else if (event.type === "token") {
            accumText += event.value;
            const snapshot = accumText;
            setMessages((m) =>
              m.map((msg) =>
                msg.id === assistantMsgId ? { ...msg, content: snapshot, streaming: true, stage: null } : msg
              )
            );
          } else if (event.type === "done") {
            const citations = event.citations as Citation[];
            const sid = event.session_id as string | undefined;
            const followUpQuestions = (event.follow_up_questions as string[] | undefined) ?? [];
            if (sid && !sessionId) setSessionId(sid);
            setMessages((m) =>
              m.map((msg) =>
                msg.id === assistantMsgId
                  ? { ...msg, citations, streaming: false, loading: false, stage: null, followUpQuestions }
                  : msg
              )
            );
          }
        } catch (e) {
          console.warn("SSE parse error:", e, payload);
        }
      }
    }
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    sendQuery(input);
  }

  function clearChat() {
    setMessages([]);
    setSessionId("");
  }

  function downloadConversation() {
    const data = {
      session_id: sessionId,
      exported_at: new Date().toISOString(),
      messages: messages.map((m) => ({
        role: m.role,
        content: m.content,
        timestamp: m.timestamp,
        citations: m.citations?.map((c) => `${c.law_id} Điều ${c.article_id}`) ?? [],
      })),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `vn-legal-chat-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const hasConversation = messages.length > 0;
  const lastMsg = messages[messages.length - 1];
  const showFollowUps =
    hasConversation &&
    !isLoading &&
    lastMsg?.role === "assistant" &&
    !lastMsg?.loading &&
    !lastMsg?.streaming;

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      {/* ── Header ── */}
      <header className="border-b-2 border-ink bg-card">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-4 py-3">
          <div>
            <h1 className="font-serif text-xl font-semibold tracking-tight">
              {t(lang, "appTitle").includes("Pháp") ? (
                <>Pháp Luật <span className="text-primary">Việt Nam</span></>
              ) : t(lang, "appTitle")}
            </h1>
            <p className="text-xs text-muted-foreground">{t(lang, "appSubtitle")}</p>
          </div>
          <div className="flex items-center gap-2">
            {/* Theme toggle */}
            <button onClick={toggleTheme} title="Toggle dark mode"
              className="neubrut rounded-lg bg-card p-1.5 text-foreground hover:bg-muted">
              {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
            {/* Lang toggle */}
            <button onClick={toggleLang} title="Switch language"
              className="neubrut rounded-lg bg-card px-2 py-1 text-xs font-bold text-foreground hover:bg-muted">
              {lang === "vi" ? "EN" : "VI"}
            </button>
            {/* User avatar → dashboard */}
            <button onClick={() => { setShowDashboard(true); setHistoryEntries(loadHistory()); }}
              className="neubrut flex items-center gap-1.5 rounded-lg bg-card px-2 py-1.5 hover:bg-muted">
              {userImage ? (
                <img src={userImage} alt={userName} className="h-6 w-6 rounded-full border border-ink" />
              ) : (
                <User className="h-4 w-4" />
              )}
              <span className="hidden max-w-[120px] truncate text-xs font-medium sm:inline">{userName}</span>
            </button>
          </div>
        </div>
      </header>

      {/* ── Dashboard Modal ── */}
      {showDashboard && (
        <div className="fixed inset-0 z-50 flex items-start justify-end">
          <div className="absolute inset-0 bg-ink/20" onClick={() => setShowDashboard(false)} />
          <div className="relative m-4 mt-16 w-80 rounded-xl border-2 border-ink bg-card shadow-brut-lg">
            <div className="flex items-center justify-between border-b border-border/30 px-4 py-3">
              <span className="font-semibold">{t(lang, "dashTitle")}</span>
              <button onClick={() => setShowDashboard(false)}><X className="h-4 w-4" /></button>
            </div>
            <div className="space-y-4 p-4">
              {/* User info */}
              <div className="flex items-center gap-3">
                {userImage ? (
                  <img src={userImage} alt={userName} className="h-10 w-10 rounded-full border-2 border-ink" />
                ) : (
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary text-primary-foreground border-2 border-ink text-sm font-bold">
                    {userName[0]?.toUpperCase()}
                  </div>
                )}
                <div>
                  <div className="font-medium text-sm">{userName}</div>
                  {userEmail && <div className="text-xs text-muted-foreground">{userEmail}</div>}
                </div>
              </div>
              {/* Stats */}
              <div className="rounded-lg bg-muted/50 px-3 py-2 text-xs">
                <span className="font-medium">{t(lang, "dashStats")}: </span>
                {Math.floor(messages.length / 2)} {t(lang, "dashQueries")}
              </div>
              {/* Theme */}
              <div className="flex items-center justify-between">
                <span className="text-sm">{t(lang, "dashTheme")}</span>
                <button onClick={toggleTheme}
                  className="neubrut flex items-center gap-1.5 rounded-lg bg-muted px-3 py-1 text-xs font-semibold">
                  {isDark ? <Sun className="h-3 w-3" /> : <Moon className="h-3 w-3" />}
                  {isDark ? t(lang, "dashThemeLight") : t(lang, "dashThemeDark")}
                </button>
              </div>
              {/* Lang */}
              <div className="flex items-center justify-between">
                <span className="text-sm">{t(lang, "dashLang")}</span>
                <button onClick={toggleLang}
                  className="neubrut flex items-center gap-1.5 rounded-lg bg-muted px-3 py-1 text-xs font-semibold">
                  <Globe className="h-3 w-3" />
                  {lang === "vi" ? "Tiếng Việt → English" : "English → Tiếng Việt"}
                </button>
              </div>
              {/* History */}
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <span className="flex items-center gap-1 text-sm font-medium">
                    <History className="h-3.5 w-3.5" />
                    {t(lang, "dashHistory")}
                  </span>
                  {historyEntries.length > 0 && (
                    <button onClick={() => { clearHistory(); setHistoryEntries([]); }}
                      className="text-xs text-muted-foreground hover:text-foreground">Xóa tất cả</button>
                  )}
                </div>
                {historyEntries.length === 0 ? (
                  <p className="text-xs text-muted-foreground">{t(lang, "dashNoHistory")}</p>
                ) : (
                  <div className="max-h-48 space-y-1 overflow-y-auto">
                    {historyEntries.map((h) => (
                      <button
                        key={h.sessionId}
                        onClick={() => loadOldSession(h.sessionId)}
                        disabled={loadingSession}
                        className="w-full rounded-lg bg-muted/40 px-3 py-2 text-left text-xs hover:bg-muted/80 transition disabled:opacity-50"
                      >
                        <div className="truncate font-medium text-foreground">{h.firstQuery}</div>
                        <div className="text-muted-foreground">{formatRelative(h.timestamp)} · {h.queryCount} {t(lang, "dashQueries")}</div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
              {/* Logout */}
              <button
                onClick={() => signOut({ callbackUrl: "/login" })}
                className="neubrut w-full rounded-lg bg-card py-2 text-xs font-semibold text-foreground hover:bg-muted"
              >
                {t(lang, "logout")}
              </button>
            </div>
          </div>
        </div>
      )}
      {/* ── Share Modal ── */}
      {shareModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
          <div className="absolute inset-0 bg-ink/30" onClick={() => setShareModal(null)} />
          <div className="relative w-full max-w-lg rounded-xl border-2 border-ink bg-card shadow-brut-lg p-5">
            <div className="mb-3 flex items-center justify-between">
              <span className="font-semibold">Chia sẻ hội thoại</span>
              <button onClick={() => setShareModal(null)}><X className="h-4 w-4" /></button>
            </div>
            <p className="mb-3 text-xs text-muted-foreground">Copy link dưới đây và gửi cho người khác. Người nhận cần đăng nhập để xem.</p>
            <div className="flex gap-2">
              <input
                readOnly
                value={shareModal}
                onClick={(e) => (e.target as HTMLInputElement).select()}
                className="flex-1 rounded-lg bg-muted px-3 py-2 text-xs font-mono"
                style={{ border: "1px solid rgb(var(--ink) / 0.4)" }}
              />
              <button
                onClick={() => {
                  navigator.clipboard.writeText(shareModal).catch(() => {});
                  const inp = document.querySelector<HTMLInputElement>("[data-share-input]");
                  inp?.select();
                }}
                className="neubrut flex items-center gap-1 rounded-lg bg-primary px-3 py-2 text-xs font-semibold text-primary-foreground"
              >
                <Copy className="h-3 w-3" /> Copy
              </button>
            </div>
          </div>
        </div>
      )}

      <div
        ref={scrollRef}
        className="mx-auto flex w-full max-w-4xl flex-1 flex-col gap-4 overflow-y-auto px-4 py-6"
      >
        {!hasConversation ? (
          <EmptyState onPickQuestion={(q) => sendQuery(q)} />
        ) : (
          <>
            {messages.map((m) => (
              <MessageBubble key={m.id} message={m} onRetry={sendQuery} />
            ))}
            {showFollowUps && (
              <FollowUpSuggestions
                questions={lastMsg?.followUpQuestions}
                onPick={(q) => sendQuery(q)}
              />
            )}
          </>
        )}
      </div>

      <form onSubmit={handleSubmit} className="border-t-2 border-ink bg-card">
        {hasConversation && (
          <div className="mx-auto max-w-4xl px-4 pt-2">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
              <div className="flex items-center gap-2">
                <MessageSquare className="h-3 w-3" />
                <span className="font-medium">
                  Hội thoại đang nhớ ngữ cảnh ({Math.floor(messages.length / 2)} câu hỏi)
                </span>
                <span className="text-muted-foreground/50">· {sessionId.slice(0, 8)}…</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setUseStreaming((s) => !s)}
                  className="neubrut flex items-center gap-1 rounded-md bg-card px-2 py-1 font-semibold"
                  title="Toggle streaming response"
                >
                  {useStreaming ? (
                    <Zap className="h-3 w-3 text-accent" />
                  ) : (
                    <ZapOff className="h-3 w-3" />
                  )}
                  Stream {useStreaming ? "ON" : "OFF"}
                </button>
                <button
                  type="button"
                  onClick={downloadConversation}
                  className="neubrut flex items-center gap-1 rounded-md bg-card px-2 py-1 font-semibold"
                  title="Download hội thoại JSON"
                >
                  <Download className="h-3 w-3" />
                  {t(lang, "save")}
                </button>
                <button
                  type="button"
                  onClick={shareConversation}
                  className="neubrut flex items-center gap-1 rounded-md bg-card px-2 py-1 font-semibold"
                  title="Share link"
                >
                  <Share2 className="h-3 w-3" />
                  {t(lang, "share")}
                </button>
              </div>
            </div>
          </div>
        )}
        <div className="mx-auto flex max-w-4xl items-end gap-2 px-4 py-3">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={hasConversation ? "Hỏi tiếp..." : "Hỏi về pháp luật Việt Nam..."}
            rows={2}
            disabled={isLoading}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e as unknown as FormEvent);
              }
            }}
            className="flex-1 resize-none rounded-lg bg-background px-3 py-2.5 text-sm placeholder:text-muted-foreground focus:outline-none disabled:opacity-50"
            style={{
              border: "1.5px solid rgb(var(--ink))",
              boxShadow: "2px 2px 0 0 rgb(var(--ink))",
            }}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="neubrut inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-bold text-primary-foreground disabled:opacity-50"
          >
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            Gửi
          </button>
          {hasConversation && (
            <button
              type="button"
              onClick={clearChat}
              disabled={isLoading}
              className="neubrut rounded-lg bg-accent px-3 py-2.5 text-xs font-semibold text-accent-foreground disabled:opacity-50"
              title="Xóa hội thoại + bắt đầu session mới"
            >
              New chat
            </button>
          )}
        </div>
      </form>
    </div>
  );
}

function EmptyState({ onPickQuestion }: { onPickQuestion: (q: string) => void }) {
  const [activeDomain, setActiveDomain] = useState<(typeof DOMAINS)[number]["id"]>("all");
  const filtered = SUGGESTED_QUESTIONS.filter(
    (q) => activeDomain === "all" || q.domain === activeDomain,
  );
  return (
    <div className="m-auto w-full max-w-3xl space-y-6 py-8 text-center">
      <div>
        <div
          className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-accent/30"
          style={{ border: "1.5px solid rgb(var(--ink))" }}
        >
          <Sparkles className="h-7 w-7 text-primary" />
        </div>
        <h2 className="font-serif text-3xl font-semibold tracking-tight">
          Hỏi gì về <span className="text-primary">pháp luật Việt Nam</span>?
        </h2>
        <p className="mt-3 text-sm text-muted-foreground">
          Tập trung 3 ngành ·{" "}
          <span className="font-medium text-foreground">Lao động</span> ·{" "}
          <span className="font-medium text-foreground">Hình sự</span> ·{" "}
          <span className="font-medium text-foreground">Doanh nghiệp & Đầu tư</span>
        </p>
      </div>

      <div className="flex flex-wrap items-center justify-center gap-2">
        {DOMAINS.map((d) => (
          <button
            key={d.id}
            onClick={() => setActiveDomain(d.id)}
            className={`neubrut rounded-full px-3 py-1 text-xs font-semibold transition ${
              activeDomain === d.id ? "bg-primary text-primary-foreground" : "bg-card text-foreground"
            }`}
          >
            {d.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {filtered.map((s, i) => (
          <button
            key={i}
            onClick={() => onPickQuestion(s.question)}
            className="neubrut group flex flex-col items-start gap-2 rounded-xl bg-card p-4 text-left"
          >
            <div className="flex items-center gap-2">
              <span className="text-xl">{s.icon}</span>
              <span className="text-[10px] font-semibold uppercase tracking-widest text-accent">
                {s.title}
              </span>
            </div>
            <p className="text-sm font-medium leading-snug group-hover:text-primary">
              {s.question}
            </p>
          </button>
        ))}
      </div>

      <div
        className="mx-auto max-w-xl rounded-lg bg-accent/15 px-4 py-3 text-xs text-foreground/80"
        style={{ border: "1.5px solid rgb(var(--ink))" }}
      >
        💡 Hệ thống <strong>nhớ ngữ cảnh hội thoại</strong> — hỏi tiếp như{" "}
        <span className="font-serif italic">&ldquo;Còn trường hợp khác thì sao?&rdquo;</span>{" "}
        hoặc{" "}
        <span className="font-serif italic">&ldquo;Giải thích cụ thể hơn&rdquo;</span>
      </div>
    </div>
  );
}

function FollowUpSuggestions({
  questions,
  onPick,
}: {
  questions?: string[];
  onPick: (q: string) => void;
}) {
  const list = questions && questions.length > 0 ? questions : FOLLOW_UP_HINTS;
  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 px-1">
      <span className="text-xs font-medium text-muted-foreground">Gợi ý hỏi tiếp:</span>
      {list.map((q) => (
        <button
          key={q}
          onClick={() => onPick(q)}
          className="neubrut inline-flex items-center gap-1.5 rounded-full bg-card px-3 py-1 text-xs font-medium text-foreground"
        >
          <Sparkles className="h-3 w-3 text-accent" />
          {q}
        </button>
      ))}
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      console.error("Copy failed:", e);
    }
  }
  return (
    <button
      onClick={handleCopy}
      className="inline-flex items-center gap-1 rounded-md border border-border bg-card px-2 py-0.5 text-xs text-muted-foreground transition hover:bg-muted hover:text-foreground"
      title="Copy câu trả lời"
    >
      {copied ? (
        <>
          <Check className="h-3 w-3 text-green-600" />
          Đã copy
        </>
      ) : (
        <>
          <Copy className="h-3 w-3" />
          Copy
        </>
      )}
    </button>
  );
}

function MessageBubble({ message, onRetry }: { message: Message; onRetry?: (query: string) => void }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-5 py-4 ${
          isUser ? "bg-primary text-primary-foreground" : "bg-card"
        }`}
        style={
          isUser
            ? { border: "1.5px solid rgb(var(--ink))", boxShadow: "3px 3px 0 0 rgb(var(--ink))" }
            : { border: "1.5px solid rgb(var(--ink))" }
        }
      >
        {message.loading || (message.stage && !message.content) ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin text-primary" />
              <span className="font-medium">
                {message.stage
                  ? STAGE_LABEL[message.stage]
                  : "💭 Đang suy nghĩ câu trả lời..."}
              </span>
            </div>
            {/* Skeleton lines for visual feedback */}
            <div className="space-y-1.5 pt-1">
              <div className="h-2 w-3/4 animate-pulse rounded bg-muted/60"></div>
              <div className="h-2 w-1/2 animate-pulse rounded bg-muted/40"></div>
            </div>
          </div>
        ) : (
          <>
            {message.rewritten && message.rewritten !== "" && (
              <div
                className="mb-3 rounded-md bg-accent/20 px-3 py-1.5 text-xs italic"
                style={{ border: "1px solid rgb(var(--ink) / 0.4)" }}
              >
                <span className="font-medium not-italic">🔄 Hiểu là:</span>{" "}
                <span className="font-serif">{message.rewritten}</span>
              </div>
            )}
            {isUser ? (
              <div className="whitespace-pre-wrap text-sm font-medium leading-relaxed">
                {message.content}
              </div>
            ) : (
              <div
                className={`markdown-body text-[15px] ${
                  message.streaming ? "streaming-cursor" : ""
                }`}
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.content || (message.streaming ? "" : "(empty)")}
                </ReactMarkdown>
              </div>
            )}
          </>
        )}
        {!isUser && !message.loading && message.content && !message.streaming && (
          <div className="mt-3 flex items-center gap-2">
            <CopyButton text={message.content} />
            {message.errorKind && message.retryQuery && onRetry && (
              <button
                onClick={() => onRetry(message.retryQuery!)}
                className="inline-flex items-center gap-1 rounded-md border border-border bg-card px-2 py-0.5 text-xs font-semibold text-foreground transition hover:bg-muted"
                title="Thử lại câu hỏi này"
              >
                🔄 Thử lại
              </button>
            )}
          </div>
        )}
        {message.citations && message.citations.length > 0 && (
          <div
            className="mt-3 flex flex-wrap gap-2 pt-3"
            style={{ borderTop: "1px dashed rgb(var(--ink) / 0.25)" }}
          >
            {message.citations.map((c, i) => (
              <span
                key={i}
                className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-semibold ${
                  c.matched ? "bg-accent text-accent-foreground" : "bg-muted text-muted-foreground"
                }`}
                style={{
                  border: "1.5px solid rgb(var(--ink))",
                  boxShadow: "2px 2px 0 0 rgb(var(--ink))",
                }}
                title={c.law_title}
              >
                {c.law_id} · Điều {c.article_id}
                {c.khoan_id && ` · Khoản ${c.khoan_id}`}
              </span>
            ))}
          </div>
        )}
        {message.sources && message.sources.length > 0 && (
          <SourceList sources={message.sources} />
        )}
      </div>
    </div>
  );
}

// Item 7: expandable source preview
function SourceList({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  return (
    <div className="mt-2 text-xs text-muted-foreground">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 hover:text-foreground"
      >
        {open ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
        📚 {sources.length} nguồn tham khảo
      </button>
      {open && (
        <div className="mt-2 space-y-2">
          {sources.slice(0, 5).map((s, i) => (
            <div
              key={i}
              className="rounded-lg border border-border/40 bg-muted/30 overflow-hidden"
              style={{ boxShadow: "1px 1px 0 0 rgb(var(--ink) / 0.15)" }}
            >
              <button
                onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
                className="flex w-full items-start justify-between gap-2 p-2 text-left hover:bg-muted/50"
              >
                <div>
                  <span className="font-semibold text-foreground">
                    {s.law_id} · Điều {s.article_id}
                  </span>
                  {s.law_title && (
                    <span className="ml-1 text-muted-foreground">— {s.law_title.slice(0, 40)}</span>
                  )}
                  <span className="ml-1 rounded bg-accent/20 px-1 text-[10px] font-medium text-accent-foreground">
                    {(s.score * 100).toFixed(0)}%
                  </span>
                </div>
                {expandedIdx === i ? <ChevronUp className="h-3 w-3 shrink-0" /> : <ChevronDown className="h-3 w-3 shrink-0" />}
              </button>
              {expandedIdx === i && (
                <div className="border-t border-border/30 bg-background/50 p-2 text-xs leading-relaxed text-foreground/80">
                  {s.text}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
