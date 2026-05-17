"use client";

import { useState, useRef, useEffect, FormEvent } from "react";
import {
  Send,
  Loader2,
  Sparkles,
  MessageSquare,
  Copy,
  Check,
  Download,
  Zap,
  ZapOff,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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

export default function ChatClient() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [useStreaming, setUseStreaming] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

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

    try {
      if (useStreaming) {
        await sendStreaming(query, assistantMsg.id);
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
        ? "⏱️ Server đang khởi động (lần đầu tải model ~2 phút) hoặc câu hỏi quá phức tạp. Bạn thử lại sau ít phút nhé."
        : kind === "network"
        ? "🌐 Mất kết nối mạng. Kiểm tra wifi và thử lại."
        : `Lỗi: ${msgStr}`;
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
            if (sid && !sessionId) setSessionId(sid);
            setMessages((m) =>
              m.map((msg) =>
                msg.id === assistantMsgId
                  ? { ...msg, citations, streaming: false, loading: false, stage: null }
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
  const showFollowUps =
    hasConversation && !isLoading && messages[messages.length - 1]?.role === "assistant";

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
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
            {showFollowUps && <FollowUpSuggestions onPick={(q) => sendQuery(q)} />}
          </>
        )}
      </div>

      <form onSubmit={handleSubmit} className="border-t-2 border-ink bg-card">
        {hasConversation && sessionId && (
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
                  Save
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

function FollowUpSuggestions({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 px-1">
      <span className="text-xs font-medium text-muted-foreground">Gợi ý hỏi tiếp:</span>
      {FOLLOW_UP_HINTS.map((q) => (
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
          <details className="mt-2 text-xs text-muted-foreground">
            <summary className="cursor-pointer select-none hover:text-foreground">
              📚 {message.sources.length} nguồn tham khảo
            </summary>
            <div className="mt-2 space-y-2">
              {message.sources.slice(0, 5).map((s, i) => (
                <div key={i} className="rounded border border-border/40 bg-muted/30 p-2">
                  <div className="font-medium">
                    {s.law_id} • Điều {s.article_id}
                    <span className="ml-1 text-muted-foreground">[{s.score.toFixed(3)}]</span>
                  </div>
                  <div className="mt-1 line-clamp-3 text-muted-foreground">{s.text}</div>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}
