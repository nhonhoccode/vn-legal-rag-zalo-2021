export type Lang = "vi" | "en";

const translations = {
  vi: {
    appTitle: "Pháp Luật Việt Nam",
    appSubtitle: "Tra cứu qua AI · Lao động · Hình sự · Doanh nghiệp",
    logout: "Đăng xuất",
    retry: "Thử lại",
    copy: "Copy",
    copied: "Đã copy",
    thinking: "Đang suy nghĩ câu trả lời...",
    followUp: "Gợi ý hỏi tiếp:",
    sources: "nguồn tham khảo",
    streamOn: "Stream BẬT",
    streamOff: "Stream TẮT",
    save: "Lưu",
    share: "Chia sẻ",
    shareCopied: "Đã copy link!",
    newChat: "Chat mới",
    sendPlaceholder: "Hỏi về pháp luật Việt Nam...",
    emptyTitle: "Hỏi bất kỳ câu hỏi pháp lý nào",
    emptySubtitle: "Tôi sẽ tra cứu văn bản pháp luật Việt Nam và trả lời kèm trích dẫn điều khoản cụ thể.",
    filterAll: "Tất cả",
    filterLabor: "Lao động",
    filterCriminal: "Hình sự",
    filterBusiness: "Doanh nghiệp",
    errorTimeout: "Server đang khởi động (lần đầu tải model ~2 phút). Thử lại sau ít phút nhé.",
    errorNetwork: "Mất kết nối mạng. Kiểm tra wifi và thử lại.",
    errorGeneric: "Lỗi",
    retryBtn: "🔄 Thử lại",
    retrying: "🔄 Đang thử lại...",
    // Dashboard
    dashTitle: "Hồ sơ",
    dashTheme: "Giao diện",
    dashThemeDark: "Tối",
    dashThemeLight: "Sáng",
    dashLang: "Ngôn ngữ",
    dashHistory: "Lịch sử chat",
    dashNoHistory: "Chưa có cuộc hội thoại nào.",
    dashStats: "Thống kê phiên",
    dashQueries: "câu hỏi",
    footerLabel: "Pháp luật VN qua RAG",
    sessionInfo: "Hội thoại đang nhớ ngữ cảnh",
    question: "câu hỏi",
  },
  en: {
    appTitle: "Vietnam Legal",
    appSubtitle: "AI-powered · Labor · Criminal · Business",
    logout: "Logout",
    retry: "Retry",
    copy: "Copy",
    copied: "Copied",
    thinking: "Thinking...",
    followUp: "Follow-up suggestions:",
    sources: "sources",
    streamOn: "Stream ON",
    streamOff: "Stream OFF",
    save: "Save",
    share: "Share",
    shareCopied: "Link copied!",
    newChat: "New chat",
    sendPlaceholder: "Ask about Vietnamese law...",
    emptyTitle: "Ask any legal question",
    emptySubtitle: "I will look up Vietnamese legal texts and answer with specific article citations.",
    filterAll: "All",
    filterLabor: "Labor",
    filterCriminal: "Criminal",
    filterBusiness: "Business",
    errorTimeout: "Server is starting up (first load ~2 min). Try again in a moment.",
    errorNetwork: "Network connection lost. Check your wifi and retry.",
    errorGeneric: "Error",
    retryBtn: "🔄 Retry",
    retrying: "🔄 Retrying...",
    dashTitle: "Profile",
    dashTheme: "Theme",
    dashThemeDark: "Dark",
    dashThemeLight: "Light",
    dashLang: "Language",
    dashHistory: "Chat history",
    dashNoHistory: "No conversations yet.",
    dashStats: "Session stats",
    dashQueries: "queries",
    footerLabel: "Vietnam Legal via RAG",
    sessionInfo: "Conversation in context",
    question: "question",
  },
} as const;

export type Translations = typeof translations.vi;

export function t(lang: Lang, key: keyof Translations): string {
  return translations[lang][key] as string;
}

export function getStoredLang(): Lang {
  if (typeof window === "undefined") return "vi";
  return (localStorage.getItem("vn-legal-lang") as Lang) ?? "vi";
}

export function setStoredLang(lang: Lang): void {
  localStorage.setItem("vn-legal-lang", lang);
}
