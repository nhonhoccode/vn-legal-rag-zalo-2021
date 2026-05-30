export interface HistoryEntry {
  sessionId: string;
  firstQuery: string;
  timestamp: number;
  queryCount: number;
}

const KEY = "vn-legal-history";
const MAX = 20;

export function saveSession(sessionId: string, firstQuery: string): void {
  if (typeof window === "undefined" || !sessionId) return;
  const entries = loadHistory();
  const existing = entries.findIndex((e) => e.sessionId === sessionId);
  if (existing >= 0) {
    entries[existing].queryCount += 1;
  } else {
    entries.unshift({ sessionId, firstQuery: firstQuery.slice(0, 80), timestamp: Date.now(), queryCount: 1 });
  }
  localStorage.setItem(KEY, JSON.stringify(entries.slice(0, MAX)));
}

export function loadHistory(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "[]") as HistoryEntry[];
  } catch {
    return [];
  }
}

export function clearHistory(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(KEY);
}

export function formatRelative(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "Vừa xong";
  if (mins < 60) return `${mins} phút trước`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs} giờ trước`;
  return new Date(ts).toLocaleDateString("vi-VN");
}
