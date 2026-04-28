import { create } from "zustand";

export type ChatMessage =
  | { kind: "user"; text: string; ts: number }
  | { kind: "assistant"; text: string; ts: number }
  | { kind: "tool"; id: string; tool: string; params: any; status: "running" | "done" | "error";
      summary?: string; count?: number; durationMs?: number; ts: number };

export type Subject = { id: string; label: string; ts: number };

export type ResultGroup = {
  id: string;          // tool_use id
  tool: string;
  payload: any;
};

export type ConfirmationRequest = {
  confirmationId: string;
  mode: "frames" | "instances" | "events";
  question: string;
  items: any[];
};

type State = {
  messages: ChatMessage[];
  resultGroups: ResultGroup[];
  subjects: Subject[];
  pendingConfirmation: ConfirmationRequest | null;
  agentBusy: boolean;
};

type Actions = {
  reset(): void;
  hydrateHistory(msgs: ChatMessage[]): void;
  pushUser(text: string): void;
  pushAssistant(text: string): void;
  upsertToolStart(id: string, tool: string, params: any): void;
  upsertToolResult(id: string, tool: string, summary: string, count: number, payload: any, durationMs: number): void;
  setPendingConfirmation(c: ConfirmationRequest | null): void;
  registerSubject(s: Subject): void;
  setAgentBusy(b: boolean): void;
};

export const useInvestigationStore = create<State & Actions>((set) => ({
  messages: [],
  resultGroups: [],
  subjects: [],
  pendingConfirmation: null,
  agentBusy: false,

  reset: () => set({ messages: [], resultGroups: [], subjects: [],
                     pendingConfirmation: null, agentBusy: false }),
  hydrateHistory: (msgs) => set({ messages: msgs }),
  pushUser: (text) => set((s) => ({ messages: [...s.messages, { kind: "user", text, ts: Date.now() }] })),
  pushAssistant: (text) => set((s) => ({
    messages: [...s.messages, { kind: "assistant", text, ts: Date.now() }],
  })),
  upsertToolStart: (id, tool, params) => set((s) => ({
    messages: [
      ...s.messages,
      { kind: "tool", id, tool, params, status: "running", ts: Date.now() },
    ],
  })),
  upsertToolResult: (id, tool, summary, count, payload, durationMs) => set((s) => ({
    messages: s.messages.map((m) =>
      m.kind === "tool" && m.id === id
        ? { ...m, status: "done", summary, count, durationMs }
        : m
    ),
    resultGroups: [...s.resultGroups, { id, tool, payload }],
  })),
  setPendingConfirmation: (c) => set({ pendingConfirmation: c }),
  registerSubject: (s) => set((st) => ({
    subjects: st.subjects.find(x => x.id === s.id) ? st.subjects : [...st.subjects, s],
  })),
  setAgentBusy: (b) => set({ agentBusy: b }),
}));
