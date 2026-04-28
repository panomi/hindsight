import type { ChatMessage } from "../store/investigationStore";
import { api } from "./client";
import type { Investigation } from "./types";

export const listInvestigations = () => api<Investigation[]>("/api/investigations");
export const getInvestigation = (id: string) => api<Investigation>(`/api/investigations/${id}`);
export const createInvestigation = (collection_id: string, title: string) =>
  api<Investigation>("/api/investigations", {
    method: "POST", json: { collection_id, title },
  });
export const sendMessage = (investigationId: string, content: string) =>
  api<{ ok: true }>(`/api/investigations/${investigationId}/message`, {
    method: "POST", json: { content },
  });
export const getInvestigationHistory = (id: string) =>
  api<{ events: ChatMessage[] }>(`/api/investigations/${id}/history`);

export const postConfirmation = (investigationId: string, payload: {
  confirmation_id: string; confirmed_ids: string[]; rejected_ids: string[];
}) =>
  api<{ ok: true }>(`/api/investigations/${investigationId}/confirm`, {
    method: "POST", json: payload,
  });
