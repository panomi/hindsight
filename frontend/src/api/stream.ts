/**
 * Subscribe to an investigation's SSE event stream.
 * Returns a cleanup function. Each event is dispatched to onEvent.
 */
export type AgentEvent =
  | { event: "tool_start"; data: { id: string; tool: string; params: any } }
  | { event: "tool_result"; data: { id: string; tool: string; summary: string; count: number; ui_payload: any; duration_ms: number } }
  | { event: "results_update"; data: any }
  | { event: "confirmation_request"; data: { confirmation_id: string; mode: "frames" | "instances" | "events"; question: string; items: any[] } }
  | { event: "subject_registered"; data: { subject_id: string; label: string } }
  | { event: "message"; data: { role: "assistant"; content: string } }
  | { event: "error"; data: { message: string } }
  | { event: "ping"; data: any }
  | { event: "done"; data: {} };

export function subscribeToInvestigation(
  investigationId: string,
  onEvent: (e: AgentEvent) => void
): () => void {
  const url = `/api/investigations/${investigationId}/stream`;
  const es = new EventSource(url);
  const handlers: Array<[string, (ev: MessageEvent) => void]> = [];
  const evNames = ["tool_start", "tool_result", "results_update",
                   "confirmation_request", "subject_registered",
                   "message", "error", "ping", "done"] as const;
  for (const name of evNames) {
    const h = (ev: MessageEvent) => {
      let data: any = {};
      try { data = JSON.parse(ev.data); } catch {}
      onEvent({ event: name, data } as AgentEvent);
      // Do NOT close on "done" — keep the EventSource open so follow-up
      // messages can stream events without a reconnect.
    };
    es.addEventListener(name, h as any);
    handlers.push([name, h]);
  }
  return () => {
    for (const [name, h] of handlers) es.removeEventListener(name, h as any);
    es.close();
  };
}
