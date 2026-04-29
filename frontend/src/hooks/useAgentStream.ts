import { useEffect } from "react";
import { subscribeToInvestigation } from "../api/stream";
import { useInvestigationStore } from "../store/investigationStore";

export function useAgentStream(investigationId: string | undefined) {
  const upsertToolStart = useInvestigationStore((s) => s.upsertToolStart);
  const upsertToolResult = useInvestigationStore((s) => s.upsertToolResult);
  const pushAssistant = useInvestigationStore((s) => s.pushAssistant);
  const setPendingConfirmation = useInvestigationStore((s) => s.setPendingConfirmation);
  const registerSubject = useInvestigationStore((s) => s.registerSubject);
  const setAgentBusy = useInvestigationStore((s) => s.setAgentBusy);

  useEffect(() => {
    if (!investigationId) return;
    // Do NOT set agentBusy here — it is set by onSend in Investigate.tsx
    // when the user actually submits a message.
    const unsubscribe = subscribeToInvestigation(investigationId, (e) => {
      switch (e.event) {
        case "tool_start":
          upsertToolStart(e.data.id, e.data.tool, e.data.params);
          break;
        case "tool_result":
          upsertToolResult(
            e.data.id, e.data.tool, e.data.summary,
            e.data.count, e.data.ui_payload, e.data.duration_ms
          );
          // Also surface a registered subject if register_subject succeeded
          if (e.data.tool === "register_subject" && e.data.ui_payload?.subject_id) {
            registerSubject({
              id: e.data.ui_payload.subject_id,
              label: e.data.ui_payload.label,
              ts: Date.now(),
            });
          }
          break;
        case "message":
          pushAssistant(e.data.content);
          break;
        case "confirmation_request":
          setPendingConfirmation({
            confirmationId: e.data.confirmation_id,
            mode: e.data.mode,
            question: e.data.question,
            items: e.data.items,
          });
          break;
        case "confirmation_resolved":
          // Backend says this confirmation is no longer pending (submitted,
          // skipped, or timed out).  Clear the popup if it's still showing
          // the same id — guards against multi-tab and "user moved on".
          {
            const cur = useInvestigationStore.getState().pendingConfirmation;
            if (!cur || cur.confirmationId === e.data.confirmation_id) {
              setPendingConfirmation(null);
            }
          }
          break;
        case "done":
        case "error":
          setAgentBusy(false);
          break;
      }
    });
    return () => { unsubscribe(); setAgentBusy(false); };
  }, [investigationId, upsertToolStart, upsertToolResult, pushAssistant,
      setPendingConfirmation, registerSubject, setAgentBusy]);
}
