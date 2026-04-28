import { Loader2, CheckCircle2, AlertCircle } from "lucide-react";

export function ToolCallBadge({
  tool, status, summary, count, durationMs,
}: { tool: string; status: "running" | "done" | "error"; summary?: string; count?: number; durationMs?: number }) {
  return (
    <div className={
      "inline-flex flex-col gap-0.5 rounded-xl px-3 py-2 text-xs " +
      (status === "running"
        ? "coral-grad text-white shadow-md"
        : status === "error"
          ? "bg-red-50 text-red-700 border border-red-200"
          : "bg-white/80 text-neutral-700 border border-neutral-200/60")
    }>
      <div className="flex items-center gap-1.5 font-medium">
        {status === "running"
          ? <Loader2 size={12} className="animate-spin" />
          : status === "error"
            ? <AlertCircle size={12} />
            : <CheckCircle2 size={12} className="text-emerald-600" />}
        <span>{tool}</span>
        {count !== undefined && status === "done" && (
          <span className="opacity-60">· {count} result{count === 1 ? "" : "s"}</span>
        )}
        {durationMs !== undefined && status === "done" && (
          <span className="opacity-60">· {durationMs}ms</span>
        )}
      </div>
      {summary && status === "done" && (
        <div className="opacity-70 leading-snug">{summary}</div>
      )}
    </div>
  );
}
