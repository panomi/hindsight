import { Check, Play, X } from "lucide-react";
import type { ResultGroup } from "../store/investigationStore";

type FrameItem = {
  frame_id?: string; detection_id?: string; video_id?: string;
  timestamp_seconds?: number; filepath?: string; score?: number;
  bbox?: { x1: number; y1: number; x2: number; y2: number };
  text?: string; class_name?: string; instance_id?: number;
};

function pickItems(g: ResultGroup): FrameItem[] {
  // Tool payloads vary; pick a sensible array
  const p = g.payload || {};
  return p.results || p.events || p.matches || [];
}

export function FrameGrid({
  group, selectable = false,
  selected, onToggle, onPlay,
}: {
  group: ResultGroup;
  selectable?: boolean;
  /**
   * Selection state.  We accept either a Map (with `.get(id)` returning the
   * confirm/reject choice — used by confirmation prompts to render the right
   * colour) or a Set-like object that only exposes `.has` (legacy callers).
   */
  selected?: { has(id: string): boolean; get?(id: string): "confirm" | "reject" | undefined };
  onToggle?: (id: string, kind: "confirm" | "reject") => void;
  onPlay?: (videoId: string, startSec: number) => void;
}) {
  const items = pickItems(group).slice(0, 24);
  if (items.length === 0) {
    return <div className="text-sm text-neutral-500">No items in this result.</div>;
  }
  // When the grid is rendered as a confirmation prompt (selectable=true) it lives
  // inside the narrow chat panel — cap columns at 2 so buttons have room.  In the
  // wide results pane (selectable=false) we keep the denser 4-up layout.
  const gridCols = selectable
    ? "grid-cols-2"
    : "grid-cols-2 sm:grid-cols-3 md:grid-cols-4";
  return (
    <div className={`grid ${gridCols} gap-2`}>
      {items.map((it, i) => {
        const id = it.detection_id || it.frame_id || String(i);
        // Prefer Map.get when available so we can colour confirm vs reject;
        // fall back to has() for any Set-like caller.
        const sel: "confirm" | "reject" | undefined =
          selected?.get?.(id) ?? (selected?.has(id) ? "confirm" : undefined);
        const ringClass =
          sel === "confirm" ? "ring-2 ring-emerald-400"
          : sel === "reject" ? "ring-2 ring-red-400"
          : "";
        return (
          <div key={id} className={
            "glass overflow-hidden rounded-xl flex flex-col text-xs " +
            ringClass
          }>
            <div
              className="aspect-video bg-neutral-100 relative overflow-hidden group/thumb"
              onClick={() => {
                if (onPlay && it.video_id && it.timestamp_seconds !== undefined) {
                  onPlay(it.video_id, it.timestamp_seconds);
                }
              }}
              style={{ cursor: onPlay && it.video_id ? "pointer" : undefined }}
            >
              {it.frame_id ? (
                <img
                  src={`/api/frames/${it.frame_id}/image`}
                  alt={`frame at ${it.timestamp_seconds?.toFixed(1)}s`}
                  className="absolute inset-0 w-full h-full object-cover"
                  loading="lazy"
                />
              ) : it.video_id && it.timestamp_seconds !== undefined ? (
                <video
                  src={`/api/videos/${it.video_id}/file#t=${it.timestamp_seconds}`}
                  preload="metadata"
                  muted
                  playsInline
                  className="absolute inset-0 w-full h-full object-cover"
                />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center text-neutral-400 text-xs">
                  {it.timestamp_seconds !== undefined
                    ? `${it.timestamp_seconds.toFixed(1)}s`
                    : "frame"}
                </div>
              )}
              {/* play overlay — visible on hover when video_id is available */}
              {onPlay && it.video_id && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/0 group-hover/thumb:bg-black/30 transition-colors">
                  <div className="opacity-0 group-hover/thumb:opacity-100 transition-opacity bg-white/90 rounded-full p-2 shadow">
                    <Play size={16} className="text-coral-600 fill-coral-600" />
                  </div>
                </div>
              )}
              {it.bbox && (
                <div
                  className="absolute border-2 border-coral-500"
                  style={{
                    left: `${it.bbox.x1 * 100}%`,
                    top: `${it.bbox.y1 * 100}%`,
                    width: `${(it.bbox.x2 - it.bbox.x1) * 100}%`,
                    height: `${(it.bbox.y2 - it.bbox.y1) * 100}%`,
                  }}
                />
              )}
            </div>
            <div className="p-2 flex flex-col gap-1">
              <div className="flex items-start justify-between gap-1 text-neutral-600">
                <span
                  className="line-clamp-3 break-words leading-snug"
                  title={it.text ?? it.class_name}
                >
                  {it.class_name ?? it.text ?? "result"}
                </span>
                {it.score !== undefined && (
                  <span className="text-coral-600 shrink-0">{it.score.toFixed(2)}</span>
                )}
              </div>
              {it.instance_id !== undefined && (
                <div className="text-neutral-400">track #{it.instance_id}</div>
              )}
              {selectable && (
                <div className="flex items-stretch gap-1.5 mt-1">
                  <button
                    type="button"
                    title="Confirm match"
                    aria-label="Confirm match"
                    onClick={() => onToggle?.(id, "confirm")}
                    className={"flex-1 min-w-0 py-1.5 rounded-lg inline-flex items-center justify-center gap-1 font-medium transition-colors " +
                      (sel === "confirm"
                        ? "bg-emerald-500 text-white"
                        : "bg-emerald-50 text-emerald-700 hover:bg-emerald-100")}
                  >
                    <Check size={14} />
                    <span className="truncate">Yes</span>
                  </button>
                  <button
                    type="button"
                    title="Reject — not a match"
                    aria-label="Reject — not a match"
                    onClick={() => onToggle?.(id, "reject")}
                    className={"flex-1 min-w-0 py-1.5 rounded-lg inline-flex items-center justify-center gap-1 font-medium transition-colors " +
                      (sel === "reject"
                        ? "bg-red-500 text-white"
                        : "bg-red-50 text-red-700 hover:bg-red-100")}
                  >
                    <X size={14} />
                    <span className="truncate">No</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
