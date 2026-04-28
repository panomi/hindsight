import { Play } from "lucide-react";
import { useMemo } from "react";
import type { ResultGroup } from "../store/investigationStore";

type Marker = { ts: number; endTs?: number; tool: string; videoId: string };

function fmtTime(s: number) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`
    : `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

/** Multi-track horizontal timeline. One row per unique video_id. */
export function TimelineBar({
  resultGroups,
  videoDurations = {},
  onPlay,
}: {
  resultGroups: ResultGroup[];
  durationSec?: number;       // kept for API compat, ignored when videoDurations provided
  videoDurations?: Record<string, number>;  // video_id → actual duration_seconds
  onPlay?: (videoId: string, startSec: number, endSec?: number) => void;
}) {
  // Prefer temporal_cluster groups (clean event spans) over raw frame hits.
  // This keeps the timeline readable: 3-5 event bands instead of 20+ hairlines.
  const { markers, durationByVideo } = useMemo(() => {
    const clusterGroups = resultGroups.filter((g) => g.tool === "temporal_cluster");
    const sourceGroups = clusterGroups.length > 0 ? clusterGroups : resultGroups;

    const marks: Marker[] = [];
    const maxByVideo: Record<string, number> = {};

    for (const g of sourceGroups) {
      const items = (g.payload?.results || g.payload?.events || []) as any[];
      for (const it of items) {
        const ts = it.timestamp_seconds ?? it.start_seconds;
        const endTs = it.end_seconds;
        const vid = it.video_id ?? "unknown";
        if (typeof ts === "number") {
          marks.push({ ts, endTs, tool: g.tool, videoId: vid });
          const dur = endTs ?? ts;
          if (dur > (maxByVideo[vid] ?? 0)) maxByVideo[vid] = dur;
        }
      }
    }
    return { markers: marks, durationByVideo: maxByVideo };
  }, [resultGroups]);

  if (markers.length === 0) return null;

  const videoIds = Array.from(new Set(markers.map((m) => m.videoId)));
  const multiVideo = videoIds.length > 1;

  return (
    <div className="glass p-3 space-y-2">
      <div className="text-xs text-neutral-500 font-medium">
        Timeline · {markers.length} hit{markers.length !== 1 ? "s" : ""}
        {multiVideo && ` · ${videoIds.length} videos`}
      </div>

      {videoIds.map((vid, rowIdx) => {
        const rowMarkers = markers.filter((m) => m.videoId === vid);
        // Prefer actual video duration; fall back to max hit timestamp + 10% buffer
        const hitMax = durationByVideo[vid] ?? 0;
        const dur = videoDurations[vid]
          ? videoDurations[vid]
          : Math.max(hitMax * 1.1, 30);

        return (
          <div key={vid}>
            {multiVideo && (
              <div className="text-[10px] text-neutral-400 mb-0.5">
                Video {rowIdx + 1}
                <span className="ml-1 font-mono text-neutral-300">…{vid.slice(-8)}</span>
              </div>
            )}
            <div className="relative h-6 bg-neutral-100/60 rounded-lg overflow-visible">
              {rowMarkers.map((m, i) => {
                const pct = (m.ts / dur) * 100;
                const widthPct =
                  m.endTs !== undefined
                    ? Math.max(((m.endTs - m.ts) / dur) * 100, 0.5)
                    : 0.5;

                return (
                  <button
                    key={i}
                    title={`${m.tool} @ ${fmtTime(m.ts)}${m.endTs !== undefined ? " – " + fmtTime(m.endTs) : ""}`}
                    onClick={() => onPlay?.(m.videoId, m.ts, m.endTs)}
                    className="absolute top-0 bottom-0 coral-grad hover:opacity-80 transition-opacity group/marker focus:outline-none rounded-sm"
                    style={{
                      left: `${pct}%`,
                      width: `max(${widthPct}%, 3px)`,
                    }}
                  >
                    {/* tooltip on hover */}
                    <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-1.5 py-0.5 bg-neutral-800 text-white rounded text-[10px] whitespace-nowrap opacity-0 group-hover/marker:opacity-100 transition-opacity pointer-events-none z-10 shadow">
                      {fmtTime(m.ts)}
                      {onPlay && (
                        <Play size={8} className="inline ml-1 fill-white text-white" />
                      )}
                    </span>
                  </button>
                );
              })}
            </div>
            {/* time axis labels */}
            <div className="flex justify-between text-[9px] text-neutral-300 mt-0.5 px-0.5">
              <span>0:00</span>
              <span>{fmtTime(dur / 2)}</span>
              <span>{fmtTime(dur)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
