import { Check, X } from "lucide-react";

/** Per-track confirmation strip — used for `mode="instances"` confirmations.
 *
 * Renders a real thumbnail when the item carries either a `frame_id` (uses
 * the frame image API) or `video_id + timestamp_seconds` (uses the video
 * file with a #t fragment).  Falls back to the text label only when neither
 * is available.
 */
export function InstanceGallery({
  items, selected, onToggle,
}: {
  items: Array<{
    id: string;
    tool?: string;
    preview?: string;
    track_id?: number;
    frame_id?: string;
    video_id?: string;
    timestamp_seconds?: number;
    thumbnail_url?: string;
    meta?: any;
  }>;
  selected: Map<string, "confirm" | "reject" | undefined>;
  onToggle: (id: string, kind: "confirm" | "reject") => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {items.map(it => {
        const sel = selected.get(it.id);
        return (
          <div key={it.id} className={
            "glass overflow-hidden rounded-xl text-xs " +
            (sel === "confirm" ? "ring-2 ring-emerald-400" : sel === "reject" ? "ring-2 ring-red-400" : "")
          }>
            <div className="aspect-video bg-neutral-100 relative overflow-hidden">
              {it.frame_id ? (
                <img
                  src={`/api/frames/${it.frame_id}/image`}
                  alt={it.preview || `track #${it.track_id ?? ""}`}
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
                <div className="absolute inset-0 flex items-center justify-center text-neutral-400">
                  {it.preview || `track #${it.track_id ?? ""}`}
                </div>
              )}
              {it.track_id !== undefined && (
                <div className="absolute bottom-1 left-1 bg-black/60 text-white text-[10px] px-1.5 py-0.5 rounded">
                  track #{it.track_id}
                </div>
              )}
            </div>
            <div className="p-1.5 flex items-stretch gap-1.5">
              <button
                type="button"
                title="Confirm match"
                aria-label="Confirm match"
                onClick={() => onToggle(it.id, "confirm")}
                className={"flex-1 min-w-0 py-1.5 rounded-lg inline-flex items-center justify-center gap-1 font-medium transition-colors " +
                  (sel === "confirm"
                    ? "bg-emerald-500 text-white"
                    : "bg-emerald-50 text-emerald-700 hover:bg-emerald-100")}
              >
                <Check size={14} />
                <span className="hidden md:inline">Yes</span>
              </button>
              <button
                type="button"
                title="Reject — not a match"
                aria-label="Reject — not a match"
                onClick={() => onToggle(it.id, "reject")}
                className={"flex-1 min-w-0 py-1.5 rounded-lg inline-flex items-center justify-center gap-1 font-medium transition-colors " +
                  (sel === "reject"
                    ? "bg-red-500 text-white"
                    : "bg-red-50 text-red-700 hover:bg-red-100")}
              >
                <X size={14} />
                <span className="hidden md:inline">No</span>
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
