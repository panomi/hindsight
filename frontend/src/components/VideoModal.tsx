import { X } from "lucide-react";

export type VideoPlay = {
  videoId: string;
  startSec: number;
  endSec?: number;
  label?: string;
};

function fmtTime(s: number) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`
    : `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

export function VideoModal({
  play,
  onClose,
}: {
  play: VideoPlay;
  onClose: () => void;
}) {
  const clipStart = Math.max(0, play.startSec - 2);
  const src = `/api/videos/${play.videoId}/file#t=${clipStart}${play.endSec !== undefined ? `,${play.endSec + 2}` : ""}`;

  const timeLabel = play.endSec !== undefined
    ? `${fmtTime(play.startSec)} – ${fmtTime(play.endSec)}`
    : fmtTime(play.startSec);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-2xl shadow-2xl overflow-hidden max-w-4xl w-full mx-4 flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-neutral-100">
          <div className="text-sm font-medium text-neutral-700">
            {play.label && <span className="mr-2 text-neutral-500">{play.label}</span>}
            <span className="font-mono">{timeLabel}</span>
          </div>
          <button
            onClick={onClose}
            className="text-neutral-400 hover:text-neutral-800 p-1 rounded-lg hover:bg-neutral-100"
          >
            <X size={16} />
          </button>
        </div>

        {/* video — key forces remount when clip changes so autoPlay fires correctly */}
        <video
          key={`${play.videoId}-${play.startSec}`}
          className="w-full max-h-[70vh] bg-black"
          controls
          autoPlay
          src={src}
        />

      </div>
    </div>
  );
}
