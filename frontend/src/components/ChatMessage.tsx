import { Play } from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { ChatMessage as TMsg } from "../store/investigationStore";
import { ToolCallBadge } from "./ToolCallBadge";

// Matches /api/videos/<uuid>/file#t=<start>[,<end>] — with or without a leading http(s)://host
const VIDEO_URL_RE = /(?:https?:\/\/[^/]*)?\/api\/videos\/([0-9a-f-]+)\/file#t=([\d.]+)(?:,([\d.]+))?/i;

function parseClipHref(href: string | undefined): { videoId: string; startSec: number; endSec?: number } | null {
  if (!href) return null;
  const m = href.match(VIDEO_URL_RE);
  if (!m) return null;
  return {
    videoId: m[1],
    startSec: parseFloat(m[2]),
    endSec: m[3] !== undefined ? parseFloat(m[3]) : undefined,
  };
}

function AssistantMarkdown({
  text,
  onPlay,
}: {
  text: string;
  onPlay?: (videoId: string, startSec: number, endSec?: number) => void;
}) {
  return (
    <ReactMarkdown
      components={{
        h1: ({ children }) => <h1 className="text-base font-semibold mt-2 mb-1">{children}</h1>,
        h2: ({ children }) => <h2 className="text-sm font-semibold mt-2 mb-1">{children}</h2>,
        h3: ({ children }) => <h3 className="text-sm font-medium mt-1 mb-0.5">{children}</h3>,
        p: ({ children }) => <p className="mb-1 last:mb-0">{children}</p>,
        ul: ({ children }) => <ul className="list-disc list-inside mb-1 space-y-0.5">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal list-inside mb-1 space-y-0.5">{children}</ol>,
        li: ({ children }) => <li className="leading-snug">{children}</li>,
        strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
        em: ({ children }) => <em className="italic">{children}</em>,
        code: ({ children }) => (
          <code className="bg-neutral-100 text-coral-700 rounded px-1 py-0.5 text-xs font-mono">
            {children}
          </code>
        ),
        blockquote: ({ children }) => (
          <blockquote className="border-l-2 border-neutral-300 pl-3 text-neutral-500 italic my-1">
            {children}
          </blockquote>
        ),
        hr: () => <hr className="border-neutral-200 my-2" />,
        // Intercept video clip links — open modal instead of navigating
        a: ({ href, children }) => {
          const clip = parseClipHref(href);
          if (clip && onPlay) {
            return (
              <button
                onClick={() => onPlay(clip.videoId, clip.startSec, clip.endSec)}
                className="inline-flex items-center gap-1 text-coral-600 hover:text-coral-800 underline underline-offset-2 font-medium"
              >
                <Play size={12} className="fill-current" />
                {children}
              </button>
            );
          }
          // External / unknown links — open in new tab safely
          return (
            <a href={href} target="_blank" rel="noopener noreferrer"
               className="text-coral-600 hover:text-coral-800 underline underline-offset-2">
              {children}
            </a>
          );
        },
      }}
    >
      {text}
    </ReactMarkdown>
  );
}

export function ChatMessage({ m, onPlay }: { m: TMsg; onPlay?: (videoId: string, startSec: number, endSec?: number) => void }) {
  if (m.kind === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] coral-grad text-white rounded-2xl rounded-br-md px-4 py-2 text-sm shadow-md">
          {m.text}
        </div>
      </div>
    );
  }
  if (m.kind === "assistant") {
    return (
      <div className="flex">
        <div className="max-w-[85%] glass rounded-2xl rounded-bl-md px-4 py-3 text-sm">
          <AssistantMarkdown text={m.text} onPlay={onPlay} />
        </div>
      </div>
    );
  }
  // tool
  return (
    <div className="flex">
      <ToolCallBadge
        tool={m.tool} status={m.status}
        summary={m.summary} count={m.count} durationMs={m.durationMs}
      />
    </div>
  );
}
