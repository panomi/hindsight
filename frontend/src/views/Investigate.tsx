import { useMutation, useQuery } from "@tanstack/react-query";
import { Send } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { listVideos } from "../api/collections";
import { getInvestigation, getInvestigationHistory, postConfirmation, sendMessage } from "../api/investigations";
import { ChatMessage } from "../components/ChatMessage";
import { FrameGrid } from "../components/FrameGrid";
import { InstanceGallery } from "../components/InstanceGallery";
import { SubjectChip } from "../components/SubjectChip";
import { TimelineBar } from "../components/TimelineBar";
import { VideoModal, type VideoPlay } from "../components/VideoModal";
import { Button } from "../components/ui/Button";
import { GlassCard } from "../components/ui/GlassCard";
import { useAgentStream } from "../hooks/useAgentStream";
import { useInvestigationStore } from "../store/investigationStore";

export default function Investigate() {
  const { id } = useParams<{ id: string }>();
  const inv = useQuery({
    queryKey: ["investigation", id], queryFn: () => getInvestigation(id!),
    enabled: !!id,
  });
  const videos = useQuery({
    queryKey: ["videos", inv.data?.collection_id],
    queryFn: () => listVideos(inv.data!.collection_id),
    enabled: !!inv.data?.collection_id,
  });
  const videoDurations = useMemo(() => {
    const map: Record<string, number> = {};
    for (const v of videos.data ?? []) {
      if (v.duration_seconds != null) map[v.id] = v.duration_seconds;
    }
    return map;
  }, [videos.data]);
  const { messages, resultGroups, subjects, pendingConfirmation, agentBusy,
          pushUser, setPendingConfirmation, setAgentBusy, reset, hydrateHistory } = useInvestigationStore();

  useEffect(() => {
    reset();
    if (!id) return;
    getInvestigationHistory(id)
      .then(({ events }) => { if (events.length > 0) hydrateHistory(events); })
      .catch(() => {}); // silently ignore — new investigation has no history yet
  }, [id]); // eslint-disable-line react-hooks/exhaustive-deps
  useAgentStream(id);

  const [input, setInput] = useState("");
  const [videoPlay, setVideoPlay] = useState<VideoPlay | null>(null);
  const onPlay = useCallback((videoId: string, startSec: number, endSec?: number) => {
    setVideoPlay({ videoId, startSec, endSec });
  }, []);

  const send = useMutation({
    mutationFn: (text: string) => sendMessage(id!, text),
    onSuccess: () => setInput(""),
  });


  // Confirmation panel state
  const [picks, setPicks] = useState(new Map<string, "confirm" | "reject" | undefined>());
  useEffect(() => { setPicks(new Map()); }, [pendingConfirmation?.confirmationId]);

  const submitConfirmation = useMutation({
    mutationFn: () => {
      if (!pendingConfirmation) throw new Error("none");
      const confirmed_ids: string[] = [];
      const rejected_ids: string[] = [];
      picks.forEach((v, k) => {
        if (v === "confirm") confirmed_ids.push(k);
        if (v === "reject") rejected_ids.push(k);
      });
      return postConfirmation(id!, {
        confirmation_id: pendingConfirmation.confirmationId,
        confirmed_ids, rejected_ids,
      });
    },
    onSuccess: () => setPendingConfirmation(null),
  });

  const onSend = () => {
    const t = input.trim();
    if (!t) return;
    pushUser(t);
    setAgentBusy(true);
    send.mutate(t);
  };


  return (
    <div className="grid grid-cols-1 lg:grid-cols-5 gap-4 max-w-7xl mx-auto h-[calc(100vh-7rem)]">
      {/* LEFT — chat */}
      <GlassCard className="lg:col-span-2 flex flex-col gap-3 min-h-0">
        <div className="flex items-center justify-between">
          <div className="font-semibold truncate">{inv.data?.title ?? "Investigation"}</div>
          {agentBusy && (
            <span className="text-xs text-coral-600 inline-flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full coral-grad animate-pulse" /> agent
            </span>
          )}
        </div>
        {subjects.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {subjects.map(s => <SubjectChip key={s.id} label={s.label} />)}
          </div>
        )}
        <div className="flex-1 overflow-y-auto flex flex-col gap-2 pr-1">
          {messages.length === 0 && (
            <div className="text-sm text-neutral-500">
              Ask a question to begin. Try: <em>"find a person reading a newspaper"</em>,
              {" "}<em>"when do these two people appear together"</em>, or
              {" "}<em>"license plate ABC123"</em>.
            </div>
          )}
          {messages.map((m, i) => <ChatMessage key={i} m={m} onPlay={onPlay} />)}

          {pendingConfirmation && (
            <GlassCard className="mt-2 border border-coral-200 !p-3">
              <div className="text-xs font-medium mb-2">{pendingConfirmation.question}</div>
              {pendingConfirmation.mode === "instances" ? (
                <InstanceGallery
                  items={pendingConfirmation.items.map((it: any, i: number) => ({
                    id: it.detection_id || it.id || String(i),
                    track_id: it.instance_id,
                    preview: it.label,
                    frame_id: it.frame_id,
                    video_id: it.video_id,
                    timestamp_seconds: it.timestamp_seconds,
                  }))}
                  selected={picks}
                  onToggle={(id, kind) => setPicks(p => {
                    const n = new Map(p); n.set(id, n.get(id) === kind ? undefined : kind); return n;
                  })}
                />
              ) : (
                <FrameGrid
                  group={{
                    id: pendingConfirmation.confirmationId,
                    tool: "confirmation",
                    payload: { results: pendingConfirmation.items },
                  }}
                  selectable
                  selected={picks}
                  onToggle={(itemId, kind) => setPicks(p => {
                    const n = new Map(p);
                    n.set(itemId, n.get(itemId) === kind ? undefined : kind);
                    return n;
                  })}
                  onPlay={onPlay}
                />
              )}
              <div className="mt-2 flex items-center justify-between">
                <span className="text-xs text-neutral-400">
                  {picks.size > 0
                    ? `${[...picks.values()].filter(v => v === "confirm").length} confirmed · ${[...picks.values()].filter(v => v === "reject").length} rejected`
                    : "Use ✓/✗ on each item, then submit"}
                </span>
                <Button onClick={() => submitConfirmation.mutate()}>Submit</Button>
              </div>
            </GlassCard>
          )}
        </div>

        <div className="flex gap-2 items-end">
          <textarea
            value={input}
            rows={1}
            onChange={e => {
              setInput(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`;
            }}
            onKeyDown={e => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSend(); }
            }}
            placeholder="Ask the investigator…"
            className="flex-1 px-3 py-2 rounded-xl border border-neutral-200 bg-white/80 outline-none focus:border-coral-400 text-sm resize-none overflow-y-auto leading-5"
            style={{ minHeight: "2.25rem", maxHeight: "7.5rem" }}
          />
          <Button onClick={onSend} disabled={!input.trim() || send.isPending}>
            <Send size={14} />
          </Button>
        </div>
      </GlassCard>

      {/* Video playback modal */}
      {videoPlay && (
        <VideoModal play={videoPlay} onClose={() => setVideoPlay(null)} />
      )}

      {/* RIGHT — results & timeline */}
      <div className="lg:col-span-3 flex flex-col gap-3 min-h-0 overflow-y-auto">
        {resultGroups.length > 0 && (
          <TimelineBar resultGroups={resultGroups} videoDurations={videoDurations} onPlay={onPlay} />
        )}
        {resultGroups.length === 0 && (
          <GlassCard className="text-sm text-neutral-500">
            Results will appear here as the agent runs tools.
          </GlassCard>
        )}
        {resultGroups.slice().reverse().map(g => (
          <GlassCard key={g.id}>
            <div className="text-xs text-neutral-500 mb-2">{g.tool}</div>
            <FrameGrid group={g} onPlay={onPlay} />
          </GlassCard>
        ))}
      </div>
    </div>
  );
}
