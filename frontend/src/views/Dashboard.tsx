import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, Plus, Search } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { listCollections } from "../api/collections";
import { createInvestigation, listInvestigations } from "../api/investigations";
import { Button } from "../components/ui/Button";
import { GlassCard } from "../components/ui/GlassCard";

export default function Dashboard() {
  const qc = useQueryClient();
  const nav = useNavigate();
  const collections = useQuery({ queryKey: ["collections"], queryFn: listCollections });
  const investigations = useQuery({ queryKey: ["investigations"], queryFn: listInvestigations });

  const [collId, setCollId] = useState("");
  const [title, setTitle] = useState("");

  const create = useMutation({
    mutationFn: () => createInvestigation(collId, title),
    onSuccess: (inv) => {
      qc.invalidateQueries({ queryKey: ["investigations"] });
      setTitle("");
      nav(`/investigate/${inv.id}`);
    },
  });

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-6xl mx-auto">
      <GlassCard>
        <h2 className="text-base font-semibold flex items-center gap-2 mb-3">
          <Plus size={16} className="text-coral-500" /> New investigation
        </h2>
        <div className="flex flex-col gap-3">
          <select
            value={collId} onChange={e => setCollId(e.target.value)}
            className="px-3 py-2 rounded-xl border border-neutral-200 bg-white/80 outline-none"
          >
            <option value="">Select collection…</option>
            {collections.data?.map(c => (
              <option key={c.id} value={c.id}>{c.name} ({c.video_count} videos)</option>
            ))}
          </select>
          <input
            value={title} onChange={e => setTitle(e.target.value)}
            placeholder="Investigation title"
            className="px-3 py-2 rounded-xl border border-neutral-200 bg-white/80 outline-none focus:border-coral-400"
          />
          <Button onClick={() => create.mutate()} disabled={!collId || !title || create.isPending}>
            <Plus size={14} /> Start
          </Button>
        </div>
      </GlassCard>

      <GlassCard>
        <h2 className="text-base font-semibold flex items-center gap-2 mb-3">
          <Search size={16} className="text-coral-500" /> Active investigations
        </h2>
        {investigations.data?.length === 0 && (
          <div className="text-sm text-neutral-500">
            No investigations yet — create one on the left.
          </div>
        )}
        <div className="flex flex-col gap-2">
          {investigations.data?.map(inv => (
            <Link
              key={inv.id} to={`/investigate/${inv.id}`}
              className="block px-3 py-2 rounded-xl border border-neutral-200/60 bg-white/60 hover:bg-white/90 transition"
            >
              <div className="font-medium">{inv.title}</div>
              <div className="text-xs text-neutral-500">
                {new Date(inv.created_at).toLocaleString()} · {inv.status}
              </div>
            </Link>
          ))}
        </div>
      </GlassCard>

      <GlassCard className="md:col-span-2">
        <h2 className="text-base font-semibold flex items-center gap-2 mb-3">
          <Activity size={16} className="text-coral-500" /> Collections
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
          {collections.data?.length === 0 && (
            <div className="text-sm text-neutral-500">No collections.</div>
          )}
          {collections.data?.map(c => (
            <Link
              key={c.id} to={`/ingest?collection=${c.id}`}
              className="flex items-center justify-between px-3 py-2 rounded-xl border border-neutral-200/60 bg-white/60 hover:bg-white/90 transition"
            >
              <span className="font-medium">{c.name}</span>
              <span className="text-xs text-neutral-500">{c.video_count} videos</span>
            </Link>
          ))}
        </div>
      </GlassCard>
    </div>
  );
}
