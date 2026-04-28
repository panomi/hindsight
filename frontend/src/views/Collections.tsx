import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, Trash2, Video as VideoIcon } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { createCollection, deleteCollection, listCollections } from "../api/collections";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { GlassCard } from "../components/ui/GlassCard";

export default function Collections() {
  const qc = useQueryClient();
  const { data, isLoading } = useQuery({ queryKey: ["collections"], queryFn: listCollections });
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");

  const create = useMutation({
    mutationFn: () => createCollection(name, desc || undefined),
    onSuccess: () => {
      setName(""); setDesc("");
      qc.invalidateQueries({ queryKey: ["collections"] });
    },
  });
  const del = useMutation({
    mutationFn: (id: string) => deleteCollection(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["collections"] }),
  });

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 max-w-6xl mx-auto">
      <GlassCard className="lg:col-span-1">
        <h2 className="text-base font-semibold mb-3">New collection</h2>
        <div className="flex flex-col gap-3">
          <input
            value={name} onChange={e => setName(e.target.value)}
            placeholder="Collection name"
            className="px-3 py-2 rounded-xl border border-neutral-200 bg-white/80 outline-none focus:border-coral-400"
          />
          <textarea
            value={desc} onChange={e => setDesc(e.target.value)}
            placeholder="Description (optional)" rows={3}
            className="px-3 py-2 rounded-xl border border-neutral-200 bg-white/80 outline-none focus:border-coral-400 resize-none"
          />
          <Button onClick={() => create.mutate()} disabled={!name || create.isPending}>
            <Plus size={14} /> Create
          </Button>
        </div>
      </GlassCard>

      <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-3">
        {isLoading && <GlassCard>Loading…</GlassCard>}
        {data?.length === 0 && (
          <GlassCard className="col-span-full text-neutral-500">
            No collections yet. Create one on the left.
          </GlassCard>
        )}
        {data?.map(c => (
          <GlassCard key={c.id} className="flex flex-col gap-3">
            <div className="flex items-start justify-between gap-2">
              <div>
                <div className="font-medium">{c.name}</div>
                <div className="text-xs text-neutral-500 mt-0.5">
                  {new Date(c.created_at).toLocaleString()}
                </div>
              </div>
              <Badge variant="coral">
                <VideoIcon size={10} /> {c.video_count}
              </Badge>
            </div>
            {c.description && <div className="text-sm text-neutral-700">{c.description}</div>}
            <div className="flex items-center justify-between mt-auto pt-2">
              <Link
                to={`/ingest?collection=${c.id}`}
                className="text-sm text-coral-600 hover:underline"
              >
                Add videos →
              </Link>
              <button
                className="text-neutral-400 hover:text-red-600"
                onClick={() => del.mutate(c.id)}
                title="Delete collection"
              >
                <Trash2 size={14} />
              </button>
            </div>
          </GlassCard>
        ))}
      </div>
    </div>
  );
}
