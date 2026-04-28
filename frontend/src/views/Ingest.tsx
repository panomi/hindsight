import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FolderInput, Plus, Upload as UploadIcon } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  createCollection, getScanRoots, listCollections, listVideos,
  scanDirectory, uploadVideo,
} from "../api/collections";
import type { Video } from "../api/types";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { GlassCard } from "../components/ui/GlassCard";

const STATUS_VARIANT: Record<Video["status"], "neutral" | "amber" | "green" | "red"> = {
  pending: "neutral", processing: "amber", ready: "green", error: "red",
};

const STAGE_LABEL: Record<string, string> = {
  queued:       "Waiting for previous video to finish",
  shots:        "Detecting scenes",
  frames:       "Extracting keyframes",
  detect_track: "Detecting & tracking objects",
  embed_global: "Indexing frames for visual search",
  embed_box:    "Indexing objects for subject search",
  transcribe:   "Transcribing speech",
  caption:      "Generating scene descriptions",
  ocr:          "Reading on-screen text",
  done:         "Complete",
};

const VIDEO_EXTS = [".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"];
const isVideo = (f: File) => VIDEO_EXTS.some(e => f.name.toLowerCase().endsWith(e));

type Mode = "upload" | "scan";

export default function Ingest() {
  const qc = useQueryClient();
  const [params, setParams] = useSearchParams();
  const collectionId = params.get("collection") ?? "";
  const [mode, setMode] = useState<Mode>("upload");

  const collections = useQuery({ queryKey: ["collections"], queryFn: listCollections });
  const scanRoots = useQuery({ queryKey: ["scan-roots"], queryFn: getScanRoots });
  const videos = useQuery({
    queryKey: ["videos", collectionId || null],
    queryFn: () => listVideos(collectionId || undefined),
    refetchInterval: (q) => {
      const data = q.state.data;
      return data?.some(v => v.status === "pending" || v.status === "processing") ? 1500 : false;
    },
  });

  // ── inline collection creation when none exist ──────────────────────────
  const [newName, setNewName] = useState("");
  const createColl = useMutation({
    mutationFn: () => createCollection(newName),
    onSuccess: (c) => {
      setNewName("");
      qc.invalidateQueries({ queryKey: ["collections"] });
      setParams({ collection: c.id });
    },
  });

  // ── multi-file upload state ────────────────────────────────────────────
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState<{ name: string; status: "queued" | "ok" | "err"; error?: string }[]>([]);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const addFiles = (incoming: FileList | File[] | null) => {
    if (!incoming) return;
    const arr = Array.from(incoming).filter(isVideo);
    setFiles(prev => {
      const seen = new Set(prev.map(f => f.name + f.size));
      return [...prev, ...arr.filter(f => !seen.has(f.name + f.size))];
    });
  };

  const upload = useMutation({
    mutationFn: async () => {
      if (!collectionId || files.length === 0) throw new Error("collection and files required");
      const queue = files.map(f => ({ name: f.name, status: "queued" as const }));
      setUploading(queue);
      const limit = 3; // small concurrency cap so the API isn't hammered
      let i = 0;
      const workers = Array.from({ length: Math.min(limit, files.length) }, async () => {
        while (i < files.length) {
          const idx = i++; const f = files[idx];
          try {
            await uploadVideo(collectionId, f);
            setUploading(prev => prev.map((u, k) => k === idx ? { ...u, status: "ok" } : u));
          } catch (e: any) {
            setUploading(prev => prev.map((u, k) => k === idx ? { ...u, status: "err", error: String(e?.message || e) } : u));
          }
        }
      });
      await Promise.all(workers);
      setFiles([]);
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ["videos"] });
      qc.invalidateQueries({ queryKey: ["collections"] });
    },
  });

  // ── drag/drop ──────────────────────────────────────────────────────────
  const [dragOver, setDragOver] = useState(false);

  // ── server-side scan ───────────────────────────────────────────────────
  const [scanPath, setScanPath] = useState("");
  const [recursive, setRecursive] = useState(true);
  const scan = useMutation({
    mutationFn: () => scanDirectory(collectionId, scanPath, recursive),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["videos"] });
      qc.invalidateQueries({ queryKey: ["collections"] });
    },
  });

  // ── derived ────────────────────────────────────────────────────────────
  const sortedVideos = useMemo(() => videos.data ?? [], [videos.data]);
  const noCollections = (collections.data?.length ?? 0) === 0;

  const reasonDisabled =
    !collectionId ? "Select or create a collection above"
    : mode === "upload" && files.length === 0 ? "Add at least one video file"
    : mode === "scan" && !scanPath ? "Enter the server-side directory path"
    : null;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 max-w-6xl mx-auto">
      {/* LEFT — collection + upload form */}
      <div className="lg:col-span-1 flex flex-col gap-4">
        {/* Help */}
        <GlassCard className="!p-4 text-xs leading-relaxed text-neutral-600">
          <div className="font-semibold text-neutral-800 mb-1.5">What's a collection?</div>
          A collection is a group of videos you want to investigate together — e.g. all
          footage from one location, one day, or one case. Create a collection first,
          then add videos to it. The agent will only search within the collection you
          pick when you start an investigation.
        </GlassCard>

        {/* Inline collection creation when none exist */}
        {noCollections ? (
          <GlassCard>
            <h2 className="text-base font-semibold mb-2">Create your first collection</h2>
            <div className="flex flex-col gap-3">
              <input
                value={newName} onChange={e => setNewName(e.target.value)}
                placeholder="e.g. 'Cafe footage Apr 26'"
                className="px-3 py-2 rounded-xl border border-neutral-200 bg-white/80 outline-none focus:border-coral-400 text-sm"
              />
              <Button onClick={() => createColl.mutate()} disabled={!newName || createColl.isPending}>
                <Plus size={14} /> Create collection
              </Button>
            </div>
          </GlassCard>
        ) : (
          <GlassCard>
            <h2 className="text-base font-semibold mb-3">Add videos</h2>

            <select
              value={collectionId}
              onChange={e => setParams(e.target.value ? { collection: e.target.value } : {})}
              className="w-full px-3 py-2 rounded-xl border border-neutral-200 bg-white/80 outline-none mb-3 text-sm"
            >
              <option value="">Select collection…</option>
              {collections.data?.map(c => (
                <option key={c.id} value={c.id}>{c.name} ({c.video_count})</option>
              ))}
            </select>

            {/* Mode tabs */}
            <div className="grid grid-cols-2 gap-1 mb-3 p-1 bg-neutral-100/60 rounded-xl">
              <button
                onClick={() => setMode("upload")}
                className={`text-xs font-medium py-1.5 rounded-lg transition ${mode === "upload" ? "coral-grad text-white shadow-sm" : "text-neutral-600 hover:text-neutral-900"}`}
              >Upload from this device</button>
              <button
                onClick={() => setMode("scan")}
                className={`text-xs font-medium py-1.5 rounded-lg transition ${mode === "scan" ? "coral-grad text-white shadow-sm" : "text-neutral-600 hover:text-neutral-900"}`}
              >Scan server directory</button>
            </div>

            {mode === "upload" ? (
              <div className="flex flex-col gap-3">
                {/* Drag-drop zone */}
                <div
                  onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={e => {
                    e.preventDefault(); setDragOver(false);
                    addFiles(e.dataTransfer.files);
                  }}
                  onClick={() => inputRef.current?.click()}
                  className={`cursor-pointer text-center py-6 px-3 rounded-xl border-2 border-dashed text-sm transition
                    ${dragOver ? "border-coral-400 bg-coral-50/50" : "border-neutral-300 bg-white/40 hover:border-coral-300 hover:bg-coral-50/30"}`}
                >
                  <UploadIcon size={20} className="mx-auto mb-1.5 text-coral-500" />
                  Drop files or click to choose
                  <div className="text-xs text-neutral-500 mt-1">
                    {VIDEO_EXTS.join(" ")} · multi-select supported
                  </div>
                </div>
                <input
                  ref={inputRef} type="file" accept="video/*" multiple
                  onChange={e => { addFiles(e.target.files); e.currentTarget.value = ""; }}
                  className="hidden"
                />

                {files.length > 0 && (
                  <div className="text-xs text-neutral-600">
                    <div className="font-medium mb-1">{files.length} file(s) ready:</div>
                    <ul className="max-h-32 overflow-y-auto space-y-0.5">
                      {files.map((f, i) => (
                        <li key={i} className="flex items-center justify-between gap-2 truncate">
                          <span className="truncate">{f.name}</span>
                          <button
                            onClick={() => setFiles(prev => prev.filter((_, k) => k !== i))}
                            className="text-neutral-400 hover:text-red-600 shrink-0"
                          >×</button>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {uploading.length > 0 && (
                  <div className="text-xs space-y-1 max-h-32 overflow-y-auto border-t border-neutral-200 pt-2">
                    {uploading.map((u, i) => (
                      <div key={i} className="flex items-center justify-between gap-2">
                        <span className="truncate">{u.name}</span>
                        <Badge variant={u.status === "ok" ? "green" : u.status === "err" ? "red" : "amber"}>
                          {u.status === "ok" ? "queued" : u.status === "err" ? "error" : "uploading…"}
                        </Badge>
                      </div>
                    ))}
                  </div>
                )}

                <Button
                  onClick={() => upload.mutate()}
                  disabled={!!reasonDisabled || upload.isPending}
                >
                  <UploadIcon size={14} />
                  {upload.isPending ? "Uploading…" : `Upload & ingest${files.length ? ` ${files.length}` : ""}`}
                </Button>
                {reasonDisabled && (
                  <div className="text-xs text-neutral-500">{reasonDisabled}.</div>
                )}
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                <div className="text-xs text-neutral-600">
                  Point at a directory <em>on the server</em> (the machine running the API).
                  Files are referenced in place — nothing is uploaded over HTTP. Useful when
                  100+ GB already lives on the GPU box.
                </div>
                {scanRoots.data && (
                  <div className="text-xs">
                    {scanRoots.data.roots.length === 0 ? (
                      <span className="text-amber-700">
                        Server-side scanning is disabled. Set{" "}
                        <code className="bg-amber-50 px-1 py-0.5 rounded">INGEST_SCAN_ROOTS</code>
                        {" "}in <code className="bg-amber-50 px-1 py-0.5 rounded">backend/.env</code>{" "}
                        to a comma-separated list of allowed roots, then restart the API.
                      </span>
                    ) : (
                      <span className="text-neutral-500">
                        Allowed roots: {scanRoots.data.roots.map(r => (
                          <button
                            key={r}
                            onClick={() => setScanPath(r)}
                            className="mr-1 px-1.5 py-0.5 rounded bg-neutral-100 hover:bg-neutral-200 font-mono text-[11px]"
                          >{r}</button>
                        ))}
                      </span>
                    )}
                  </div>
                )}
                <input
                  value={scanPath} onChange={e => setScanPath(e.target.value)}
                  placeholder="/path/on/server/to/footage"
                  disabled={scanRoots.data?.roots.length === 0}
                  className="px-3 py-2 rounded-xl border border-neutral-200 bg-white/80 outline-none focus:border-coral-400 text-sm font-mono disabled:opacity-50"
                />
                <label className="text-xs text-neutral-600 inline-flex items-center gap-2">
                  <input type="checkbox" checked={recursive} onChange={e => setRecursive(e.target.checked)} />
                  Recurse into subdirectories
                </label>
                <Button
                  onClick={() => scan.mutate()}
                  disabled={!!reasonDisabled || scan.isPending || scanRoots.data?.roots.length === 0}
                >
                  <FolderInput size={14} />
                  {scan.isPending ? "Scanning…" : "Scan & ingest"}
                </Button>
                {reasonDisabled && (
                  <div className="text-xs text-neutral-500">{reasonDisabled}.</div>
                )}
                {scan.isSuccess && (
                  <div className="text-xs text-emerald-700">
                    Queued {scan.data.queued.length} video(s).
                    {scan.data.skipped.length > 0 && ` ${scan.data.skipped.length} skipped.`}
                  </div>
                )}
                {scan.isError && (
                  <div className="text-xs text-red-600">{(scan.error as Error).message}</div>
                )}
              </div>
            )}
          </GlassCard>
        )}
      </div>

      {/* RIGHT — videos list */}
      <div className="lg:col-span-2 flex flex-col gap-3">
        <div className="text-sm text-neutral-600 ml-1">
          {collectionId ? "Videos in this collection" : "All recent videos"}
        </div>
        {sortedVideos.length === 0 && (
          <GlassCard className="text-neutral-500 text-sm">No videos yet.</GlassCard>
        )}
        {sortedVideos.map(v => (
          <GlassCard key={v.id} className="flex items-center gap-4">
            <div className="flex-1 min-w-0">
              <div className="font-medium truncate">{v.filename}</div>
              <div className="text-xs text-neutral-500 mt-0.5 flex items-center gap-3 flex-wrap">
                {v.duration_seconds ? `${v.duration_seconds.toFixed(1)}s` : "—"}
                {v.resolution && <span>{v.resolution}</span>}
                {v.fps && <span>{v.fps.toFixed(1)} fps</span>}
                {v.stage && (
                  <span className="text-coral-600">
                    {STAGE_LABEL[v.stage] ?? v.stage}
                    {v.status === "processing" && v.progress_pct > 0 && ` · ${v.progress_pct}%`}
                  </span>
                )}
              </div>
              {(v.status === "processing" || v.status === "pending") && (
                <div className="mt-2 h-1.5 w-full bg-neutral-200/60 rounded-full overflow-hidden">
                  <div
                    className="h-full coral-grad transition-all duration-500"
                    style={{ width: `${v.progress_pct}%` }}
                  />
                </div>
              )}
              {v.error && <div className="text-xs text-red-600 mt-1">{v.error}</div>}
            </div>
            <Badge variant={STATUS_VARIANT[v.status]}>{v.status}</Badge>
          </GlassCard>
        ))}
      </div>
    </div>
  );
}