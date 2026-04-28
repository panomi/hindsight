export type Collection = {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  video_count: number;
};

export type Video = {
  id: string;
  collection_id: string;
  filename: string;
  duration_seconds: number | null;
  fps: number | null;
  resolution: string | null;
  status: "pending" | "processing" | "ready" | "error";
  stage: string | null;
  progress_pct: number;
  error: string | null;
  created_at: string;
};

export type Investigation = {
  id: string;
  collection_id: string;
  title: string;
  status: string;
  created_at: string;
};
