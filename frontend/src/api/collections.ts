import { api } from "./client";
import type { Collection, Video } from "./types";

export const listCollections = () => api<Collection[]>("/api/collections");
export const getCollection = (id: string) => api<Collection>(`/api/collections/${id}`);
export const createCollection = (name: string, description?: string) =>
  api<Collection>("/api/collections", { method: "POST", json: { name, description } });
export const deleteCollection = (id: string) =>
  api<void>(`/api/collections/${id}`, { method: "DELETE" });

export const listVideos = (collectionId?: string) => {
  const q = collectionId ? `?collection_id=${collectionId}` : "";
  return api<Video[]>(`/api/videos${q}`);
};
export const getVideo = (id: string) => api<Video>(`/api/videos/${id}`);

export const uploadVideo = async (collectionId: string, file: File) => {
  const fd = new FormData();
  fd.append("collection_id", collectionId);
  fd.append("file", file);
  return api<{ video_id: string; status: string }>("/api/ingest", {
    method: "POST",
    body: fd,
  });
};

export const getScanRoots = () =>
  api<{ roots: string[] }>("/api/ingest/scan-roots");

export const scanDirectory = (collectionId: string, serverPath: string, recursive = true) =>
  api<{ queued: { video_id: string; status: string }[]; skipped: string[] }>(
    "/api/ingest/scan",
    { method: "POST", json: { collection_id: collectionId, server_path: serverPath, recursive } },
  );