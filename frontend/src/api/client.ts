export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T>(
  path: string,
  init?: RequestInit & { json?: unknown }
): Promise<T> {
  const headers = new Headers(init?.headers);
  let body = init?.body as BodyInit | undefined;
  if (init?.json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(init.json);
  }
  const res = await fetch(path, { ...init, headers, body });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const data = await res.json();
      msg = data.detail ?? msg;
    } catch {}
    throw new ApiError(res.status, msg);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}
