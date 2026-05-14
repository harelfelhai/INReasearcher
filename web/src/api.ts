import type {
  AuthUser,
  ClarificationRequest,
  EntityDiscoveryPlan,
  EntityDiscoveryResult,
  EntityResult,
  FieldAuditReport,
  LoginResponse,
  ManagedUser,
  MockRow,
  ResearchPlan,
  SearchEngine,
  SessionOut,
} from "./types";

const TOKEN_KEY = "inr_auth_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string | null): void {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (token) h["Authorization"] = `Bearer ${token}`;
  return h;
}

const headers = { "Content-Type": "application/json" };

/**
 * Reads an error response body and extracts the most useful message we can.
 * FastAPI returns `{"detail": "..."}` on errors thrown by our exception
 * handler, but we also fall back gracefully to plain text or generic
 * status messages.
 */
async function readError(r: Response): Promise<string> {
  let body = "";
  try {
    body = await r.text();
  } catch {
    /* empty */
  }
  try {
    const json = JSON.parse(body);
    if (typeof json.detail === "string") return json.detail;
    if (json.detail) return JSON.stringify(json.detail);
  } catch {
    /* not JSON, fall through */
  }
  return body || `HTTP ${r.status} ${r.statusText}`;
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  let r: Response;
  try {
    r = await fetch(url, { method: "POST", headers: authHeaders(), body: JSON.stringify(body) });
  } catch (e) {
    throw new Error(`Network error: ${(e as Error).message}`);
  }
  if (!r.ok) throw new Error(await readError(r));
  return r.json();
}

async function getJson<T>(url: string): Promise<T> {
  let r: Response;
  try {
    r = await fetch(url, { headers: authHeaders() });
  } catch (e) {
    throw new Error(`Network error: ${(e as Error).message}`);
  }
  if (!r.ok) throw new Error(await readError(r));
  return r.json();
}

async function putJson<T>(url: string, body: unknown): Promise<T> {
  let r: Response;
  try {
    r = await fetch(url, { method: "PUT", headers: authHeaders(), body: JSON.stringify(body) });
  } catch (e) {
    throw new Error(`Network error: ${(e as Error).message}`);
  }
  if (!r.ok) throw new Error(await readError(r));
  return r.json();
}

async function del(url: string): Promise<void> {
  let r: Response;
  try {
    r = await fetch(url, { method: "DELETE", headers: authHeaders() });
  } catch (e) {
    throw new Error(`Network error: ${(e as Error).message}`);
  }
  if (!r.ok && r.status !== 204) throw new Error(await readError(r));
}

// ── Auth ────────────────────────────────────────────────────────────────────

export async function login(username: string, password: string): Promise<LoginResponse> {
  const r = await fetch("/api/auth/login", {
    method: "POST",
    headers,
    body: JSON.stringify({ username, password }),
  });
  if (!r.ok) throw new Error(await readError(r));
  const data = (await r.json()) as LoginResponse;
  setToken(data.access_token);
  return data;
}

export function logout(): void {
  setToken(null);
}

export async function fetchMe(): Promise<AuthUser> {
  return getJson("/api/auth/me");
}

// ── Admin ───────────────────────────────────────────────────────────────────

export async function listManagedUsers(): Promise<ManagedUser[]> {
  return getJson("/api/admin/users");
}

export async function createManagedUser(payload: {
  username: string;
  password: string;
  credit_balance: number;
}): Promise<AuthUser> {
  return postJson("/api/admin/users", { ...payload, role: "user" });
}

export async function updateUserBudget(
  userId: string,
  payload: { set_to?: number; add?: number },
): Promise<AuthUser> {
  return putJson(`/api/admin/users/${userId}/budget`, payload);
}

export async function disableUser(userId: string): Promise<void> {
  return del(`/api/admin/users/${userId}`);
}

export async function listUserSessions(userId: string): Promise<SessionOut[]> {
  return getJson(`/api/admin/users/${userId}/sessions`);
}

// ── User sessions / downloads ───────────────────────────────────────────────

export async function listOwnSessions(): Promise<SessionOut[]> {
  return getJson("/api/user/sessions");
}

export function exportDownloadUrl(exportId: string): string {
  // Used by an anchor's href; auth header isn't attached, so we use a token
  // query string only if needed. Instead we open in a new tab via fetch+blob.
  return `/api/user/exports/${exportId}`;
}

export async function downloadExport(exportId: string, filename: string): Promise<void> {
  const r = await fetch(exportDownloadUrl(exportId), { headers: authHeaders() });
  if (!r.ok) throw new Error(await readError(r));
  const blob = await r.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

export interface CompileResponse {
  kind: "clarification" | "plan";
  clarification?: ClarificationRequest;
  plan?: ResearchPlan;
}

export async function compileSchema(
  question: string,
  entity_type = "",
): Promise<CompileResponse> {
  return postJson("/api/compile-schema", { question, entity_type });
}

export async function auditPlan(plan: ResearchPlan): Promise<FieldAuditReport> {
  return postJson("/api/audit", { plan });
}

export async function mockPreview(plan: ResearchPlan): Promise<MockRow[]> {
  const data = await postJson<{ rows: MockRow[] }>("/api/mock-preview", { plan });
  return data.rows;
}

export async function enrichPlan(plan: ResearchPlan): Promise<ResearchPlan> {
  return postJson("/api/enrich", { plan });
}

// ── Entity discovery (optional, opt-in) ─────────────────────────────────────

export async function planDiscovery(
  question: string,
  entity_type = "",
): Promise<EntityDiscoveryPlan> {
  return postJson("/api/discover-entities/plan", { question, entity_type });
}

export async function runDiscovery(
  plan: ResearchPlan,
  discovery: EntityDiscoveryPlan,
  search_engine: SearchEngine,
): Promise<EntityDiscoveryResult | null> {
  return postJson("/api/discover-entities/run", { plan, discovery, search_engine });
}

export interface RunDonePayload {
  n: number;
  session_id: string;
  cost_used: number;
  tokens_in: number;
  tokens_out: number;
  export: { export_id: string; filename: string } | null;
}

export interface RunEvents {
  onSessionStarted?: (info: { session_id: string; credit_balance: number }) => void;
  onEntityStart?: (entity: string) => void;
  onEntityDone?: (result: EntityResult) => void;
  onDone?: (info: RunDonePayload) => void;
  onError?: (msg: string) => void;
}

/**
 * Streams Server-Sent Events from /api/run. Returns an AbortController
 * the caller can use to cancel.
 */
export type SeededProbe = Record<
  string,
  Record<string, { value: string; quote?: string; source_url?: string }>
>;

export function runResearch(
  plan: ResearchPlan,
  entities: string[],
  search_engine: SearchEngine,
  events: RunEvents,
  seeded_probe?: SeededProbe,
): AbortController {
  const ctrl = new AbortController();

  (async () => {
    try {
      const r = await fetch("/api/run", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ plan, entities, search_engine, seeded_probe }),
        signal: ctrl.signal,
      });
      if (!r.ok || !r.body) throw new Error(await readError(r));

      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const block of parts) {
          let eventName = "message";
          let data = "";
          for (const line of block.split("\n")) {
            if (line.startsWith("event: ")) eventName = line.slice(7).trim();
            else if (line.startsWith("data: ")) data += line.slice(6);
          }
          if (!data) continue;
          const payload = JSON.parse(data);

          if (eventName === "session_started") events.onSessionStarted?.(payload);
          else if (eventName === "entity_start") events.onEntityStart?.(payload.entity);
          else if (eventName === "entity_done") events.onEntityDone?.(payload as EntityResult);
          else if (eventName === "done") events.onDone?.(payload as RunDonePayload);
          else if (eventName === "error") events.onError?.(payload.message);
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        events.onError?.((e as Error).message);
      }
    }
  })();

  return ctrl;
}
