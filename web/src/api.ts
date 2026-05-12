import type {
  ClarificationRequest,
  EntityResult,
  FieldAuditReport,
  MockRow,
  ResearchPlan,
  SearchEngine,
} from "./types";

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
    r = await fetch(url, { method: "POST", headers, body: JSON.stringify(body) });
  } catch (e) {
    throw new Error(`Network error: ${(e as Error).message}`);
  }
  if (!r.ok) throw new Error(await readError(r));
  return r.json();
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

export interface RunEvents {
  onEntityStart?: (entity: string) => void;
  onEntityDone?: (result: EntityResult) => void;
  onDone?: () => void;
  onError?: (msg: string) => void;
}

/**
 * Streams Server-Sent Events from /api/run. Returns an AbortController
 * the caller can use to cancel.
 */
export function runResearch(
  plan: ResearchPlan,
  entities: string[],
  search_engine: SearchEngine,
  events: RunEvents,
): AbortController {
  const ctrl = new AbortController();

  (async () => {
    try {
      const r = await fetch("/api/run", {
        method: "POST",
        headers,
        body: JSON.stringify({ plan, entities, search_engine }),
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

          if (eventName === "entity_start") events.onEntityStart?.(payload.entity);
          else if (eventName === "entity_done") events.onEntityDone?.(payload as EntityResult);
          else if (eventName === "done") events.onDone?.();
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
