import type {
  ClarificationRequest,
  EntityResult,
  FieldAuditReport,
  MockRow,
  ResearchPlan,
  SearchEngine,
} from "./types";

const headers = { "Content-Type": "application/json" };

export interface CompileResponse {
  kind: "clarification" | "plan";
  clarification?: ClarificationRequest;
  plan?: ResearchPlan;
}

export async function compileSchema(
  question: string,
  entity_type = "",
): Promise<CompileResponse> {
  const r = await fetch("/api/compile-schema", {
    method: "POST",
    headers,
    body: JSON.stringify({ question, entity_type }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function auditPlan(plan: ResearchPlan): Promise<FieldAuditReport> {
  const r = await fetch("/api/audit", {
    method: "POST",
    headers,
    body: JSON.stringify({ plan }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function mockPreview(plan: ResearchPlan): Promise<MockRow[]> {
  const r = await fetch("/api/mock-preview", {
    method: "POST",
    headers,
    body: JSON.stringify({ plan }),
  });
  if (!r.ok) throw new Error(await r.text());
  const data = await r.json();
  return data.rows;
}

export async function enrichPlan(plan: ResearchPlan): Promise<ResearchPlan> {
  const r = await fetch("/api/enrich", {
    method: "POST",
    headers,
    body: JSON.stringify({ plan }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
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
      if (!r.ok || !r.body) throw new Error(await r.text());

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
