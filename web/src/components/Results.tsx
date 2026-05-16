import { useEffect, useRef, useState } from "react";
import {
  deepenResearch,
  downloadExport,
  exportToExcel,
  runResearch,
  submitFeedback,
  type DeepenDonePayload,
  type DeepenMissingCell,
  type RunDonePayload,
  type SeededProbe,
} from "../api";
import type { CellFeedback, EntityResult, ResearchPlan, SearchEngine, VerifiedCell } from "../types";

interface Props {
  plan: ResearchPlan;
  entities: string[];
  searchEngine: SearchEngine;
  seededProbe?: SeededProbe;
  onRestart: () => void;
}

// ── Post-run feedback panel ───────────────────────────────────────────────────

interface FeedbackState {
  [entityField: string]: { is_correct: boolean; correct_value: string };
}

function FeedbackPanel({ results, plan, sessionId }: {
  results: EntityResult[];
  plan: ResearchPlan;
  sessionId: string;
}) {
  const [feedback, setFeedback] = useState<FeedbackState>({});
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<{ successes: number; failures: number } | null>(null);

  const key = (entity: string, fieldId: string) => `${entity}||${fieldId}`;

  function toggle(entity: string, fieldId: string, is_correct: boolean) {
    setFeedback((prev) => ({
      ...prev,
      [key(entity, fieldId)]: { is_correct, correct_value: prev[key(entity, fieldId)]?.correct_value ?? "" },
    }));
  }

  function setCorrection(entity: string, fieldId: string, val: string) {
    setFeedback((prev) => ({
      ...prev,
      [key(entity, fieldId)]: { ...prev[key(entity, fieldId)], correct_value: val },
    }));
  }

  async function submit() {
    const cells: CellFeedback[] = Object.entries(feedback).map(([k, v]) => {
      const [entity_name, field_id] = k.split("||");
      return {
        entity_name,
        field_id,
        is_correct: v.is_correct,
        correct_value: v.is_correct ? null : (v.correct_value || null),
      };
    });
    if (cells.length === 0) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await submitFeedback(sessionId, cells);
      setStats({ successes: res.recorded_successes, failures: res.recorded_failures });
      setSubmitted(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  const reviewedCount = Object.keys(feedback).length;
  const valuedCells = results.flatMap((r) =>
    plan.columns.filter((c) => r.cells[c.id]?.value).map((c) => ({ entity: r.entity_name, col: c }))
  );

  if (submitted && stats) {
    return (
      <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4 text-sm text-emerald-800">
        ✓ משוב נשמר — {stats.successes} תקינים, {stats.failures} שגויים. השיפורים ייכנסו לתוקף בהרצה הבאה.
      </div>
    );
  }

  return (
    <details className="bg-white border border-slate-200 rounded-xl shadow-sm">
      <summary className="p-4 cursor-pointer font-medium text-sm select-none">
        סקירת תוצאות לשיפור עתידי
        <span className="text-slate-400 font-normal mr-2">
          ({reviewedCount}/{valuedCells.length} סומנו)
        </span>
      </summary>
      <div className="p-4 pt-0 space-y-3">
        <p className="text-xs text-slate-500">
          סמן כל תא כנכון או שגוי. התשובות ישמרו כדוגמאות לשיפור ההרצות הבאות.
        </p>
        <div className="overflow-x-auto">
          <table className="text-sm w-full">
            <thead className="bg-slate-50 text-right">
              <tr>
                <th className="p-2">ישות</th>
                <th className="p-2">שדה</th>
                <th className="p-2">ערך</th>
                <th className="p-2">נכון?</th>
                <th className="p-2">ערך נכון (אם שגוי)</th>
              </tr>
            </thead>
            <tbody>
              {valuedCells.map(({ entity, col }) => {
                const cell = results.find((r) => r.entity_name === entity)?.cells[col.id];
                const fb = feedback[key(entity, col.id)];
                return (
                  <tr key={key(entity, col.id)} className="border-t border-slate-100">
                    <td className="p-2" dir="auto">{entity}</td>
                    <td className="p-2 text-slate-500 font-mono text-xs">{col.id}</td>
                    <td className="p-2" dir="auto">{cell?.value}</td>
                    <td className="p-2 whitespace-nowrap">
                      <button
                        onClick={() => toggle(entity, col.id, true)}
                        className={`px-2 py-0.5 rounded text-xs mr-1 border ${
                          fb?.is_correct === true
                            ? "bg-emerald-100 border-emerald-400 text-emerald-800"
                            : "border-slate-300 text-slate-600 hover:bg-slate-50"
                        }`}
                      >✓ נכון</button>
                      <button
                        onClick={() => toggle(entity, col.id, false)}
                        className={`px-2 py-0.5 rounded text-xs border ${
                          fb?.is_correct === false
                            ? "bg-rose-100 border-rose-400 text-rose-800"
                            : "border-slate-300 text-slate-600 hover:bg-slate-50"
                        }`}
                      >✗ שגוי</button>
                    </td>
                    <td className="p-2">
                      {fb?.is_correct === false && (
                        <input
                          type="text"
                          placeholder="הזן ערך נכון"
                          value={fb.correct_value}
                          onChange={(e) => setCorrection(entity, col.id, e.target.value)}
                          className="border border-slate-300 rounded px-2 py-0.5 text-xs w-40"
                          dir="auto"
                        />
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {error && <div className="text-xs text-rose-700">{error}</div>}
        <div className="flex justify-end">
          <button
            onClick={submit}
            disabled={submitting || reviewedCount === 0}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-1.5 rounded-lg text-sm font-semibold disabled:opacity-50"
          >
            {submitting ? "שולח…" : "שלח משוב"}
          </button>
        </div>
      </div>
    </details>
  );
}


// ── Main Results component ────────────────────────────────────────────────────

export default function Results({ plan, entities, searchEngine, seededProbe, onRestart }: Props) {
  const [results, setResults] = useState<EntityResult[]>([]);
  const [current, setCurrent] = useState<string | null>(null);
  const [status, setStatus] = useState<"running" | "done" | "error">("running");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [doneInfo, setDoneInfo] = useState<RunDonePayload | null>(null);
  const ctrlRef = useRef<AbortController | null>(null);

  // Deep-retry state
  const [deepenStatus, setDeepenStatus] = useState<"idle" | "running" | "done">("idle");
  const [deepenInfo, setDeepenInfo] = useState<DeepenDonePayload | null>(null);
  const deepenCtrlRef = useRef<AbortController | null>(null);

  useEffect(() => {
    ctrlRef.current = runResearch(plan, entities, searchEngine, {
      onEntityStart: (e) => setCurrent(e),
      onEntityDone: (r) => setResults((prev) => [...prev, r]),
      onDone: (info) => { setStatus("done"); setCurrent(null); setDoneInfo(info); },
      onError: (m) => { setStatus("error"); setErrorMsg(m); },
    }, seededProbe);
    return () => ctrlRef.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function missingCells(): DeepenMissingCell[] {
    const missing: DeepenMissingCell[] = [];
    for (const r of results) {
      for (const c of plan.columns) {
        if (!r.cells[c.id]?.value) {
          missing.push({ entity: r.entity_name, field_id: c.id });
        }
      }
    }
    return missing;
  }

  function startDeepen() {
    const missing = missingCells();
    if (missing.length === 0 || deepenStatus !== "idle") return;
    setDeepenStatus("running");
    deepenCtrlRef.current = deepenResearch(plan, missing, searchEngine, {
      onEntityDeeped: (entity, newCells) => {
        setResults((prev) =>
          prev.map((r) =>
            r.entity_name === entity
              ? { ...r, cells: { ...r.cells, ...(newCells as Record<string, VerifiedCell>) } }
              : r,
          ),
        );
      },
      onDone: (info) => { setDeepenStatus("done"); setDeepenInfo(info); },
      onError: (msg) => { setDeepenStatus("done"); setErrorMsg(msg); },
    });
  }

  function downloadCsv() {
    const headers = ["entity", ...plan.columns.map((c) => c.id), "flags"];
    const rows = results.map((r) => {
      const cells = plan.columns.map((c) => {
        const cell = r.cells[c.id];
        return cell?.value ?? "";
      });
      return [r.entity_name, ...cells, r.row_flags.join(";")];
    });
    const csv = [headers, ...rows]
      .map((row) => row.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(","))
      .join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "results.csv";
    a.click();
  }

  return (
    <div className="space-y-4">
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-5 flex items-center justify-between">
        <div className="text-sm">
          {status === "running" && (
            <>
              <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 ml-2 animate-pulse" />
              מריץ… {results.length}/{entities.length} הושלמו
              {current && <span className="text-slate-500"> · כעת: <span dir="auto">{current}</span></span>}
            </>
          )}
          {status === "done" && (
            <span className="text-emerald-700">
              ✓ הסתיים — {results.length} ישויות
              {doneInfo && (
                <span className="text-slate-500 mr-2">
                  · עלות: <span className="font-mono">${doneInfo.cost_used.toFixed(4)}</span>
                </span>
              )}
            </span>
          )}
          {status === "error" && <span className="text-rose-700">שגיאה: {errorMsg}</span>}
        </div>
        <div className="flex gap-2">
          {doneInfo?.export && (
            <button
              onClick={() => downloadExport(doneInfo.export!.export_id, doneInfo.export!.filename)}
              className="px-3 py-1.5 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700"
            >
              הורד Excel
            </button>
          )}
          {results.length > 0 && (
            <button onClick={downloadCsv} className="px-3 py-1.5 rounded-lg border border-slate-300 bg-white text-sm font-medium hover:bg-slate-50 transition-colors">
              הורד CSV
            </button>
          )}
          <button onClick={onRestart} className="px-3 py-1.5 rounded-lg border border-slate-300 bg-white text-sm font-medium hover:bg-slate-50 transition-colors">
            שאלה חדשה
          </button>
        </div>
      </div>

      {/* Deepen-search banner — shown after run completes if any cells are missing */}
      {status === "done" && deepenStatus === "idle" && missingCells().length > 0 && (
        <div className="bg-amber-50 border border-amber-300 rounded-xl p-4 flex items-center justify-between gap-4">
          <div className="text-sm text-amber-800">
            <span className="font-semibold">{missingCells().length} שדות לא נמצאו</span>
            {" — "}חיפוש מעמיק ינסה מחדש רק את השדות החסרים עם תוצאות נוספות.
          </div>
          <button
            onClick={startDeepen}
            className="shrink-0 bg-amber-600 hover:bg-amber-700 text-white px-4 py-1.5 rounded-lg text-sm font-semibold"
          >
            חפש לעמוק יותר
          </button>
        </div>
      )}
      {deepenStatus === "running" && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-3 text-sm text-blue-800 flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
          מחפש לעמוק יותר בשדות החסרים…
        </div>
      )}
      {deepenStatus === "done" && deepenInfo && (
        <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-3 text-sm text-emerald-800 flex items-center justify-between gap-4">
          <span>
            ✓ חיפוש מעמיק הסתיים — נמצאו{" "}
            <span className="font-semibold">{deepenInfo.cells_found}</span> שדות נוספים
            {" · "}עלות: <span className="font-mono">${deepenInfo.cost_usd.toFixed(4)}</span>
          </span>
          <button
            onClick={() => exportToExcel(plan, results)}
            className="shrink-0 bg-emerald-700 hover:bg-emerald-800 text-white px-3 py-1 rounded-lg text-xs font-semibold"
          >
            הורד Excel מעודכן
          </button>
        </div>
      )}

      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-x-auto">
        <table className="text-sm w-full">
          <thead className="bg-slate-100 text-right">
            <tr>
              <th className="p-2">ישות</th>
              {plan.columns.map((c) => (
                <th key={c.id} className="p-2 font-mono">{c.id}</th>
              ))}
              <th className="p-2">הערות</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <tr key={r.entity_name} className="border-t border-slate-100">
                <td className="p-2 font-medium" dir="auto">{r.entity_name}</td>
                {plan.columns.map((c) => {
                  const cell = r.cells[c.id];
                  if (!cell || cell.value === null) {
                    return <td key={c.id} className="p-2 text-slate-400">לא נמצא</td>;
                  }
                  return (
                    <td key={c.id} className="p-2 align-top">
                      <div dir="auto">{cell.value}</div>
                      <div className="text-xs text-slate-500 mt-0.5">
                        [{cell.confidence}]
                        {cell.primary_source?.url && (
                          <>
                            {" · "}
                            <a
                              href={cell.primary_source.url}
                              target="_blank"
                              rel="noreferrer"
                              className="text-blue-600 hover:underline font-mono"
                              title={cell.primary_source.url}
                            >
                              {cell.primary_source.domain ?? new URL(cell.primary_source.url).hostname}
                            </a>
                            {cell.primary_source.date && (
                              <span className={
                                cell.flags?.some(f => f.startsWith("all_sources_stale") || f.startsWith("primary_source_stale"))
                                  ? "text-amber-600 mr-1"
                                  : "text-slate-400 mr-1"
                              }>
                                {" "}· {cell.primary_source.date.slice(0, 7)}
                              </span>
                            )}
                          </>
                        )}
                        {cell.flags?.some(f => f.startsWith("all_sources_stale")) && (
                          <span className="text-amber-600 mr-1" title="כל המקורות ישנים מ-2 שנים — ייתכן שהמידע אינו עדכני">
                            {" "}⚠ ישן
                          </span>
                        )}
                      </div>
                    </td>
                  );
                })}
                <td className="p-2 text-xs text-slate-500">{r.row_flags.join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {status === "done" && results.length > 0 && doneInfo?.session_id && (
        <FeedbackPanel results={results} plan={plan} sessionId={doneInfo.session_id} />
      )}
    </div>
  );
}
