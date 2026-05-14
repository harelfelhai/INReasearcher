import { useEffect, useRef, useState } from "react";
import { downloadExport, runResearch, type RunDonePayload, type SeededProbe } from "../api";
import type { EntityResult, ResearchPlan, SearchEngine } from "../types";

interface Props {
  plan: ResearchPlan;
  entities: string[];
  searchEngine: SearchEngine;
  seededProbe?: SeededProbe;
  onRestart: () => void;
}

export default function Results({ plan, entities, searchEngine, seededProbe, onRestart }: Props) {
  const [results, setResults] = useState<EntityResult[]>([]);
  const [current, setCurrent] = useState<string | null>(null);
  const [status, setStatus] = useState<"running" | "done" | "error">("running");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [doneInfo, setDoneInfo] = useState<RunDonePayload | null>(null);
  const ctrlRef = useRef<AbortController | null>(null);

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
                              className="text-blue-600 hover:underline"
                            >
                              מקור
                            </a>
                          </>
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
    </div>
  );
}
