import { useEffect, useState } from "react";
import { downloadExport, listOwnSessions } from "../api";
import type { SessionOut } from "../types";

interface Props {
  onNewResearch: () => void;
}

export default function UserDashboard({ onNewResearch }: Props) {
  const [sessions, setSessions] = useState<SessionOut[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function refresh() {
    setLoading(true);
    setErr(null);
    try {
      setSessions(await listOwnSessions());
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  return (
    <div className="space-y-4">
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-5 flex items-center justify-between">
        <div className="text-sm">
          <h2 className="text-base font-semibold">היסטוריית מחקרים</h2>
          <p className="text-slate-500 mt-0.5">כל השאלות שהרצת והקבצים שנוצרו</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={refresh}
            className="px-3 py-1.5 rounded-lg border border-slate-300 bg-white text-sm hover:bg-slate-50"
          >
            רענן
          </button>
          <button
            onClick={onNewResearch}
            className="px-3 py-1.5 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700"
          >
            מחקר חדש
          </button>
        </div>
      </div>

      {err && (
        <div className="text-sm text-rose-700 bg-rose-50 border border-rose-200 rounded-lg p-3">
          {err}
        </div>
      )}

      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-x-auto">
        <table className="text-sm w-full">
          <thead className="bg-slate-100 text-right">
            <tr>
              <th className="p-2 w-1/2">שאלה</th>
              <th className="p-2">סטטוס</th>
              <th className="p-2">עלות (USD)</th>
              <th className="p-2">תאריך</th>
              <th className="p-2">קובץ</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={5} className="p-4 text-center text-slate-500">טוען…</td>
              </tr>
            )}
            {!loading && sessions.length === 0 && (
              <tr>
                <td colSpan={5} className="p-4 text-center text-slate-500">אין עדיין מחקרים</td>
              </tr>
            )}
            {sessions.map((s) => (
              <tr key={s.id} className="border-t border-slate-100 align-top">
                <td className="p-2" dir="auto">{s.question}</td>
                <td className="p-2">
                  <span
                    className={
                      s.status === "completed"
                        ? "text-emerald-700"
                        : s.status === "failed"
                        ? "text-rose-700"
                        : "text-amber-700"
                    }
                  >
                    {s.status}
                  </span>
                </td>
                <td className="p-2 font-mono">{s.cost_used.toFixed(4)}</td>
                <td className="p-2 text-slate-600">
                  {new Date(s.created_at).toLocaleString("he-IL")}
                </td>
                <td className="p-2">
                  {s.exports.length === 0 ? (
                    <span className="text-slate-400">—</span>
                  ) : (
                    s.exports.map((x) => (
                      <button
                        key={x.id}
                        onClick={() => downloadExport(x.id, x.filename)}
                        className="text-blue-600 hover:underline text-xs block"
                      >
                        ⬇ {x.filename}
                      </button>
                    ))
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
