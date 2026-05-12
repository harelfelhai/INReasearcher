import { useState } from "react";
import { enrichPlan } from "../api";
import type { FieldAuditReport, MockRow, ResearchPlan } from "../types";

interface Props {
  plan: ResearchPlan;
  audit: FieldAuditReport;
  mockRows: MockRow[];
  onBack: () => void;
  onApproved: (enriched: ResearchPlan) => void;
}

export default function SchemaReview({ plan, audit, mockRows, onBack, onApproved }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function approve() {
    setLoading(true);
    setError(null);
    try {
      const enriched = await enrichPlan(plan);
      onApproved(enriched);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <section className="bg-white border border-slate-200 rounded-lg p-5">
        <h2 className="font-semibold mb-3">סכמה</h2>
        <div className="text-sm text-slate-500 mb-3">
          סוג ישות: <span className="font-mono text-slate-900" dir="auto">{plan.entity_type}</span>
        </div>
        <div className="overflow-x-auto">
          <table className="text-sm w-full">
            <thead className="bg-slate-100 text-right">
              <tr>
                <th className="p-2">מזהה</th>
                <th className="p-2">תווית (עברית)</th>
                <th className="p-2">תווית (אנגלית)</th>
                <th className="p-2">סוג</th>
                <th className="p-2">עיגון זמן</th>
                <th className="p-2">תלוי ב</th>
              </tr>
            </thead>
            <tbody>
              {plan.columns.map((c) => (
                <tr key={c.id} className="border-t border-slate-100">
                  <td className="p-2 font-mono">{c.id}</td>
                  <td className="p-2" dir="auto">{c.label_he}</td>
                  <td className="p-2" dir="auto">{c.label_en}</td>
                  <td className="p-2 font-mono text-slate-600">{c.type}</td>
                  <td className="p-2 font-mono text-slate-600">{c.temporal_anchor || "—"}</td>
                  <td className="p-2 font-mono text-slate-600">{c.depends_on || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="bg-white border border-slate-200 rounded-lg p-5">
        <h2 className="font-semibold mb-3">ביקורת בהירות שדות</h2>
        {audit.all_clear ? (
          <div className="text-sm text-emerald-700">✓ כל השדות נראים ברורים.</div>
        ) : (
          <ul className="space-y-3">
            {audit.issues.map((i, idx) => (
              <li key={idx} className="bg-amber-50 border border-amber-200 rounded p-3 text-sm">
                <div className="font-medium text-amber-900">
                  [{i.field_id}] — {i.issue_kind}
                </div>
                <div className="text-amber-900 mt-1" dir="auto">{i.explanation_he}</div>
                <div className="text-amber-700 mt-1" dir="auto">{i.explanation_en}</div>
                <div className="text-amber-800 mt-2" dir="auto">
                  <span className="font-medium">תיקון מוצע: </span>{i.suggested_fix_he}
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="bg-white border border-slate-200 rounded-lg p-5">
        <h2 className="font-semibold mb-1">תצוגה מקדימה (נתונים מדומים)</h2>
        <p className="text-sm text-slate-500 mb-3">
          ערכים מדומים שמדגימים את צורת הטבלה בלבד — שום דבר עוד לא נחקר.
        </p>
        {mockRows.length === 0 ? (
          <div className="text-sm text-slate-500">לא נוצרה תצוגה מקדימה.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="text-sm w-full">
              <thead className="bg-slate-100 text-right">
                <tr>
                  <th className="p-2">ישות</th>
                  {plan.columns.map((c) => (
                    <th key={c.id} className="p-2 font-mono">{c.id}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {mockRows.map((r, idx) => (
                  <tr key={idx} className="border-t border-slate-100">
                    <td className="p-2" dir="auto">{r.entity_name}</td>
                    {plan.columns.map((c) => (
                      <td key={c.id} className="p-2 text-slate-600" dir="auto">
                        {r.values[c.id] || "—"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {error && (
        <div className="bg-rose-50 border border-rose-200 text-rose-900 rounded p-3 text-sm" dir="auto">
          {error}
        </div>
      )}

      <div className="flex justify-between">
        <button onClick={onBack} className="px-4 py-2 rounded border border-slate-300 text-sm">
          › חזרה לעריכת השאלה
        </button>
        <button
          onClick={approve}
          disabled={loading}
          className="bg-slate-900 text-white px-4 py-2 rounded text-sm font-medium disabled:opacity-50"
        >
          {loading ? "מתכנן שאילתות חיפוש…" : "אשר והרץ ‹"}
        </button>
      </div>
    </div>
  );
}
