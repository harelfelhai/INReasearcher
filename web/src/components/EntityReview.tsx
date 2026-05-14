import { useEffect, useState } from "react";
import { planDiscovery, runDiscovery } from "../api";
import type {
  EntityDiscoveryAuditIssue,
  EntityDiscoveryPlan,
  EntityDiscoveryResult,
  HarvestedValue,
  ResearchPlan,
  SearchEngine,
} from "../types";

interface Props {
  plan: ResearchPlan;
  question: string;
  entityType: string;
  searchEngine: SearchEngine;
  onBack: () => void;
  onApproved: (entities: string[], harvested: HarvestedValue[], sourceUrl: string) => void;
}

const ISSUE_KIND_LABEL: Record<string, string> = {
  unbounded_count: "כמות לא מוגדרת",
  ambiguous_ranking: "קריטריון דירוג עמום",
  missing_anchor: "חסר עיגון בזמן",
  subjective_criterion: "קריטריון סובייקטיבי",
  no_canonical_source: "אין מקור קנוני אמין",
};

export default function EntityReview(props: Props) {
  const [discoveryPlan, setDiscoveryPlan] = useState<EntityDiscoveryPlan | null>(null);
  const [result, setResult] = useState<EntityDiscoveryResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [editedNames, setEditedNames] = useState<string[]>([]);
  const [acknowledged, setAcknowledged] = useState(false);

  // Step 1: get the discovery plan + audit when the screen opens
  useEffect(() => {
    (async () => {
      setBusy(true);
      setErr(null);
      try {
        const dp = await planDiscovery(props.question, props.entityType);
        setDiscoveryPlan(dp);
      } catch (e) {
        setErr((e as Error).message);
      } finally {
        setBusy(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function runDiscover() {
    if (!discoveryPlan) return;
    setBusy(true);
    setErr(null);
    try {
      const r = await runDiscovery(props.plan, discoveryPlan, props.searchEngine);
      if (!r || r.entities.length === 0) {
        setErr("לא נמצאו ישויות בעמוד שהמערכת אחזרה. נסה לחדד את השאלה.");
      } else {
        setResult(r);
        setEditedNames(r.entities.map((e) => e.name));
      }
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function approve() {
    if (!result) return;
    const cleaned = editedNames.map((s) => s.trim()).filter(Boolean);
    // Only keep harvest rows whose entity_name still appears in the edited list.
    const allowed = new Set(cleaned);
    const harvested = result.harvested.filter((h) => allowed.has(h.entity_name));
    props.onApproved(cleaned, harvested, result.source_url);
  }

  const issues: EntityDiscoveryAuditIssue[] = discoveryPlan?.audit_issues ?? [];
  const clean = discoveryPlan && issues.length === 0;

  return (
    <div className="space-y-4">
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-5">
        <h2 className="text-base font-semibold mb-1">אימות רשימת ישויות</h2>
        <p className="text-sm text-slate-600">
          המערכת תנסה למצוא את רשימת הישויות הנדרשת מהשאלה. בדוק את שאילתת
          החיפוש לפני הגילוי, ואשר את הרשימה לפני הרצת המחקר.
        </p>
      </div>

      {err && (
        <div className="bg-rose-50 border border-rose-200 text-rose-900 rounded p-3 text-sm" dir="auto">
          {err}
        </div>
      )}

      {discoveryPlan && (
        <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-5 space-y-3">
          <h3 className="font-semibold text-sm">שאילתת גילוי שתופעל</h3>
          <div className="text-sm border border-slate-200 rounded p-2 bg-slate-50" dir="auto">
            <div>HE: <span className="font-mono">{discoveryPlan.query_he}</span></div>
            <div>EN: <span className="font-mono" dir="ltr">{discoveryPlan.query_en}</span></div>
            {discoveryPlan.expected_count && (
              <div className="text-slate-600 mt-1">
                מספר צפוי: {discoveryPlan.expected_count}
              </div>
            )}
            {discoveryPlan.extraction_hint && (
              <div className="text-slate-600 mt-1">
                רמז חילוץ: {discoveryPlan.extraction_hint}
              </div>
            )}
          </div>

          {issues.length > 0 && (
            <div className="bg-amber-50 border border-amber-200 rounded p-3 text-sm space-y-2">
              <div className="font-medium text-amber-900">⚠ בעיות בשאלת הישויות</div>
              {issues.map((iss, i) => (
                <div key={i} className="border-r-2 border-amber-400 pr-2">
                  <div className="text-amber-900">
                    <span className="font-mono text-xs">
                      [{ISSUE_KIND_LABEL[iss.issue_kind] ?? iss.issue_kind}]
                    </span>{" "}
                    {iss.explanation_he}
                  </div>
                  <div className="text-amber-800 text-xs mt-0.5">
                    תיקון מוצע: {iss.suggested_fix_he}
                  </div>
                </div>
              ))}
              <label className="flex items-center gap-2 text-amber-900 pt-1">
                <input
                  type="checkbox"
                  checked={acknowledged}
                  onChange={(e) => setAcknowledged(e.target.checked)}
                />
                אני מבין/ה את הסיכונים ורוצה להמשיך בכל זאת
              </label>
            </div>
          )}

          {!result && (
            <div className="flex gap-2 justify-end">
              <button
                onClick={props.onBack}
                className="px-3 py-1.5 rounded-lg border border-slate-300 bg-white text-sm hover:bg-slate-50"
              >
                חזרה
              </button>
              <button
                onClick={runDiscover}
                disabled={busy || (issues.length > 0 && !acknowledged)}
                className="px-3 py-1.5 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
              >
                {busy ? "מגלה ישויות…" : clean ? "הפעל גילוי" : "המשך בכל זאת"}
              </button>
            </div>
          )}
        </div>
      )}

      {result && (
        <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-sm">ישויות שזוהו ({result.entities.length})</h3>
            <a
              href={result.source_url}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-blue-600 hover:underline"
            >
              מקור: {result.source_domain}
            </a>
          </div>

          <p className="text-xs text-slate-500">
            ערוך, הסר או הוסף ישויות לפני האישור. בנוסף, נחלצו{" "}
            <span className="font-mono">{result.harvested.length}</span> ערכי שדות
            מהעמוד שיוזרמו ישירות לתוצאות (חוסך חיפושים).
          </p>

          <ul className="space-y-1">
            {editedNames.map((name, i) => (
              <li key={i} className="flex items-center gap-2">
                <span className="text-xs text-slate-400 w-6 text-left font-mono">
                  {i + 1}.
                </span>
                <input
                  value={name}
                  onChange={(e) => {
                    const next = [...editedNames];
                    next[i] = e.target.value;
                    setEditedNames(next);
                  }}
                  dir="auto"
                  className="flex-1 border border-slate-300 rounded px-2 py-1 text-sm"
                />
                <button
                  onClick={() => setEditedNames(editedNames.filter((_, j) => j !== i))}
                  className="text-xs text-rose-600 hover:underline"
                >
                  הסר
                </button>
                {result.entities[i]?.quote && (
                  <span className="text-xs text-slate-400 truncate max-w-[14rem]" title={result.entities[i].quote ?? ""}>
                    {result.entities[i].quote}
                  </span>
                )}
              </li>
            ))}
            <li>
              <button
                onClick={() => setEditedNames([...editedNames, ""])}
                className="text-xs text-blue-600 hover:underline mt-1"
              >
                + הוסף ישות
              </button>
            </li>
          </ul>

          {result.harvested.length > 0 && (
            <details className="text-xs text-slate-600">
              <summary className="cursor-pointer">הצג ערכים שכבר נחלצו ({result.harvested.length})</summary>
              <table className="w-full mt-2 border-t border-slate-200">
                <thead>
                  <tr className="text-right">
                    <th className="p-1">ישות</th>
                    <th className="p-1">שדה</th>
                    <th className="p-1">ערך</th>
                  </tr>
                </thead>
                <tbody>
                  {result.harvested.map((h: HarvestedValue, i) => (
                    <tr key={i} className="border-t border-slate-100">
                      <td className="p-1" dir="auto">{h.entity_name}</td>
                      <td className="p-1 font-mono">{h.field_id}</td>
                      <td className="p-1" dir="auto">{h.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          )}

          <div className="flex gap-2 justify-end pt-2">
            <button
              onClick={props.onBack}
              className="px-3 py-1.5 rounded-lg border border-slate-300 bg-white text-sm hover:bg-slate-50"
            >
              חזרה
            </button>
            <button
              onClick={approve}
              disabled={editedNames.filter((s) => s.trim()).length === 0}
              className="px-3 py-1.5 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
            >
              אשר והמשך למחקר
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
