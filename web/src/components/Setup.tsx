import { useState } from "react";
import { auditPlan, compileSchema, mockPreview } from "../api";
import type {
  ClarificationRequest,
  FieldAuditReport,
  MockRow,
  ResearchPlan,
  SearchEngine,
} from "../types";

interface Props {
  question: string;
  setQuestion: (s: string) => void;
  entityType: string;
  setEntityType: (s: string) => void;
  entitiesText: string;
  setEntitiesText: (s: string) => void;
  searchEngine: SearchEngine;
  setSearchEngine: (e: SearchEngine) => void;
  clarification: ClarificationRequest | null;
  onCompiled: (
    plan: ResearchPlan | null,
    audit: FieldAuditReport | null,
    mockRows: MockRow[],
    clarification: ClarificationRequest | null,
  ) => void;
}

const GOOD = [
  "עבור כל עיריה ישראלית, מצא: (א) מי כיהן כראש העיר ב-1990 (שם מלא), (ב) כתובת האתר הרשמי של העיריה.",
  "עבור כל חבר/ת כנסת ה-25, מצא: מפלגה, סיעה בעת הבחירה, שנת היבחרות ראשונה.",
];
const BAD: [string, string][] = [
  ["ספר לי על ראשי ערים בישראל.", "אין שדות, אין היקף, אין מסגרת זמן"],
  ["עבור כל עיר, תאר את מסלול הקריירה של ראש העיר.", "‘מסלול קריירה’ הוא בלתי תחום"],
  ["מי היה ראש העיר הטוב ביותר בכל עיר?", "‘הטוב ביותר’ הוא סובייקטיבי"],
];

export default function Setup(props: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function build() {
    setLoading(true);
    setError(null);
    try {
      const compiled = await compileSchema(props.question, props.entityType);
      if (compiled.kind === "clarification") {
        props.onCompiled(null, null, [], compiled.clarification ?? null);
        return;
      }
      const plan = compiled.plan!;
      const [audit, mockRows] = await Promise.all([
        auditPlan(plan),
        mockPreview(plan).catch(() => []),
      ]);
      props.onCompiled(plan, audit, mockRows, null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <section className="bg-white border border-slate-200 rounded-xl shadow-sm p-5">
        <h2 className="font-semibold mb-2">איך לכתוב שאלת מחקר טובה</h2>
        <p className="text-sm text-slate-600 mb-4">
          שאלה טובה = טבלה מוגדרת היטב. כל ישות היא שורה, כל נתון מבוקש הוא עמודה
          עם ערך אחד שניתן לאמת ממקור.
        </p>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <div className="text-sm font-medium text-emerald-700 mb-1">✓ דוגמאות טובות</div>
            <ul className="text-sm text-slate-700 space-y-2">
              {GOOD.map((g) => (
                <li key={g} className="border-r-2 border-emerald-400 pr-2">{g}</li>
              ))}
            </ul>
          </div>
          <div>
            <div className="text-sm font-medium text-rose-700 mb-1">✗ דוגמאות גרועות</div>
            <ul className="text-sm text-slate-700 space-y-2">
              {BAD.map(([q, why]) => (
                <li key={q} className="border-r-2 border-rose-400 pr-2">
                  {q} <span className="text-slate-500">— {why}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className="bg-white border border-slate-200 rounded-xl shadow-sm p-5 space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">שאלת מחקר</label>
          <textarea
            className="w-full border border-slate-300 rounded-lg p-3 text-base bg-white"
            rows={4}
            placeholder="לדוגמה: עבור כל עיריה ישראלית מצא מי היה ראש העיר ב-1990 ואת כתובת האתר הרשמי."
            value={props.question}
            onChange={(e) => props.setQuestion(e.target.value)}
            dir="auto"
          />
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">סוג הישות (אופציונלי)</label>
            <input
              className="w-full border border-slate-300 rounded-lg p-3 text-base bg-white"
              placeholder="לדוגמה: עיריה ישראלית"
              value={props.entityType}
              onChange={(e) => props.setEntityType(e.target.value)}
              dir="auto"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">מנוע חיפוש</label>
            <select
              className="w-full border border-slate-300 rounded-lg p-3 text-base bg-white"
              value={props.searchEngine}
              onChange={(e) => props.setSearchEngine(e.target.value as SearchEngine)}
            >
              <option value="serpapi">SerpAPI (Google)</option>
              <option value="wikipedia">Wikipedia</option>
              <option value="duckduckgo">DuckDuckGo</option>
              <option value="mock">דמה (ללא API)</option>
            </select>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">ישויות (אחת בכל שורה)</label>
          <textarea
            className="w-full border border-slate-300 rounded-lg p-3 text-base bg-white"
            rows={6}
            placeholder={"תל אביב\nחיפה\nירושלים"}
            value={props.entitiesText}
            onChange={(e) => props.setEntitiesText(e.target.value)}
            dir="auto"
          />
        </div>

        {props.clarification && (
          <div className="bg-amber-50 border border-amber-200 rounded p-3 text-sm">
            <div className="font-medium text-amber-900 mb-1">נדרשת הבהרה</div>
            <div className="text-amber-900 mb-2">{props.clarification.reason}</div>
            <ul className="list-disc list-inside text-amber-900 space-y-1">
              {props.clarification.questions.map((q) => (
                <li key={q.field}>
                  <span className="font-mono">[{q.field}]</span> {q.question_he}{" "}
                  <span className="text-amber-700">(דוגמה: {q.example})</span>
                </li>
              ))}
            </ul>
            {props.clarification.prompt_template && (
              <pre className="mt-2 bg-white border border-amber-200 rounded p-2 text-xs whitespace-pre-wrap" dir="auto">
                {props.clarification.prompt_template}
              </pre>
            )}
          </div>
        )}

        {error && (
          <div className="bg-rose-50 border border-rose-200 text-rose-900 rounded p-3 text-sm" dir="auto">
            {error}
          </div>
        )}

        <div className="flex justify-end">
          <button
            onClick={build}
            disabled={loading || !props.question.trim()}
            className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-lg text-sm font-semibold shadow-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "בונה סכמה…" : "בנה סכמה ‹"}
          </button>
        </div>
      </section>
    </div>
  );
}
