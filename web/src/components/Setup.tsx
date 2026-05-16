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
  autoDiscover: boolean;
  setAutoDiscover: (b: boolean) => void;
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

// Map machine-readable field codes from the preflight LLM to Hebrew labels.
const FIELD_LABELS_HE: Record<string, string> = {
  entity_type: "סוג ישות",
  entity_specification: "פירוט הישויות",
  specific_fields: "שדות לאיסוף",
  fields: "שדות לאיסוף",
  field_ambiguity: "עמימות בשדה",
  temporal_context: "מסגרת זמן",
  timeframe: "מסגרת זמן",
  time_period: "מסגרת זמן",
  geographic_scope: "היקף גאוגרפי",
  jurisdiction: "תחום שיפוט",
  scope: "היקף",
  data_completeness_expectation: "ציפיית שלמות נתונים",
  data_completeness: "ציפיית שלמות נתונים",
};

function fieldLabelHe(field: string): string {
  const key = field.toLowerCase().replace(/[\s-]+/g, "_");
  return FIELD_LABELS_HE[key] ?? field;
}

export default function Setup(props: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Per-question answers for the clarification panel, keyed by question field.
  const [answers, setAnswers] = useState<Record<string, string>>({});

  function parsedEntities(): string[] {
    if (props.autoDiscover) return [];
    return props.entitiesText
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
  }

  async function build() {
    setLoading(true);
    setError(null);
    try {
      const compiled = await compileSchema(
        props.question,
        props.entityType,
        parsedEntities(),
      );
      if (compiled.kind === "clarification") {
        setAnswers({});
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
          <div className="flex items-center justify-between mb-1">
            <label className="block text-sm font-medium">
              ישויות {props.autoDiscover ? "(יתגלו אוטומטית)" : "(אחת בכל שורה)"}
            </label>
            <label className="text-xs text-slate-600 flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={props.autoDiscover}
                onChange={(e) => props.setAutoDiscover(e.target.checked)}
              />
              גלה ישויות אוטומטית מהשאלה
            </label>
          </div>
          {props.autoDiscover ? (
            <div className="border border-amber-200 bg-amber-50 rounded-lg p-3 text-sm text-amber-900 space-y-1">
              <div className="font-medium">⚠ מצב גילוי אוטומטי</div>
              <p>
                המערכת תנסה לחלץ את רשימת הישויות מהשאלה (למשל "10 הערים הגדולות בישראל").
                לפני הרצת המחקר המלא תוצג רשימה לאישור.
              </p>
              <p className="text-amber-800">
                <strong>גילוי אוטומטי פחות מדויק ויקר יותר.</strong> אם הרשימה ידועה לך —
                עדיף לספק אותה ידנית.
              </p>
            </div>
          ) : (
            <textarea
              className="w-full border border-slate-300 rounded-lg p-3 text-base bg-white"
              rows={6}
              placeholder={"תל אביב\nחיפה\nירושלים"}
              value={props.entitiesText}
              onChange={(e) => props.setEntitiesText(e.target.value)}
              dir="auto"
            />
          )}
        </div>

        {props.clarification && (
          <div className="bg-amber-50 border border-amber-200 rounded p-3 text-sm space-y-3">
            <div>
              <div className="font-medium text-amber-900 mb-1">נדרשת הבהרה</div>
              <div className="text-amber-900">{props.clarification.reason}</div>
            </div>

            <div className="space-y-2">
              {props.clarification.questions.map((q) => (
                <div key={q.field} className="bg-white border border-amber-200 rounded p-2">
                  <label className="block text-amber-900 mb-1">
                    <span className="inline-block bg-amber-200 text-amber-900 rounded px-1.5 py-0.5 text-xs ml-2">
                      {fieldLabelHe(q.field)}
                    </span>
                    {q.question_he}
                  </label>
                  <div className="text-xs text-amber-700 mb-1.5" dir="auto">
                    דוגמה: {q.example}
                  </div>
                  <input
                    className="w-full border border-amber-300 rounded p-1.5 text-sm bg-white"
                    placeholder={q.example}
                    value={answers[q.field] ?? ""}
                    onChange={(e) =>
                      setAnswers({ ...answers, [q.field]: e.target.value })
                    }
                    dir="auto"
                  />
                </div>
              ))}
            </div>

            <div className="flex justify-end">
              <button
                type="button"
                className="bg-amber-600 hover:bg-amber-700 text-white px-3 py-1.5 rounded text-xs font-semibold disabled:opacity-50"
                disabled={
                  props.clarification.questions.length === 0 ||
                  props.clarification.questions.every((q) => !(answers[q.field] ?? "").trim())
                }
                onClick={() => {
                  const tpl = props.clarification?.prompt_template ?? "";
                  let nextQuestion = tpl || props.question;
                  for (const q of props.clarification?.questions ?? []) {
                    const ans = (answers[q.field] ?? "").trim();
                    if (!ans) continue;
                    // Replace [PLACEHOLDER: …] markers tied to this field, or
                    // any [PLACEHOLDER …] if we can't disambiguate.
                    const fieldRe = new RegExp(
                      `\\[PLACEHOLDER[^\\]]*${q.field}[^\\]]*\\]`,
                      "gi",
                    );
                    if (fieldRe.test(nextQuestion)) {
                      nextQuestion = nextQuestion.replace(fieldRe, ans);
                    } else {
                      // Fallback: replace the first remaining [PLACEHOLDER …].
                      nextQuestion = nextQuestion.replace(/\[PLACEHOLDER[^\]]*\]/i, ans);
                    }
                  }
                  // Any leftover placeholders → append answers as a clarifying suffix.
                  const leftover = /\[PLACEHOLDER[^\]]*\]/i.test(nextQuestion);
                  if (!tpl || leftover) {
                    const suffix = (props.clarification?.questions ?? [])
                      .map((q) => {
                        const a = (answers[q.field] ?? "").trim();
                        return a ? `${fieldLabelHe(q.field)}: ${a}` : null;
                      })
                      .filter(Boolean)
                      .join("; ");
                    if (suffix) {
                      nextQuestion = nextQuestion.replace(/\[PLACEHOLDER[^\]]*\]/gi, "").trim();
                      nextQuestion = `${nextQuestion}\n(${suffix})`;
                    }
                  }
                  props.setQuestion(nextQuestion);
                  // Reset the clarification panel so the user can re-submit.
                  props.onCompiled(null, null, [], null);
                  setAnswers({});
                }}
              >
                החל תשובות על השאלה
              </button>
            </div>

            {props.clarification.prompt_template && (
              <details className="text-xs text-amber-800">
                <summary className="cursor-pointer">תבנית מוצעת (לעריכה ידנית)</summary>
                <pre
                  className="mt-2 bg-white border border-amber-200 rounded p-2 whitespace-pre-wrap"
                  dir="auto"
                >
                  {props.clarification.prompt_template}
                </pre>
              </details>
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
