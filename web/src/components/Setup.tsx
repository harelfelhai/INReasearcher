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
  "For each Israeli municipality, find: (a) who served as mayor in 1990 (full name), (b) the official municipality website URL.",
  "For each member of the 25th Knesset, find: party, faction at election, year first elected.",
];
const BAD = [
  ["Tell me about Israeli mayors.", "no fields, no scope, no timeframe"],
  ["For each city, describe the mayor's career path.", "‘career path’ is unbounded"],
  ["Who was the best mayor of each city?", "‘best’ is subjective"],
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
      <section className="bg-white border border-slate-200 rounded-lg p-5">
        <h2 className="font-semibold mb-2">How to write a good research question</h2>
        <p className="text-sm text-slate-600 mb-4">
          A good question = a clearly defined table. Each entity is a row, each requested
          data point is a column with one verifiable value.
        </p>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <div className="text-sm font-medium text-emerald-700 mb-1">✓ Good</div>
            <ul className="text-sm text-slate-700 space-y-2">
              {GOOD.map((g) => (
                <li key={g} className="border-l-2 border-emerald-400 pl-2">{g}</li>
              ))}
            </ul>
          </div>
          <div>
            <div className="text-sm font-medium text-rose-700 mb-1">✗ Bad</div>
            <ul className="text-sm text-slate-700 space-y-2">
              {BAD.map(([q, why]) => (
                <li key={q} className="border-l-2 border-rose-400 pl-2">
                  {q} <span className="text-slate-500">— {why}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Research question</label>
          <textarea
            className="w-full border border-slate-300 rounded p-2 text-sm font-mono"
            rows={4}
            placeholder="e.g. עבור כל עיריה ישראלית מצא מי היה ראש העיר ב-1990 ואת כתובת האתר הרשמי."
            value={props.question}
            onChange={(e) => props.setQuestion(e.target.value)}
            dir="auto"
          />
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">Entity type hint (optional)</label>
            <input
              className="w-full border border-slate-300 rounded p-2 text-sm"
              placeholder="e.g. עיריה ישראלית"
              value={props.entityType}
              onChange={(e) => props.setEntityType(e.target.value)}
              dir="auto"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Search engine</label>
            <select
              className="w-full border border-slate-300 rounded p-2 text-sm"
              value={props.searchEngine}
              onChange={(e) => props.setSearchEngine(e.target.value as SearchEngine)}
            >
              <option value="serpapi">SerpAPI (Google)</option>
              <option value="wikipedia">Wikipedia</option>
              <option value="duckduckgo">DuckDuckGo</option>
              <option value="mock">Mock (no API)</option>
            </select>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Entities (one per line)</label>
          <textarea
            className="w-full border border-slate-300 rounded p-2 text-sm font-mono"
            rows={6}
            placeholder={"תל אביב\nחיפה\nירושלים"}
            value={props.entitiesText}
            onChange={(e) => props.setEntitiesText(e.target.value)}
            dir="auto"
          />
        </div>

        {props.clarification && (
          <div className="bg-amber-50 border border-amber-200 rounded p-3 text-sm">
            <div className="font-medium text-amber-900 mb-1">Needs clarification</div>
            <div className="text-amber-900 mb-2">{props.clarification.reason}</div>
            <ul className="list-disc list-inside text-amber-900 space-y-1">
              {props.clarification.questions.map((q) => (
                <li key={q.field}>
                  <span className="font-mono">[{q.field}]</span> {q.question_en}{" "}
                  <span className="text-amber-700">(example: {q.example})</span>
                </li>
              ))}
            </ul>
            {props.clarification.prompt_template && (
              <pre className="mt-2 bg-white border border-amber-200 rounded p-2 text-xs whitespace-pre-wrap">
                {props.clarification.prompt_template}
              </pre>
            )}
          </div>
        )}

        {error && (
          <div className="bg-rose-50 border border-rose-200 text-rose-900 rounded p-3 text-sm">
            {error}
          </div>
        )}

        <div className="flex justify-end">
          <button
            onClick={build}
            disabled={loading || !props.question.trim()}
            className="bg-slate-900 text-white px-4 py-2 rounded text-sm font-medium disabled:opacity-50"
          >
            {loading ? "Building schema…" : "Build Schema →"}
          </button>
        </div>
      </section>
    </div>
  );
}
