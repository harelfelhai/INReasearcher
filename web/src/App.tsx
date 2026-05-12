import { useState } from "react";
import type {
  ClarificationRequest,
  FieldAuditReport,
  MockRow,
  ResearchPlan,
  SearchEngine,
} from "./types";
import Setup from "./components/Setup";
import SchemaReview from "./components/SchemaReview";
import Results from "./components/Results";

type Step = "setup" | "review" | "results";

export default function App() {
  const [step, setStep] = useState<Step>("setup");

  const [question, setQuestion] = useState("");
  const [entityType, setEntityType] = useState("");
  const [entitiesText, setEntitiesText] = useState("");
  const [searchEngine, setSearchEngine] = useState<SearchEngine>("serpapi");

  const [clarification, setClarification] = useState<ClarificationRequest | null>(null);
  const [plan, setPlan] = useState<ResearchPlan | null>(null);
  const [audit, setAudit] = useState<FieldAuditReport | null>(null);
  const [mockRows, setMockRows] = useState<MockRow[]>([]);
  const [enrichedPlan, setEnrichedPlan] = useState<ResearchPlan | null>(null);

  const entities = entitiesText
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <h1 className="text-lg font-semibold">Autonomous Research Agent</h1>
          <nav className="text-sm text-slate-500 flex gap-3">
            <span className={step === "setup" ? "text-slate-900 font-medium" : ""}>1. Setup</span>
            <span>›</span>
            <span className={step === "review" ? "text-slate-900 font-medium" : ""}>2. Review</span>
            <span>›</span>
            <span className={step === "results" ? "text-slate-900 font-medium" : ""}>3. Results</span>
          </nav>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8">
        {step === "setup" && (
          <Setup
            question={question}
            setQuestion={setQuestion}
            entityType={entityType}
            setEntityType={setEntityType}
            entitiesText={entitiesText}
            setEntitiesText={setEntitiesText}
            searchEngine={searchEngine}
            setSearchEngine={setSearchEngine}
            clarification={clarification}
            onCompiled={(plan, audit, mockRows, clarification) => {
              setClarification(clarification);
              setPlan(plan);
              setAudit(audit);
              setMockRows(mockRows);
              if (plan) setStep("review");
            }}
          />
        )}

        {step === "review" && plan && audit && (
          <SchemaReview
            plan={plan}
            audit={audit}
            mockRows={mockRows}
            onBack={() => setStep("setup")}
            onApproved={(enriched) => {
              setEnrichedPlan(enriched);
              setStep("results");
            }}
          />
        )}

        {step === "results" && enrichedPlan && (
          <Results
            plan={enrichedPlan}
            entities={entities}
            searchEngine={searchEngine}
            onRestart={() => {
              setStep("setup");
              setPlan(null);
              setAudit(null);
              setMockRows([]);
              setEnrichedPlan(null);
              setClarification(null);
            }}
          />
        )}
      </main>
    </div>
  );
}
