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
    <div className="min-h-screen text-slate-900">
      <header className="bg-gradient-to-l from-blue-700 to-blue-600 text-white shadow-sm">
        <div className="max-w-5xl mx-auto px-6 py-5 flex items-center justify-between">
          <h1 className="text-xl font-semibold tracking-tight">סוכן מחקר אוטונומי</h1>
          <nav className="text-sm flex gap-3 items-center">
            <span className={step === "setup" ? "text-white font-semibold" : "text-blue-100"}>1. הגדרה</span>
            <span className="text-blue-200">‹</span>
            <span className={step === "review" ? "text-white font-semibold" : "text-blue-100"}>2. סקירה</span>
            <span className="text-blue-200">‹</span>
            <span className={step === "results" ? "text-white font-semibold" : "text-blue-100"}>3. תוצאות</span>
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
