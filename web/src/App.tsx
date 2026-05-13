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
import Login from "./components/Login";
import UserDashboard from "./components/UserDashboard";
import AdminDashboard from "./components/AdminDashboard";
import { AuthProvider, useAuth } from "./auth";

type View = "dashboard" | "setup" | "review" | "results" | "admin";

function AppShell() {
  const { user, loading, logout } = useAuth();
  const [view, setView] = useState<View>("dashboard");

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

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-slate-500">
        טוען…
      </div>
    );
  }

  if (!user) return <Login />;

  const isAdmin = user.role === "admin";

  function newResearch() {
    setQuestion("");
    setEntityType("");
    setEntitiesText("");
    setPlan(null);
    setAudit(null);
    setMockRows([]);
    setEnrichedPlan(null);
    setClarification(null);
    setView("setup");
  }

  return (
    <div className="min-h-screen text-slate-900">
      <header className="bg-gradient-to-l from-blue-700 to-blue-600 text-white shadow-sm">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">סוכן מחקר אוטונומי</h1>
            <p className="text-xs text-blue-100 mt-0.5">
              {isAdmin ? "ממשק מנהל" : "ממשק חוקר"} ·{" "}
              <span dir="ltr">{user.username}</span> · יתרה:{" "}
              <span className="font-mono">${user.credit_balance.toFixed(2)}</span>
            </p>
          </div>
          <nav className="text-sm flex gap-2 items-center">
            <button
              onClick={() => setView("dashboard")}
              className={`px-2 py-1 rounded ${view === "dashboard" ? "bg-white/20 font-semibold" : "hover:bg-white/10"}`}
            >
              היסטוריה
            </button>
            <button
              onClick={newResearch}
              className={`px-2 py-1 rounded ${view === "setup" || view === "review" || view === "results" ? "bg-white/20 font-semibold" : "hover:bg-white/10"}`}
            >
              מחקר חדש
            </button>
            {isAdmin && (
              <button
                onClick={() => setView("admin")}
                className={`px-2 py-1 rounded ${view === "admin" ? "bg-white/20 font-semibold" : "hover:bg-white/10"}`}
              >
                ניהול משתמשים
              </button>
            )}
            <button
              onClick={() => {
                logout();
                setView("dashboard");
              }}
              className="px-2 py-1 rounded hover:bg-white/10"
            >
              יציאה
            </button>
          </nav>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        {view === "dashboard" && <UserDashboard onNewResearch={newResearch} />}

        {view === "admin" && isAdmin && <AdminDashboard />}

        {view === "setup" && (
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
              if (plan) setView("review");
            }}
          />
        )}

        {view === "review" && plan && audit && (
          <SchemaReview
            plan={plan}
            audit={audit}
            mockRows={mockRows}
            onBack={() => setView("setup")}
            onApproved={(enriched) => {
              setEnrichedPlan(enriched);
              setView("results");
            }}
          />
        )}

        {view === "results" && enrichedPlan && (
          <Results
            plan={enrichedPlan}
            entities={entities}
            searchEngine={searchEngine}
            onRestart={() => setView("dashboard")}
          />
        )}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppShell />
    </AuthProvider>
  );
}
