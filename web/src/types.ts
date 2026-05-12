// Mirrors research_agent/models.py — keep manually in sync.

export type ColumnType =
  | "person_name"
  | "url"
  | "date"
  | "free_text"
  | "organization"
  | "number";

export type Confidence = "HIGH" | "MEDIUM" | "LOW" | "NOT_FOUND";

export interface ColumnPlan {
  id: string;
  label_he: string;
  label_en: string;
  type: ColumnType;
  temporal_anchor?: string | null;
  search_queries_he: string[];
  search_queries_en: string[];
  preferred_source_domains: string[];
  min_corroborations: number;
  depends_on?: string | null;
}

export interface ResearchPlan {
  entity_type: string;
  research_question_original: string;
  columns: ColumnPlan[];
}

export interface ClarificationQuestion {
  field: string;
  question_he: string;
  question_en: string;
  example: string;
}

export interface ClarificationRequest {
  is_executable: false;
  reason: string;
  questions: ClarificationQuestion[];
  prompt_template: string;
}

export type IssueKind =
  | "unbounded"
  | "subjective"
  | "missing_anchor"
  | "ambiguous_format"
  | "no_canonical_source";

export interface FieldAuditIssue {
  field_id: string;
  issue_kind: IssueKind;
  explanation_he: string;
  explanation_en: string;
  suggested_fix_he: string;
  suggested_fix_en: string;
}

export interface FieldAuditReport {
  all_clear: boolean;
  issues: FieldAuditIssue[];
}

export interface MockRow {
  entity_name: string;
  values: Record<string, string>;
}

export interface VerifiedCell {
  field_id: string;
  label_he: string;
  value: string | null;
  confidence: Confidence;
  corroboration_count: number;
  primary_source?: { url?: string; quote?: string; domain?: string } | null;
  all_sources: unknown[];
  flags: string[];
}

export interface EntityResult {
  entity_name: string;
  cells: Record<string, VerifiedCell>;
  row_flags: string[];
}

export type SearchEngine = "wikipedia" | "serpapi" | "duckduckgo" | "mock";
