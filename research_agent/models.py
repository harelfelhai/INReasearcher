from __future__ import annotations
from typing import Optional, Literal, Dict, List
from pydantic import BaseModel, Field


class ColumnPlan(BaseModel):
    id: str
    label_he: str
    label_en: str
    type: Literal["person_name", "url", "date", "free_text", "organization", "number"]
    temporal_anchor: Optional[str] = None          # e.g. "1990"
    search_queries_he: List[str]                   # Hebrew queries with {entity} placeholder
    search_queries_en: List[str]                   # English queries with {entity} placeholder
    preferred_source_domains: List[str] = Field(default_factory=list)
    min_corroborations: int = 1                    # independent domains required
    depends_on: Optional[str] = None               # id of field whose value must resolve first


class ResearchPlan(BaseModel):
    entity_type: str
    research_question_original: str
    columns: List[ColumnPlan]


class ExtractionResult(BaseModel):
    field_id: str
    value: Optional[str]                # None means "not found" — never a guess
    quote_original: Optional[str]       # exact copy-paste from source text
    source_url: str
    source_domain: str
    is_grounded: bool                   # passed the substring grounding check
    extractor_confidence: float         # 0.0–1.0 self-reported by LLM


class VerifiedCell(BaseModel):
    field_id: str
    label_he: str
    value: Optional[str]
    confidence: Literal["HIGH", "MEDIUM", "LOW", "NOT_FOUND"]
    corroboration_count: int
    primary_source: Optional[dict] = None
    all_sources: List[dict] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)


class EntityResult(BaseModel):
    entity_name: str
    cells: Dict[str, VerifiedCell]
    row_flags: List[str] = Field(default_factory=list)


# ── Guided Prompting ─────────────────────────────────────────────────────────

class ClarificationQuestion(BaseModel):
    field: str          # which aspect is ambiguous (e.g. "timeframe", "entity_type")
    question_he: str    # question to show user in Hebrew
    question_en: str
    example: str        # concrete example answer to guide the user


class ClarificationRequest(BaseModel):
    """Returned by the compiler when the prompt is not yet executable."""
    is_executable: Literal[False] = False
    reason: str                             # short explanation of why it's not executable
    questions: List[ClarificationQuestion]  # what we need the user to clarify
    prompt_template: str                    # a filled-in template they can copy-edit


class ExecutableResearchPlan(BaseModel):
    """Wrapper that signals the prompt passed the preflight check."""
    is_executable: Literal[True] = True
    plan: ResearchPlan
