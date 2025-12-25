from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

@dataclass
class CodeSpan:
    file_path: str
    start_line: int
    end_line: int

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CodeSpan":
        return CodeSpan(
            file_path=str(d["file_path"]),
            start_line=int(d["start_line"]),
            end_line=int(d["end_line"]),
        )

@dataclass
class Evidence:
    span: CodeSpan
    snippet: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Evidence":
        span = CodeSpan.from_dict(d["span"])
        return Evidence(span=span, snippet=str(d.get("snippet","")))

@dataclass
class TraceStep:
    step: int
    kind: str  # "extract" | "reason" | "answer"
    content: str
    evidence_refs: List[int] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TraceStep":
        return TraceStep(
            step=int(d["step"]),
            kind=str(d["kind"]),
            content=str(d.get("content","")),
            evidence_refs=[int(x) for x in d.get("evidence_refs", [])],
        )

@dataclass
class QAItem:
    id: str
    scenario: str               # "scenario1" | "scenario2"
    rule_id: str
    title: str
    question: str
    answer: str
    evidence: List[Evidence] = field(default_factory=list)
    trace: List[TraceStep] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "QAItem":
        ev = [Evidence.from_dict(x) for x in d.get("evidence", [])]
        tr = [TraceStep.from_dict(x) for x in d.get("trace", [])]
        return QAItem(
            id=str(d["id"]),
            scenario=str(d.get("scenario","scenario1")),
            rule_id=str(d.get("rule_id","")),
            title=str(d.get("title","")),
            question=str(d.get("question","")),
            answer=str(d.get("answer","")),
            evidence=ev,
            trace=tr,
            meta=dict(d.get("meta", {})),
        )
