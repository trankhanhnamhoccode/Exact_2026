from dataclasses import dataclass, field
from typing import Any, Optional

from xai_physics.core.answer import AnswerEnvelope


@dataclass
class TraceStep:
    title: str
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolveResult:
    status: str
    domain: str
    answer: Optional[Any] = None
    answer_meta: Optional[AnswerEnvelope] = None
    trace: list[TraceStep] = field(default_factory=list)
    error: Optional[str] = None

    def add_step(self, title: str, detail: str, **data: Any) -> None:
        self.trace.append(TraceStep(title=title, detail=detail, data=data))

    def set_answer(self, answer: Any, meta: AnswerEnvelope | None = None) -> None:
        self.answer = answer
        self.answer_meta = meta
