from dataclasses import dataclass, field
from typing import Any, Optional


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
    trace: list[TraceStep] = field(default_factory=list)
    error: Optional[str] = None

    def add_step(self, title: str, detail: str, **data: Any) -> None:
        self.trace.append(TraceStep(title=title, detail=detail, data=data))
