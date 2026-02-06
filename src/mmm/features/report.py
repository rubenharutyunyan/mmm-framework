from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FeatureStepReport:
    """Traceability for a single transformer step."""

    transformer: str
    params: dict[str, Any]
    added_features: list[str]
    notes: str | None = None


@dataclass
class FeatureReport:
    """Aggregate traceability for a pipeline of transformers (ordered)."""

    steps: list[FeatureStepReport] = field(default_factory=list)

    def add_step(self, step: FeatureStepReport) -> None:
        self.steps.append(step)

    @property
    def added_features(self) -> list[str]:
        out: list[str] = []
        for step in self.steps:
            out.extend(step.added_features)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [
                {
                    "transformer": s.transformer,
                    "params": s.params,
                    "added_features": s.added_features,
                    "notes": s.notes,
                }
                for s in self.steps
            ]
        }
