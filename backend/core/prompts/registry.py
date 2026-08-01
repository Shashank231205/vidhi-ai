"""Prompt definitions and context rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from core.retrieval import Hit


@dataclass(frozen=True, slots=True)
class Prompt[T: BaseModel]:
    """A versioned prompt bound to the schema its output must satisfy.

    `version` is part of the identity so an eval run can record which revision
    produced a number, and so two revisions can be compared on the same set.
    """

    name: str
    version: str
    system: str
    template: str
    schema: type[T]

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"

    def messages(self, **values: Any) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.template.format(**values)},
        ]


def render_context(hits: list[Hit]) -> str:
    """Format retrieved chunks for a prompt.

    Each chunk is labelled with its id and citation. The id is what the model
    must echo back in `chunk_id`, which is what lets the verifier check a claim
    against the exact text the model was shown — not merely against the corpus.
    """
    if not hits:
        return "(no relevant provisions were retrieved)"

    blocks = []
    for hit in hits:
        blocks.append(
            f"[chunk_id: {hit.chunk_id}]\nSource: {hit.citation}\nText: {hit.content.strip()}"
        )
    return "\n\n---\n\n".join(blocks)
