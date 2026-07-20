from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalFilter:
    user_id: str | None = None
    reference_doc: str | None = None
    document_ids: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.document_ids is not None:
            normalized = frozenset(self.document_ids) or None
            object.__setattr__(self, "document_ids", normalized)
