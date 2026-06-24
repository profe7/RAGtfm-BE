from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalFilter:
    """Scopes a retrieval query to a subset of stored chunks.

    The three fields always travel together through the retrieval chain, so we
    group them into one object instead of threading three parameters everywhere.

    `frozen=True` makes instances immutable *and* hashable, which lets a filter
    serve directly as an `lru_cache` key in the BM25 retriever. `document_ids` is
    normalized to a `frozenset` so two filters that list the same ids in a
    different order compare equal (and hit the same cache entry).
    """

    user_id: str | None = None
    reference_doc: str | None = None
    document_ids: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.document_ids is not None:
            # `object.__setattr__` is how you assign on a frozen dataclass: normal
            # assignment is blocked. An empty collection collapses to None so it
            # behaves like "no filter".
            normalized = frozenset(self.document_ids) or None
            object.__setattr__(self, "document_ids", normalized)
