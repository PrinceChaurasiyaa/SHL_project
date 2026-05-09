from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

from app.catalog import Catalog, CatalogEntry

# Lazy imports to avoid loading torch/sentence-transformers at import time
# when running tests without GPU
_encoder = None
_faiss_index = None
_index_ids: list[str] = []  # entity_id at each faiss position


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        _encoder = SentenceTransformer(model_name)
    return _encoder


def _get_index() -> tuple:
    """Returns (faiss_index, index_ids). Loads from disk or builds in memory."""
    global _faiss_index, _index_ids
    if _faiss_index is not None:
        return _faiss_index, _index_ids

    index_path = Path(__file__).parent.parent / "data" / "index.faiss"
    ids_path = Path(__file__).parent.parent / "data" / "index_ids.npy"

    if index_path.exists() and ids_path.exists():
        import faiss
        _faiss_index = faiss.read_index(str(index_path))
        _index_ids = np.load(str(ids_path), allow_pickle=True).tolist()
    else:
        # Build in memory from catalog (fallback for cold starts)
        _faiss_index, _index_ids = _build_index_in_memory()

    return _faiss_index, _index_ids


def _build_index_in_memory():
    import faiss
    catalog = Catalog.load()
    entries = catalog.entries
    encoder = _get_encoder()
    texts = [e.search_text for e in entries]
    ids = [e.entity_id for e in entries]
    embeddings = encoder.encode(
        texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    )
    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalized vecs
    index.add(embeddings)
    return index, ids


class RetrievalEngine:
    """
    Two-stage retrieval:
    1. FAISS semantic search over catalog search_text embeddings
    2. Keyword/BM25-style re-rank + hard constraint filtering
    """

    def __init__(self, catalog: Optional[Catalog] = None) -> None:
        self._catalog = catalog or Catalog.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 20,
        job_level: Optional[str] = None,
        language: Optional[str] = None,
        test_type_codes: Optional[list[str]] = None,
        remote_only: bool = False,
        adaptive_only: bool = False,
        must_include_names: Optional[list[str]] = None,
    ) -> list[CatalogEntry]:
        """
        Return up to top_k CatalogEntry objects relevant to query,
        applying optional hard filters.
        """
        # Stage 1: semantic candidates
        semantic_hits = self._semantic_search(query, top_k=min(top_k * 3, 60))

        # Stage 2: keyword boost
        keyword_hits = self._keyword_search(query, top_k=top_k * 2)

        # Merge: semantic first, then keyword additions
        seen: set[str] = set()
        merged: list[CatalogEntry] = []
        for e in semantic_hits + keyword_hits:
            if e.entity_id not in seen:
                merged.append(e)
                seen.add(e.entity_id)

        # Stage 3: hard filters
        filtered = self._catalog.keyword_filter(
            merged,
            job_level=job_level,
            language=language,
            test_type_codes=test_type_codes,
            remote_only=remote_only,
            adaptive_only=adaptive_only,
        )

        # Ensure must-include names are present
        if must_include_names:
            present_names = {e.name.lower() for e in filtered}
            for name in must_include_names:
                if name.lower() not in present_names:
                    entry = self._catalog.get_by_name(name)
                    if entry:
                        filtered.insert(0, entry)

        return filtered[:top_k]

    def search_for_comparison(
        self, names: list[str]
    ) -> list[CatalogEntry]:
        """Retrieve specific assessments by name for comparison queries."""
        results: list[CatalogEntry] = []
        for name in names:
            entry = self._catalog.get_by_name(name)
            if entry:
                results.append(entry)
            else:
                # Fuzzy fallback: token overlap
                best = self._fuzzy_name_match(name)
                if best:
                    results.append(best)
        return results

    def get_context_for_query(
        self,
        query: str,
        constraints: dict,
        max_entries: int = 15,
    ) -> str:
        """
        High-level: search + format catalog context string for LLM prompt.
        constraints keys: job_level, language, test_type_codes,
                          remote_only, adaptive_only, must_include_names
        """
        hits = self.search(
            query=query,
            top_k=max_entries,
            job_level=constraints.get("job_level"),
            language=constraints.get("language"),
            test_type_codes=constraints.get("test_type_codes"),
            remote_only=constraints.get("remote_only", False),
            adaptive_only=constraints.get("adaptive_only", False),
            must_include_names=constraints.get("must_include_names"),
        )
        return self._catalog.build_context_for_llm(hits, max_entries=max_entries)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _semantic_search(self, query: str, top_k: int) -> list[CatalogEntry]:
        try:
            index, ids = _get_index()
            encoder = _get_encoder()
            qvec = encoder.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )
            qvec = np.array(qvec, dtype="float32")
            scores, positions = index.search(qvec, top_k)
            results: list[CatalogEntry] = []
            for pos in positions[0]:
                if pos < 0 or pos >= len(ids):
                    continue
                entry = self._catalog.get_by_id(ids[pos])
                if entry:
                    results.append(entry)
            return results
        except Exception:
            # If FAISS fails (e.g. no model loaded in test), fall through
            return []

    def _keyword_search(self, query: str, top_k: int) -> list[CatalogEntry]:
        """Simple TF-style keyword match against pre-built search_text."""
        tokens = set(re.sub(r"[^a-z0-9 ]", " ", query.lower()).split())
        if not tokens:
            return []

        scored: list[tuple[float, CatalogEntry]] = []
        for entry in self._catalog.entries:
            score = sum(
                1.0 + entry.search_text.count(tok) * 0.1
                for tok in tokens
                if tok in entry.search_text
            )
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def _fuzzy_name_match(self, name: str) -> Optional[CatalogEntry]:
        """Return the catalog entry whose name has the most token overlap with name."""
        name_tokens = set(name.lower().split())
        best_score = 0
        best_entry: Optional[CatalogEntry] = None
        for entry in self._catalog.entries:
            entry_tokens = set(entry.name.lower().split())
            overlap = len(name_tokens & entry_tokens)
            if overlap > best_score:
                best_score = overlap
                best_entry = entry
        return best_entry if best_score > 0 else None
    
    def warmup(self):
        """Preload encoder and FAISS index during app startup."""
        _get_encoder()
        _get_index()
