from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Maps catalog "keys" strings to short test-type codes used in API responses.
# Based on SHL catalog key taxonomy.
_KEY_TO_CODE: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def keys_to_test_type(keys: list[str]) -> str:
    """Convert a list of SHL catalog key strings into a compact test-type code string."""
    codes = []
    seen: set[str] = set()
    for k in keys:
        code = _KEY_TO_CODE.get(k)
        if code and code not in seen:
            codes.append(code)
            seen.add(code)
    return ",".join(codes) if codes else "K"


@dataclass
class CatalogEntry:
    entity_id: str
    name: str
    url: str
    description: str
    keys: list[str]
    test_type: str
    job_levels: list[str]
    languages: list[str]
    duration: str
    remote: str
    adaptive: str

    # Pre-built text blob used for embedding / keyword search
    search_text: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        self.search_text = self._build_search_text()

    def _build_search_text(self) -> str:
        parts = [
            self.name,
            self.description,
            " ".join(self.keys),
            " ".join(self.job_levels),
            " ".join(self.languages),
            f"duration {self.duration}",
            f"remote {self.remote}",
            f"adaptive {self.adaptive}",
        ]
        return " ".join(p for p in parts if p).lower()

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "url": self.url,
            "description": self.description,
            "keys": self.keys,
            "test_type": self.test_type,
            "job_levels": self.job_levels,
            "languages": self.languages,
            "duration": self.duration,
            "remote": self.remote,
            "adaptive": self.adaptive,
        }


class Catalog:
    """Singleton catalog. Call Catalog.load() once at startup."""

    _instance: Optional["Catalog"] = None

    def __init__(self, entries: list[CatalogEntry]) -> None:
        self._entries: list[CatalogEntry] = entries
        self._by_id: dict[str, CatalogEntry] = {e.entity_id: e for e in entries}
        self._by_name: dict[str, CatalogEntry] = {
            e.name.lower(): e for e in entries
        }
        self._valid_urls: set[str] = {e.url for e in entries}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Catalog":
        if cls._instance is not None:
            return cls._instance
        if path is None:
            path = Path(__file__).parent.parent / "data" / "catalog.json"
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Catalog not found at {path}")
        with open(path, "r", encoding="utf-8") as fh:
            raw: list[dict] = json.load(fh)
        entries = [cls._parse_entry(r) for r in raw if r.get("status") == "ok"]
        cls._instance = cls(entries)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """For testing only."""
        cls._instance = None

    @staticmethod
    def _parse_entry(raw: dict) -> CatalogEntry:
        keys: list[str] = raw.get("keys") or []
        return CatalogEntry(
            entity_id=str(raw.get("entity_id", "")),
            name=raw.get("name", ""),
            url=raw.get("link", raw.get("url", "")),
            description=raw.get("description", ""),
            keys=keys,
            test_type=keys_to_test_type(keys),
            job_levels=raw.get("job_levels") or [],
            languages=raw.get("languages") or [],
            duration=raw.get("duration", ""),
            remote=raw.get("remote", ""),
            adaptive=raw.get("adaptive", ""),
        )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[CatalogEntry]:
        return self._entries

    @property
    def valid_urls(self) -> set[str]:
        return self._valid_urls

    def is_valid_url(self, url: str) -> bool:
        return url in self._valid_urls

    def get_by_id(self, entity_id: str) -> Optional[CatalogEntry]:
        return self._by_id.get(entity_id)

    def get_by_name(self, name: str) -> Optional[CatalogEntry]:
        return self._by_name.get(name.lower())

    def get_by_ids(self, ids: list[str]) -> list[CatalogEntry]:
        return [self._by_id[i] for i in ids if i in self._by_id]

    def keyword_filter(
        self,
        candidates: list[CatalogEntry],
        job_level: Optional[str] = None,
        language: Optional[str] = None,
        test_type_codes: Optional[list[str]] = None,
        remote_only: bool = False,
        adaptive_only: bool = False,
    ) -> list[CatalogEntry]:
        """Apply hard filters to a candidate list."""
        results = candidates

        if job_level:
            jl_lower = job_level.lower()
            results = [
                e for e in results
                if not e.job_levels  # no restriction = applies to all
                or any(jl_lower in j.lower() for j in e.job_levels)
            ]

        if language:
            lang_lower = language.lower()
            results = [
                e for e in results
                if not e.languages
                or any(lang_lower in l.lower() for l in e.languages)
            ]

        if test_type_codes:
            results = [
                e for e in results
                if any(code in e.test_type for code in test_type_codes)
            ]

        if remote_only:
            results = [e for e in results if e.remote.lower() == "yes"]

        if adaptive_only:
            results = [e for e in results if e.adaptive.lower() == "yes"]

        return results

    def build_context_for_llm(
        self, entries: list[CatalogEntry], max_entries: int = 20
    ) -> str:
        """Serialize a subset of entries into a compact string for LLM context."""
        lines: list[str] = []
        for e in entries[:max_entries]:
            lines.append(
                f"[{e.entity_id}] {e.name} | type={e.test_type} | "
                f"levels={', '.join(e.job_levels) if e.job_levels else 'all'} | "
                f"duration={e.duration} | adaptive={e.adaptive} | "
                f"url={e.url}\n"
                f"  desc: {e.description[:200].strip()}"
            )
        return "\n\n".join(lines)

    def build_full_catalog_summary(self, max_entries: int = 50) -> str:
        """Build a summary of the full catalog for initial context."""
        return self.build_context_for_llm(self._entries, max_entries=max_entries)

    def validate_and_filter_recommendations(
        self, recs: list[dict]
    ) -> list[dict]:
        """
        Given a list of raw recommendation dicts from LLM output,
        validate each URL against the catalog and drop any hallucinated entries.
        Also enriches test_type from catalog if missing.
        """
        valid: list[dict] = []
        for r in recs:
            url = r.get("url", "")
            name = r.get("name", "")
            # Try to find by URL or name
            entry = None
            for e in self._entries:
                if e.url == url or e.name.lower() == name.lower():
                    entry = e
                    break
            if entry is None:
                # Drop: not in catalog
                continue
            valid.append({
                "name": entry.name,
                "url": entry.url,
                "test_type": entry.test_type,
            })
        return valid
