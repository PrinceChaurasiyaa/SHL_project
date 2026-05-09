"""
app/agent.py

Core conversation agent.
Orchestrates: guard check → constraint extraction → retrieval → LLM → response parsing.
All state is derived from the message history passed in per call (stateless design).
"""
from __future__ import annotations
import json
import logging
import os
import re
from typing import Optional

from groq import Groq

from app.catalog import Catalog, CatalogEntry
from app.guards import (
    GuardResult,
    check as guard_check,
    is_comparison_query,
    is_refinement,
    extract_mentioned_names,
)
from app.models import ChatRequest, ChatResponse, Recommendation, Message
from app.prompts import (
    SYSTEM_PROMPT,
    CONSTRAINT_EXTRACTION_SYSTEM,
    CONSTRAINT_EXTRACTION_USER,
    COMPARISON_ADDENDUM,
    build_catalog_context_block,
    format_conversation_for_extraction,
    inject_turn_warning,
)
from app.retrieval import RetrievalEngine

logger = logging.getLogger(__name__)

MAX_TURNS = 8
MAX_RECOMMENDATIONS = 10


class Agent:
    """
    Stateless conversation agent.
    Instantiate once at startup and call .process() per request.
    """

    def __init__(
        self,
        catalog: Optional[Catalog] = None,
        retrieval: Optional[RetrievalEngine] = None,
        client: Optional[Groq] = None,
    ) -> None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        self._catalog = catalog or Catalog.load()
        self._retrieval = retrieval or RetrievalEngine(self._catalog)
        self._client = client or Groq(
            api_key=os.environ.get("GROQ_API_KEY", ""),
            timeout=121.0,
            max_retries=3,
        )
        logger.info("Groq key loaded: %s...", api_key[:8]) 
        self._catalog_names = [e.name for e in self._catalog.entries]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, request: ChatRequest) -> ChatResponse:
        messages = [m.model_dump() for m in request.messages]
        user_message = messages[-1]["content"]
        turn_number = self._count_user_turns(messages)
        previous_turns = len(messages) - 1  # excludes current user turn

        # ── 1. Guard check ─────────────────────────────────────────────
        guard: GuardResult = guard_check(
            user_message, turn_number, previous_turns
        )
        if guard.blocked:
            return ChatResponse(
                reply=guard.reply,
                recommendations=[],
                end_of_conversation=False,
            )

        # ── 2. Detect conversation mode ────────────────────────────────
        comparison = is_comparison_query(user_message)
        refinement = is_refinement(user_message) and previous_turns > 0

        # ── 3. Lightweight rule-based constraints ──────────────────────
        constraints = {}

        logger.debug("Using lightweight rule-based constraints")

        # ── 4. Retrieve relevant catalog entries ───────────────────────
        mentioned_names = extract_mentioned_names(user_message, self._catalog_names)

        if comparison and constraints.get("comparison_names"):
            # For comparisons, fetch the specific named assessments
            compare_entries = self._retrieval.search_for_comparison(
                constraints["comparison_names"] or mentioned_names
            )
            catalog_context = self._catalog.build_context_for_llm(compare_entries)
        else:
            search_query = constraints.get("search_query") or user_message
            catalog_context = self._retrieval.get_context_for_query(
                query=search_query,
                constraints={
                    "job_level": constraints.get("job_level_filter"),
                    "language": constraints.get("language"),
                    "test_type_codes": constraints.get("test_types_wanted") or None,
                    "remote_only": False,
                    "adaptive_only": False,
                    "must_include_names": mentioned_names or None,
                },
                max_entries=20,
            )

        # ── 5. Build system prompt with catalog context ─────────────────
        catalog_block = build_catalog_context_block(catalog_context)
        system = SYSTEM_PROMPT + catalog_block

        if comparison:
            system += COMPARISON_ADDENDUM

        # Inject turn budget warning if needed
        total_turns = len(messages)
        system = inject_turn_warning(system, total_turns)

        # ── 6. Call LLM ────────────────────────────────────────────────
        llm_reply = self._call_llm(system, messages)

        # ── 7. Parse recommendations from reply ────────────────────────
        raw_recs, clean_reply = self._parse_recommendations(llm_reply)

        # ── 8. Validate all URLs against catalog ───────────────────────
        validated_recs = self._catalog.validate_and_filter_recommendations(raw_recs)
        validated_recs = validated_recs[:MAX_RECOMMENDATIONS]

        # ── 9. Determine end_of_conversation ──────────────────────────
        eoc = self._detect_end_of_conversation(
            user_message, clean_reply, validated_recs, total_turns
        )

        return ChatResponse(
            reply=clean_reply,
            recommendations=[
                Recommendation(**r) for r in validated_recs
            ],
            end_of_conversation=eoc,
        )

    # ------------------------------------------------------------------
    # Constraint extraction (lightweight LLM call)
    # ------------------------------------------------------------------

    def _extract_constraints(self, messages: list[dict]) -> dict:
        """
        Use a fast LLM call to extract structured constraints from the conversation.
        Falls back to empty dict on any error.
        """
        convo_text = format_conversation_for_extraction(messages)
        user_prompt = CONSTRAINT_EXTRACTION_USER.format(
            conversation_text=convo_text
        )
        try:
            response = self._client.chat.completions.create(
                model="llama-3.1-8b-instant",
                temperature=0,
                max_tokens=512,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": CONSTRAINT_EXTRACTION_SYSTEM
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
            )

            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Constraint extraction failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # LLM call (main reasoning)
    # ------------------------------------------------------------------

    def _call_llm(self, system: str, messages: list[dict]) -> str:
        """Call Claude with the full conversation history."""
        # Filter to only user/assistant roles, strip any system-role messages
        clean_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] in ("user", "assistant")
        ]
        response = self._client.chat.completions.create(
            model=os.environ.get(
                "GROQ_MODEL",
                "llama-3.1-8b-instant"
            ),
            temperature=0.1,
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": system
                },
                *clean_messages
            ],
        )

        content = response.choices[0].message.content

        if isinstance(content, list):

            parts = []

            for block in content:

                if isinstance(block, dict):
                    parts.append(block.get("text", ""))

                elif hasattr(block, "text"):
                    parts.append(block.text)

                else:
                    parts.append(str(block))

            content = "\n".join(parts)

        return str(content)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_recommendations(
        self, llm_reply: str
    ) -> tuple[list[dict], str]:
        """
        Extract the <RECOMMENDATIONS>[...]</RECOMMENDATIONS> block from the reply.
        Returns (raw_rec_dicts, cleaned_reply_text).
        """
        pattern = re.compile(
            r"<RECOMMENDATIONS>\s*(.*?)\s*</RECOMMENDATIONS>",
            re.DOTALL | re.IGNORECASE,
        )
        match = pattern.search(llm_reply)
        if not match:
            return [], llm_reply.strip()

        json_str = match.group(1).strip()
        # Remove the recommendations block from the display reply
        clean_reply = pattern.sub("", llm_reply).strip()
        # Clean up any double newlines left behind
        clean_reply = re.sub(r"\n{3,}", "\n\n", clean_reply)

        try:
            raw = json.loads(json_str)
            if not isinstance(raw, list):
                return [], clean_reply
            # Normalise keys: accept 'name', 'url', 'test_type'
            result: list[dict] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                name = item.get("name", "").strip()
                url = item.get("url", "").strip()
                test_type = item.get("test_type", "").strip()
                if name and url:
                    result.append({"name": name, "url": url, "test_type": test_type})
            return result, clean_reply
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse recommendations JSON: %s\n%s", exc, json_str)
            return [], clean_reply

    # ------------------------------------------------------------------
    # End-of-conversation detection
    # ------------------------------------------------------------------

    def _detect_end_of_conversation(
        self,
        user_message: str,
        reply: str,
        recommendations: list[dict],
        total_turns: int,
    ) -> bool:
        """
        Heuristic: conversation is over when:
        - User explicitly says they're done, OR
        - We have recommendations AND user has acknowledged the list, OR
        - Turn cap reached
        """
        # Hard cap
        if total_turns >= MAX_TURNS:
            return True

        user_done_patterns = [
            r"\b(thank\s+you|thanks|that.s\s+(all|it|perfect|great)|done|perfect|confirmed)\b",
            r"\b(that\s+works|looks\s+good|sounds\s+good|great|brilliant|excellent)\b",
            r"\b(no\s+(more\s+)?questions?|nothing\s+else|i.m\s+(good|all\s+set))\b",
        ]
        for pat in user_done_patterns:
            if re.search(pat, user_message, re.IGNORECASE):
                if recommendations:  # Only close if we have delivered something
                    return True

        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_user_turns(messages: list[dict]) -> int:
        return sum(1 for m in messages if m.get("role") == "user")
