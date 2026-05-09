"""
app/guards.py

Pre-LLM guard layer. Fast heuristic checks that run before spending tokens.
Returns a GuardResult indicating whether to block and with what message.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class GuardResult:
    blocked: bool
    reason: str = ""
    reply: str = ""


# ---------------------------------------------------------------------------
# Injection patterns
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"ignore\s+(previous|all|prior|above|earlier)\s+instructions?",
        r"disregard\s+(previous|all|prior|above|earlier)",
        r"forget\s+(everything|all|prior|previous|your instructions?)",
        r"you\s+are\s+now\s+a?\s*(general|different|new|another)",
        r"pretend\s+(you\s+(have\s+no|are\s+not|aren.t|don.t\s+have)|to\s+be)",
        r"act\s+as\s+(if\s+you\s+(are|were)|a\s+(general|different|unrestricted))",
        r"(jailbreak|DAN|do\s+anything\s+now)",
        r"(override|bypass|circumvent|disable)\s+(your\s+)?(instructions?|restrictions?|rules?|guidelines?|safety|filters?)",
        r"(reveal|show|print|output|display)\s+(your\s+)?(system\s+prompt|instructions?|prompt|context)",
        r"(print|output|show|display|repeat|write)\s+(all\s+)?(instructions?|the\s+instructions?)\s+(you\s+were\s+given|above)",
        r"(in\s+(this|a)\s+hypothetical|hypothetically|imagine\s+you\s+(are|were|have\s+no))",
        r"(you\s+have\s+no\s+restrictions?|without\s+restrictions?|unrestricted\s+mode)",
        r"(translate|repeat|output|print|write)\s+(the\s+above|everything\s+above|all\s+of\s+the\s+above)",
    ]
]

# ---------------------------------------------------------------------------
# Off-topic patterns (things clearly not about SHL assessment selection)
# ---------------------------------------------------------------------------

_OFF_TOPIC_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(salary|compensation|pay|wage|bonus|equity|stock)\b",
        r"\b(visa|immigration|work\s+permit|right\s+to\s+work)\b",
        r"\b(lawsuit|litigation|legal\s+action|sue|discrimination\s+claim)\b",
        r"\b(write|can\s+you\s+write|create|generate|make)\s+(me\s+)?(a\s+)?(python|java|javascript|code|program|script|app)\b",
        r"\b(recipe|cook|food|weather|sports?|movie|music|song)\b",
        r"\b(stock\s+price|cryptocurrency|bitcoin|invest(ment|ing)?)\b",
        r"\b(competitor|rival\s+(product|company)|hogan|criteria\s+corp|psytech|talentplus)\b",
        r"\b(write|draft|create|generate)\s+(me\s+)?(a\s+)?(cover\s+letter|cv|resume|essay|poem|story|blog)\b",
        r"\b(how\s+do\s+I\s+(cook|make|bake|drive|fly|swim))\b",
    ]
]

# ---------------------------------------------------------------------------
# Sensitive legal/compliance topics that we deflect
# ---------------------------------------------------------------------------

_LEGAL_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(GDPR|EEOC|ADA|disparate\s+impact|adverse\s+impact|protected\s+class)\b",
        r"\b(legal(ly)?\s+(required|mandated|compliant|obligation))\b",
        r"\b(can\s+I\s+(legally|lawfully)\s+(use|ask|require))\b",
        r"\b(is\s+it\s+legal\s+to)\b",
    ]
]


# ---------------------------------------------------------------------------
# Vagueness heuristics for first-turn detection
# ---------------------------------------------------------------------------

_VAGUE_FIRST_TURN_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^i\s+(need|want|am\s+looking\s+for)\s+an?\s+assessment[.?]?\s*$",
        r"^(help|assist)\s+(me)?\s*(with)?\s*(assessments?|hiring|selection)[.?]?\s*$",
        r"^what\s+assessments?\s+(do\s+you\s+have|are\s+available|can\s+you\s+offer)[?.]?\s*$",
        r"^(show|give|tell)\s+me\s+(your\s+)?(assessments?|catalog|products?)[.?]?\s*$",
        r"^(hi|hello|hey)\s*[,.]?\s*can\s+you\s+help(\s+me)?[?.]?\s*$",
        r"^(hi|hello|hey)\s*[,.]?\s*i\s+need\s+help[.?]?\s*$",
        r"^(hi|hello|hey)\s*[,.]?\s*$",
    ]
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check(
    user_message: str,
    turn_number: int,
    previous_turns: int,
) -> GuardResult:
    """
    Run all guard checks on the latest user message.

    Args:
        user_message: The raw text of the latest user turn.
        turn_number: Which turn number this is (1-indexed).
        previous_turns: Number of turns already in history (user+assistant).

    Returns:
        GuardResult(blocked=True, ...) if we should short-circuit the LLM.
        GuardResult(blocked=False) if the message is safe to process.
    """
    stripped = user_message.strip()

    # 1. Injection detection (always block, never soft-refuse)
    for pat in _INJECTION_PATTERNS:
        if pat.search(stripped):
            return GuardResult(
                blocked=True,
                reason="prompt_injection",
                reply=(
                    "I'm here to help you select SHL assessments for your roles. "
                    "I can't process that kind of request — is there a specific role "
                    "you're hiring for that I can help you with?"
                ),
            )

    # 2. Legal/compliance deflection
    for pat in _LEGAL_PATTERNS:
        if pat.search(stripped):
            return GuardResult(
                blocked=True,
                reason="legal_question",
                reply=(
                    "I'm not able to give legal or compliance advice. "
                    "For questions about assessment legality, please consult SHL's "
                    "legal documentation or your HR legal team. "
                    "Would you like help selecting the right assessment for a role instead?"
                ),
            )

    # 3. Off-topic detection
    for pat in _OFF_TOPIC_PATTERNS:
        if pat.search(stripped):
            return GuardResult(
                blocked=True,
                reason="off_topic",
                reply=(
                    "I can only help with selecting SHL assessments for hiring and "
                    "development. Could you describe the role or competency you're "
                    "trying to assess?"
                ),
            )

    # 4. Vagueness on first user turn
    if previous_turns == 0:
        for pat in _VAGUE_FIRST_TURN_PATTERNS:
            if pat.match(stripped):
                return GuardResult(
                    blocked=True,
                    reason="vague_first_turn",
                    reply=(
                        "I'd be happy to help you find the right SHL assessment! "
                        "To get started, could you tell me a bit about the role "
                        "you're hiring for?"
                    ),
                )

    return GuardResult(blocked=False)


def is_comparison_query(user_message: str) -> bool:
    """Heuristic: does the message ask to compare assessments?"""
    patterns = [
        r"\b(compare|comparison|difference|vs\.?|versus|which\s+is\s+better|how\s+do\s+.+differ)\b",
        r"\bwhat.s\s+the\s+difference\s+between\b",
        r"\b(OPQ|GSA|Verify|MQ|DSI|Automata)\b.{0,30}\b(OPQ|GSA|Verify|MQ|DSI|Automata)\b",
    ]
    for p in patterns:
        if re.search(p, user_message, re.IGNORECASE):
            return True
    return False


def is_refinement(user_message: str) -> bool:
    """Heuristic: is the user updating/refining a previous recommendation?"""
    patterns = [
        r"\b(actually|instead|also\s+add|add\s+.+test|remove|drop|exclude|include|change\s+to)\b",
        r"\b(more\s+(junior|senior|entry|experienced))\b",
        r"\b(update|revise|modify|adjust)\s+(the\s+)?(list|shortlist|recommendation)\b",
        r"\b(what\s+about|can\s+you\s+also)\b",
    ]
    for p in patterns:
        if re.search(p, user_message, re.IGNORECASE):
            return True
    return False


def extract_mentioned_names(user_message: str, catalog_names: list[str]) -> list[str]:
    """
    Find any catalog assessment names mentioned in the user message.
    Case-insensitive substring match.
    """
    msg_lower = user_message.lower()
    found: list[str] = []
    for name in catalog_names:
        if len(name) >= 4 and name.lower() in msg_lower:
            found.append(name)
    return found
