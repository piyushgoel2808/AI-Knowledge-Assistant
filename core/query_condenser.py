"""Context-aware conversational query condensation engine (TICK-006).

Resolves pronouns, ellipsis, and conversational context into fully qualified
standalone retrieval queries.
"""

from __future__ import annotations

import re
from typing import Sequence

from core.ai_service import BaseLLMProvider


CONDENSE_SYSTEM_PROMPT = (
    "You are a query condensation assistant. "
    "Your job is to rephrase the user's latest follow-up question into a single, standalone search query "
    "that incorporates all necessary context, entities, and keywords from the prior conversation. "
    "Output ONLY the standalone search query without any explanation, markdown, prefixes, or quotes."
)

CONDENSE_FEW_SHOT_TEMPLATE = """Given a conversation history and a follow-up user question, rephrase the follow-up question into a standalone search query that contains all necessary context, entities, and keywords from the previous conversation.
Do NOT answer the question. Only output the rephrased standalone query. If the question is already standalone, output it as-is.

Example 1:
Chat History:
user: What is the parental leave policy for software engineers?
assistant: Full-time software engineers receive 16 weeks of paid parental leave.
Follow-up question: Does it also apply to contractors?
Standalone query: Does parental leave apply to independent contractors for Software Engineers?

Example 2:
Chat History:
user: Tell me about travel reimbursement.
assistant: Meals and flights under $500 are reimbursed within 5 business days.
Follow-up question: What about mileage?
Standalone query: What is the travel reimbursement policy for mileage?

Example 3:
Chat History:
user: What are the working hours?
assistant: Core hours are 10 AM to 4 PM.
Follow-up question: How many sick days do I get?
Standalone query: How many sick days are employees entitled to?

Chat History:
{history_block}

Follow-up question: {question}
Standalone query:"""


def condense_query(
    question: str,
    chat_history: Sequence[object] | None = None,
    provider: BaseLLMProvider | None = None,
) -> str:
    """Rephrase follow-up questions referencing previous turns into standalone queries."""
    clean_question = question.strip()
    if not clean_question:
        return clean_question

    if not chat_history:
        return clean_question

    # Filter out empty turns
    valid_history = [
        msg for msg in chat_history
        if getattr(msg, "content", None) and getattr(msg, "content", "").strip()
    ]
    if not valid_history:
        return clean_question

    if provider is None:
        return clean_question

    # Build history block
    history_lines: list[str] = []
    for msg in valid_history[-6:]:
        role = getattr(msg, "role", "user")
        content = getattr(msg, "content", "").strip()
        history_lines.append(f"{role}: {content}")
    history_block = "\n".join(history_lines)

    prompt = CONDENSE_FEW_SHOT_TEMPLATE.format(
        history_block=history_block,
        question=clean_question,
    )

    try:
        response = provider.generate(
            prompt=prompt,
            system_prompt=CONDENSE_SYSTEM_PROMPT,
        )
        condensed = response.text.strip()
        # Clean any prefixes or quotes
        condensed = re.sub(
            r"^(Standalone query:\s*|Query:\s*|Rephrased query:\s*)",
            "",
            condensed,
            flags=re.IGNORECASE,
        ).strip()
        condensed = condensed.strip("\"'`")
        return condensed if condensed else clean_question
    except Exception:
        return clean_question
