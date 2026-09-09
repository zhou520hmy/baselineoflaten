"""Versioned Judger prompts for reasoning before constrained vector readout."""

from __future__ import annotations

from typing import Mapping

from .schema import (
    BenchmarkCase,
    CaseVariant,
    PolicyProgram,
    canonical_sha256,
)
from .prompts import (
    canonical_fact_assertion,
    information_questions,
)


PROMPT_PROTOCOL_VERSION = "deliberative_joint_information_vector_v2_3_dev_20260903"
FULL_PROMPT_PROTOCOL_VERSION = (
    "deliberative_joint_information_vector_v2_3_full_20260903"
)

DELIBERATIVE_JOINT_VECTOR_SYSTEM = (
    "You are a joint private-information recovery probe. Recover every ordered "
    "binary status from the received communication state. First reason carefully "
    "inside the provided <think> block: inspect each question in order, preserve "
    "uncertainty instead of inventing evidence, and do not apply the action policy. "
    "Keep the entire reasoning concise and below 384 tokens. "
    "After </think>, return only the ordered comma-separated binary vector."
)


def prompt_content_sha256() -> str:
    """Hash the executable prompt contract independently of split governance."""
    return canonical_sha256(
        {
            "system": DELIBERATIVE_JOINT_VECTOR_SYSTEM,
            "role_policy_visibility": "judger_only",
            "semantic_readout": "receiver_reasoning_then_constrained_x_bit_vector",
            "thinking_open": "<think>\n",
            "thinking_close": "</think>",
            "answer_prefix": "\n\n",
        }
    )


def prompt_protocol_sha256() -> str:
    return canonical_sha256(
        {
            "version": PROMPT_PROTOCOL_VERSION,
            "system": DELIBERATIVE_JOINT_VECTOR_SYSTEM,
            "role_policy_visibility": "judger_only",
            "semantic_readout": "receiver_reasoning_then_constrained_x_bit_vector",
            "thinking_open": "<think>\n",
            "thinking_close": "</think>",
            "answer_prefix": "\n\n",
        }
    )


def full_prompt_protocol_sha256() -> str:
    return canonical_sha256(
        {
            "version": FULL_PROMPT_PROTOCOL_VERSION,
            "content_sha256": prompt_content_sha256(),
            "scope": "development_and_previously_exposed_test",
        }
    )


def build_deliberative_joint_vector_prompt(
    case: BenchmarkCase,
    program: PolicyProgram,
    *,
    visible_context: str = "",
) -> tuple[dict[str, str], ...]:
    if case.policy_program_id != program.policy_program_id:
        raise ValueError("case and policy program differ")
    questions = "\n".join(information_questions(program))
    content = (
        f"Domain: {case.domain}\n"
        f"Public scenario:\n{case.public_scenario}\n\n"
        f"Ordered private-information questions:\n{questions}\n\n"
        "Reason through all questions in Q1-to-Q"
        f"{len(program.facts)} order. After closing </think>, return exactly "
        f"{len(program.facts)} binary digits in that order, separated by single "
        "commas, with no text after the vector."
    )
    if visible_context.strip():
        content += "\n\nReceived communication:\n" + visible_context.strip()
    return (
        {"role": "system", "content": DELIBERATIVE_JOINT_VECTOR_SYSTEM},
        {"role": "user", "content": content},
    )


def scan_deliberative_prompt_leakage(
    case: BenchmarkCase,
    variant: CaseVariant,
    program: PolicyProgram,
) -> tuple[str, ...]:
    text = "\n".join(
        message["content"]
        for message in build_deliberative_joint_vector_prompt(case, program)
    )
    errors: list[str] = []
    for fact in variant.facts:
        if canonical_fact_assertion(fact) in text:
            errors.append(f"{variant.variant_id}:{fact.fact_id}: realized fact leaked")
    for forbidden in ("Gold", "Expected action", "REQUIRED", "NOT_REQUIRED"):
        if forbidden in text:
            errors.append(f"{variant.variant_id}: forbidden token {forbidden!r}")
    return tuple(errors)


def messages_sha256(messages: tuple[Mapping[str, str], ...]) -> str:
    return canonical_sha256(tuple(dict(message) for message in messages))
