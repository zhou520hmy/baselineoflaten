"""Prompt construction for joint information recovery and action selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .schema import (
    BenchmarkCase,
    CaseVariant,
    FactSpec,
    FactValue,
    PolicyProgram,
    RoleSpec,
    canonical_sha256,
)


PROMPT_PROTOCOL_VERSION = "joint_information_vector_v2_2_20260801"
ACTION_LABELS = tuple("ABCDEFGH")

INFORMATION_SYSTEM = (
    "You are an information agent in a controlled multi-agent workflow. Preserve "
    "and transmit every assigned private status exactly. Do not apply the final "
    "action policy and do not choose an action label."
)
INTEGRATOR_SYSTEM = (
    "You are an information integrator. Preserve every status in the accumulated "
    "communication state without changing, dropping, or inventing information."
)
VERIFIER_SYSTEM = (
    "You are an information verifier. Check that the accumulated communication "
    "preserves every supported private status. Do not infer missing statuses."
)
JOINT_VECTOR_SYSTEM = (
    "You are a joint private-information recovery probe. Answer all ordered binary "
    "status questions from the received communication state in one response. Do not "
    "apply the action policy. The decoder permits only one binary digit per question "
    "and inserts fixed commas between answers."
)
ACTION_SYSTEM = (
    "You are the final deterministic domain policy executor. Apply only the supplied "
    "status record and public policy. Select exactly one action label from A through H."
)


@dataclass(frozen=True)
class PromptSnapshotV2:
    benchmark_version: str
    split: str
    case_id: str
    variant_id: str
    protocol_version: str
    protocol_sha256: str
    role_prompt_sha256: Mapping[str, Mapping[str, str]]
    semantic_question_sha256: str
    semantic_vector_length: int
    action_prompt_template_sha256: str
    prompt_bundle_sha256: str


def prompt_protocol_sha256() -> str:
    return canonical_sha256(
        {
            "version": PROMPT_PROTOCOL_VERSION,
            "information_system": INFORMATION_SYSTEM,
            "integrator_system": INTEGRATOR_SYSTEM,
            "verifier_system": VERIFIER_SYSTEM,
            "joint_vector_system": JOINT_VECTOR_SYSTEM,
            "action_system": ACTION_SYSTEM,
            "role_policy_visibility": "judger_only",
            "semantic_readout": "one_prompt_one_constrained_x_bit_vector",
            "action_candidates": ACTION_LABELS,
        }
    )


def canonical_fact_assertion(fact: FactValue) -> str:
    return f"Private status: {fact.fact_id} ({fact.name}) = {fact.value}."


def _role(case: BenchmarkCase, role_id: str) -> RoleSpec:
    matches = [role for role in case.roles if role.role_id == role_id]
    if len(matches) != 1:
        raise ValueError(f"{case.case_id}: expected exactly one role {role_id}")
    return matches[0]


def _owned_facts(variant: CaseVariant, role: RoleSpec) -> tuple[FactValue, ...]:
    index = {fact.fact_id: fact for fact in variant.facts}
    return tuple(index[fact_id] for fact_id in role.owned_fact_ids)


def build_role_prompt(
    case: BenchmarkCase,
    variant: CaseVariant,
    *,
    role_id: str,
    channel: str,
) -> tuple[dict[str, str], ...]:
    if case.case_id != variant.case_id:
        raise ValueError("case and variant IDs differ")
    if channel not in {"text", "latent"}:
        raise ValueError("channel must be text or latent")
    role = _role(case, role_id)
    if role.role_type == "judger":
        raise ValueError("the Judger uses a readout prompt")
    if role.role_type == "information":
        system = INFORMATION_SYSTEM
        facts = "\n".join(
            canonical_fact_assertion(fact) for fact in _owned_facts(variant, role)
        )
        instruction = (
            "Emit every assigned private status as one exact fact_id=value line. "
            "Preserve earlier visible communication and emit no action label."
            if channel == "text"
            else "Preserve every assigned private status in the internal communication "
            "state for downstream roles. Emit no action label."
        )
        private_section = f"\n\nAssigned private statuses:\n{facts}"
    elif role.role_type == "integrator":
        system = INTEGRATOR_SYSTEM
        instruction = (
            "Emit one concise fact_id=value line for every status preserved in earlier "
            "visible communication."
            if channel == "text"
            else "Integrate the accumulated internal state while preserving every status."
        )
        private_section = ""
    elif role.role_type == "verifier":
        system = VERIFIER_SYSTEM
        instruction = (
            "Emit the verified fact_id=value record from earlier visible communication."
            if channel == "text"
            else "Verify and preserve the accumulated internal status record."
        )
        private_section = ""
    else:
        raise ValueError(f"unsupported role type: {role.role_type}")
    return (
        {"role": "system", "content": f"{system} Your role is {role.role_name}."},
        {
            "role": "user",
            "content": (
                f"Public scenario:\n{case.public_scenario}{private_section}\n\n"
                f"Role instruction:\n{instruction}"
            ),
        },
    )


def append_visible_context(
    messages: Sequence[Mapping[str, str]], visible_context: str
) -> tuple[dict[str, str], ...]:
    copied = [dict(message) for message in messages]
    if not copied or copied[-1].get("role") != "user":
        raise ValueError("the final message must be a user message")
    if visible_context.strip():
        copied[-1]["content"] += (
            "\n\nEarlier visible communication:\n" + visible_context.strip()
        )
    return tuple(copied)


def oracle_visible_context(variant: CaseVariant) -> str:
    return "\n".join(canonical_fact_assertion(fact) for fact in variant.facts)


def information_questions(program: PolicyProgram) -> tuple[str, ...]:
    return tuple(
        f"Q{index}. For {fact.fact_id} ({fact.name}), output 1 if the communicated "
        f"value is {fact.true_value}; output 0 if it is {fact.false_value}."
        for index, fact in enumerate(program.facts, start=1)
    )


def build_joint_vector_prompt(
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
        f"Return exactly {len(program.facts)} binary digits in Q1-Q{len(program.facts)} "
        "order, separated by single commas, with no spaces or explanation."
    )
    if visible_context.strip():
        content += "\n\nReceived communication:\n" + visible_context.strip()
    return (
        {"role": "system", "content": JOINT_VECTOR_SYSTEM},
        {"role": "user", "content": content},
    )


def assignment_from_vector(
    program: PolicyProgram, vector: str
) -> dict[str, bool]:
    if len(vector) != len(program.facts) or set(vector) - {"0", "1"}:
        raise ValueError("information vector has an invalid shape")
    return {
        fact.fact_id: bit == "1"
        for fact, bit in zip(program.facts, vector, strict=True)
    }


def vector_from_variant(variant: CaseVariant) -> str:
    return "".join("1" if fact.active else "0" for fact in variant.facts)


def build_action_prompt(
    case: BenchmarkCase,
    program: PolicyProgram,
    information_vector: str,
) -> tuple[dict[str, str], ...]:
    assignment = assignment_from_vector(program, information_vector)
    status_lines = "\n".join(
        f"- {fact.fact_id} ({fact.name}) = "
        f"{fact.true_value if assignment[fact.fact_id] else fact.false_value} "
        f"(code {int(assignment[fact.fact_id])})"
        for fact in program.facts
    )
    content = (
        f"Domain: {case.domain}\n"
        f"Public scenario:\n{case.public_scenario}\n\n"
        f"Recovered information vector: {information_vector}\n"
        f"Recovered status record:\n{status_lines}\n\n"
        f"Public policy:\n{case.public_policy}\n\n"
        "Return exactly one action label from A, B, C, D, E, F, G, H."
    )
    return (
        {"role": "system", "content": ACTION_SYSTEM},
        {"role": "user", "content": content},
    )


def _messages_sha256(messages: Sequence[Mapping[str, str]]) -> str:
    return canonical_sha256(list(messages))


def build_prompt_snapshot(
    case: BenchmarkCase,
    variant: CaseVariant,
    program: PolicyProgram,
) -> PromptSnapshotV2:
    role_hashes = {
        role.role_id: {
            channel: _messages_sha256(
                build_role_prompt(
                    case, variant, role_id=role.role_id, channel=channel
                )
            )
            for channel in ("text", "latent")
        }
        for role in case.roles
        if role.role_type != "judger"
    }
    semantic_hash = _messages_sha256(build_joint_vector_prompt(case, program))
    action_template_hash = _messages_sha256(
        build_action_prompt(case, program, "0" * len(program.facts))
    )
    payload = {
        "benchmark_version": variant.benchmark_version,
        "split": variant.split,
        "case_id": variant.case_id,
        "variant_id": variant.variant_id,
        "protocol_version": PROMPT_PROTOCOL_VERSION,
        "protocol_sha256": prompt_protocol_sha256(),
        "role_prompt_sha256": role_hashes,
        "semantic_question_sha256": semantic_hash,
        "semantic_vector_length": len(program.facts),
        "action_prompt_template_sha256": action_template_hash,
    }
    return PromptSnapshotV2(
        **payload,
        prompt_bundle_sha256=canonical_sha256(payload),
    )


def build_prompt_snapshots(
    *,
    cases: Sequence[BenchmarkCase],
    variants: Sequence[CaseVariant],
) -> tuple[PromptSnapshotV2, ...]:
    case_index = {case.case_id: case for case in cases}
    # Programs are reconstructed from case-independent IDs by the caller in normal
    # execution. Dataset writing uses this small deterministic rebuild.
    from .policy import build_policy_program
    from latent_audit.semantic_fidelity_v1.blueprints import get_blueprint

    program_cache: dict[str, PolicyProgram] = {}
    snapshots: list[PromptSnapshotV2] = []
    for variant in variants:
        case = case_index[variant.case_id]
        program = program_cache.get(variant.policy_program_id)
        if program is None:
            blueprint = get_blueprint(case.domain)
            facts = blueprint.facts_for_information_level(case.information_level)
            program = build_policy_program(
                policy_program_id=case.policy_program_id,
                semantic_family_id=case.semantic_family_id,
                domain=case.domain,
                information_level=case.information_level,
                reasoning_level=case.reasoning_level,
                facts=facts,
                action_names=blueprint.action_names,
            )
            if program.program_sha256 != case.policy_program_sha256:
                raise ValueError(f"{case.case_id}: reconstructed program hash differs")
            program_cache[variant.policy_program_id] = program
        snapshots.append(build_prompt_snapshot(case, variant, program))
    return tuple(snapshots)


def scan_semantic_prompt_leakage(
    case: BenchmarkCase,
    variant: CaseVariant,
    program: PolicyProgram,
) -> tuple[str, ...]:
    text = "\n".join(
        message["content"] for message in build_joint_vector_prompt(case, program)
    )
    errors: list[str] = []
    for fact in variant.facts:
        assertion = canonical_fact_assertion(fact)
        if assertion in text:
            errors.append(f"{variant.variant_id}:{fact.fact_id}: realized fact leaked")
    for forbidden in ("Gold", "Expected action", "REQUIRED", "NOT_REQUIRED"):
        if forbidden in text:
            errors.append(f"{variant.variant_id}: forbidden token {forbidden!r}")
    return tuple(errors)
