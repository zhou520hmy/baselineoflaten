"""Typed records and canonical serialization for semantic-fidelity v1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


ACTION_LABELS = {
    "000": "A",
    "001": "B",
    "010": "C",
    "011": "D",
    "100": "E",
    "101": "F",
    "110": "G",
    "111": "H",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def record_to_dict(value: Any) -> dict[str, Any]:
    if not hasattr(value, "__dataclass_fields__"):
        raise TypeError(f"expected dataclass record, got {type(value)!r}")
    return asdict(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: list[Any] | tuple[Any, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        canonical_json(record_to_dict(value) if hasattr(value, "__dataclass_fields__") else value)
        + "\n"
        for value in values
    )
    path.write_text(payload, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


@dataclass(frozen=True)
class SourceRecord:
    source_family_id: str
    split: str
    domain: str
    repository: str
    commit: str
    file: str
    index: int
    record_sha256: str
    user_query: str
    agent_names: tuple[str, ...]
    agent_descriptions: tuple[str, ...]


@dataclass(frozen=True)
class FactSpec:
    fact_id: str
    owner_agent_index: int
    slot_index: int
    name: str
    false_value: str
    true_value: str


@dataclass(frozen=True)
class PolicyNode:
    node_id: str
    op: str
    inputs: tuple[str, ...]
    layer: int
    params: Mapping[str, Any]
    description: str


@dataclass(frozen=True)
class PolicyProgram:
    policy_program_id: str
    semantic_family_id: str
    domain: str
    information_level: str
    reasoning_level: str
    facts: tuple[FactSpec, ...]
    nodes: tuple[PolicyNode, ...]
    outputs: tuple[str, str, str]
    action_names: tuple[str, str, str]
    public_rules: tuple[str, ...]
    program_sha256: str


@dataclass(frozen=True)
class TraceStep:
    node_id: str
    op: str
    input_values: tuple[Any, ...]
    result: Any
    layer: int


@dataclass(frozen=True)
class GoldRecord:
    action_vector: str
    candidate_label: str
    active_action_names: tuple[str, ...]
    intermediate_values: Mapping[str, Any]
    trace: tuple[TraceStep, ...]
    fact_to_action_paths: Mapping[str, Mapping[str, int]]


@dataclass(frozen=True)
class FactValue:
    fact_id: str
    owner_agent_index: int
    slot_index: int
    name: str
    active: bool
    value: str


@dataclass(frozen=True)
class RoleSpec:
    role_id: str
    role_name: str
    position: int
    role_type: str
    owned_fact_ids: tuple[str, ...]
    sees_public_scenario: bool
    sees_public_policy: bool
    sees_accumulated_state: bool
    has_gold_access: bool


@dataclass(frozen=True)
class CounterfactualRecord:
    variant_id: str
    target_fact_id: str
    target_agent_index: int
    target_slot_index: int
    replacement_active: bool
    replacement_value: str
    changed_action_bits: tuple[int, ...]
    expected_action_vector: str
    expected_label: str


@dataclass(frozen=True)
class BenchmarkCase:
    benchmark_version: str
    split: str
    case_id: str
    semantic_family_id: str
    domain: str
    information_level: str
    reasoning_level: str
    graph_level: str
    source_family_id: str
    public_scenario: str
    public_policy: str
    policy_program_id: str
    policy_program_sha256: str
    roles: tuple[RoleSpec, ...]
    facts: tuple[FactValue, ...]
    base_gold: GoldRecord
    counterfactuals: tuple[CounterfactualRecord, ...]


@dataclass(frozen=True)
class CaseVariant:
    benchmark_version: str
    split: str
    case_id: str
    variant_id: str
    semantic_family_id: str
    domain: str
    information_level: str
    reasoning_level: str
    graph_level: str
    source_family_id: str
    policy_program_id: str
    target_fact_id: str | None
    target_agent_index: int | None
    target_slot_index: int | None
    facts: tuple[FactValue, ...]
    gold: GoldRecord


def fact_spec_from_dict(row: Mapping[str, Any]) -> FactSpec:
    return FactSpec(
        fact_id=str(row["fact_id"]),
        owner_agent_index=int(row["owner_agent_index"]),
        slot_index=int(row["slot_index"]),
        name=str(row["name"]),
        false_value=str(row["false_value"]),
        true_value=str(row["true_value"]),
    )


def policy_node_from_dict(row: Mapping[str, Any]) -> PolicyNode:
    params = row.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError("policy node params must be an object")
    return PolicyNode(
        node_id=str(row["node_id"]),
        op=str(row["op"]),
        inputs=tuple(str(value) for value in row["inputs"]),
        layer=int(row["layer"]),
        params=dict(params),
        description=str(row["description"]),
    )


def policy_program_from_dict(row: Mapping[str, Any]) -> PolicyProgram:
    outputs = tuple(str(value) for value in row["outputs"])
    actions = tuple(str(value) for value in row["action_names"])
    if len(outputs) != 3 or len(actions) != 3:
        raise ValueError("policy programs require exactly three outputs and actions")
    return PolicyProgram(
        policy_program_id=str(row["policy_program_id"]),
        semantic_family_id=str(row["semantic_family_id"]),
        domain=str(row["domain"]),
        information_level=str(row["information_level"]),
        reasoning_level=str(row["reasoning_level"]),
        facts=tuple(fact_spec_from_dict(value) for value in row["facts"]),
        nodes=tuple(policy_node_from_dict(value) for value in row["nodes"]),
        outputs=(outputs[0], outputs[1], outputs[2]),
        action_names=(actions[0], actions[1], actions[2]),
        public_rules=tuple(str(value) for value in row["public_rules"]),
        program_sha256=str(row["program_sha256"]),
    )
