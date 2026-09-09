"""Minimal PolicySpec builders, deterministic evaluation, and validation."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import asdict, replace
from functools import reduce
from operator import xor
from typing import Any, Mapping

from .schema import (
    ACTION_LABELS,
    FactSpec,
    GoldRecord,
    PolicyNode,
    PolicyProgram,
    TraceStep,
    canonical_sha256,
)


SUPPORTED_OPERATORS = {
    "IDENTITY",
    "NOT",
    "AND",
    "OR",
    "XOR",
    "LOOKUP",
    "COUNT_GE",
    "ADD",
    "GREATER_THAN",
}


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def program_sha256(program: PolicyProgram) -> str:
    payload = asdict(program)
    payload.pop("program_sha256", None)
    return canonical_sha256(payload)


def _bool_xor(values: list[Any]) -> bool:
    if not values:
        raise ValueError("XOR requires at least one input")
    if not all(isinstance(value, bool) for value in values):
        raise TypeError("XOR inputs must be Boolean")
    return bool(reduce(xor, values))


def _evaluate_node(node: PolicyNode, input_values: list[Any]) -> Any:
    op = node.op
    if op == "IDENTITY":
        if len(input_values) != 1:
            raise ValueError("IDENTITY requires one input")
        return input_values[0]
    if op == "NOT":
        if len(input_values) != 1 or not isinstance(input_values[0], bool):
            raise ValueError("NOT requires one Boolean input")
        return not input_values[0]
    if op == "AND":
        if not input_values or not all(isinstance(value, bool) for value in input_values):
            raise ValueError("AND requires Boolean inputs")
        return all(input_values)
    if op == "OR":
        if not input_values or not all(isinstance(value, bool) for value in input_values):
            raise ValueError("OR requires Boolean inputs")
        return any(input_values)
    if op == "XOR":
        return _bool_xor(input_values)
    if op == "LOOKUP":
        if not input_values or not all(isinstance(value, bool) for value in input_values):
            raise ValueError("LOOKUP requires Boolean inputs")
        table = node.params.get("table")
        if not isinstance(table, Mapping):
            raise ValueError("LOOKUP requires a table object")
        key = "".join("1" if value else "0" for value in input_values)
        if key not in table:
            raise ValueError(f"LOOKUP table lacks key {key}")
        result = table[key]
        if not isinstance(result, bool):
            raise ValueError("LOOKUP results must be Boolean")
        return result
    if op == "COUNT_GE":
        if not all(isinstance(value, bool) for value in input_values):
            raise ValueError("COUNT_GE requires Boolean inputs")
        threshold = node.params.get("threshold")
        if not isinstance(threshold, int):
            raise ValueError("COUNT_GE requires integer threshold")
        return sum(bool(value) for value in input_values) >= threshold
    if op == "ADD":
        if not input_values or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in input_values
        ):
            raise ValueError("ADD requires numeric inputs")
        return sum(input_values)
    if op == "GREATER_THAN":
        if len(input_values) != 1:
            raise ValueError("GREATER_THAN requires one dynamic input")
        threshold = node.params.get("threshold")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise ValueError("GREATER_THAN requires numeric threshold")
        value = input_values[0]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("GREATER_THAN input must be numeric")
        return value > threshold
    raise ValueError(f"unsupported policy operator: {op}")


def evaluate_policy(
    program: PolicyProgram,
    assignment: Mapping[str, bool],
) -> GoldRecord:
    fact_ids = tuple(fact.fact_id for fact in program.facts)
    if set(assignment) != set(fact_ids):
        raise ValueError(
            f"assignment keys do not match program facts: "
            f"expected {sorted(fact_ids)}, got {sorted(assignment)}"
        )
    if not all(isinstance(assignment[fact_id], bool) for fact_id in fact_ids):
        raise TypeError("formal v1 fact assignments must be Boolean")

    values: dict[str, Any] = dict(assignment)
    dependencies: dict[str, set[str]] = {
        fact_id: {fact_id} for fact_id in fact_ids
    }
    distances: dict[str, dict[str, int]] = {
        fact_id: {fact_id: 0} for fact_id in fact_ids
    }
    trace: list[TraceStep] = []

    for node in program.nodes:
        try:
            input_values = [values[input_id] for input_id in node.inputs]
        except KeyError as exc:
            raise ValueError(
                f"node {node.node_id} references unavailable input {exc.args[0]}"
            ) from exc
        result = _evaluate_node(node, input_values)
        values[node.node_id] = result

        node_dependencies: set[str] = set()
        node_distances: dict[str, int] = {}
        for input_id in node.inputs:
            node_dependencies.update(dependencies[input_id])
            for fact_id, distance in distances[input_id].items():
                node_distances[fact_id] = max(
                    node_distances.get(fact_id, 0),
                    distance + 1,
                )
        dependencies[node.node_id] = node_dependencies
        distances[node.node_id] = node_distances
        trace.append(
            TraceStep(
                node_id=node.node_id,
                op=node.op,
                input_values=tuple(input_values),
                result=result,
                layer=node.layer,
            )
        )

    output_values = [values[output] for output in program.outputs]
    if not all(isinstance(value, bool) for value in output_values):
        raise TypeError("all action outputs must be Boolean")
    vector = "".join("1" if value else "0" for value in output_values)
    label = ACTION_LABELS[vector]
    active_actions = tuple(
        action for action, active in zip(program.action_names, output_values) if active
    )
    paths: dict[str, dict[str, int]] = {}
    for fact_id in fact_ids:
        paths[fact_id] = {
            f"action_bit_{index + 1}": distances[output][fact_id]
            for index, output in enumerate(program.outputs)
            if fact_id in dependencies[output]
        }

    return GoldRecord(
        action_vector=vector,
        candidate_label=label,
        active_action_names=active_actions,
        intermediate_values={
            node.node_id: values[node.node_id] for node in program.nodes
        },
        trace=tuple(trace),
        fact_to_action_paths=paths,
    )


def evaluate_action_vector_reference(
    program: PolicyProgram,
    assignment: Mapping[str, bool],
) -> str:
    """Evaluate outputs through an independent recursive reference path."""
    fact_ids = {fact.fact_id for fact in program.facts}
    if set(assignment) != fact_ids:
        raise ValueError("reference assignment keys do not match program facts")
    if not all(type(value) is bool for value in assignment.values()):
        raise TypeError("reference assignments must be Boolean")
    nodes = {node.node_id: node for node in program.nodes}
    memo: dict[str, Any] = dict(assignment)
    visiting: set[str] = set()

    def resolve(identifier: str) -> Any:
        if identifier in memo:
            return memo[identifier]
        if identifier in visiting:
            raise ValueError(f"reference evaluator found a cycle at {identifier}")
        if identifier not in nodes:
            raise ValueError(f"reference evaluator cannot resolve {identifier}")
        visiting.add(identifier)
        node = nodes[identifier]
        values = [resolve(input_id) for input_id in node.inputs]

        if node.op == "IDENTITY":
            if len(values) != 1:
                raise ValueError("reference IDENTITY arity mismatch")
            result = values[0]
        elif node.op == "NOT":
            if len(values) != 1 or type(values[0]) is not bool:
                raise ValueError("reference NOT type mismatch")
            result = not values[0]
        elif node.op == "AND":
            if not values or any(type(value) is not bool for value in values):
                raise ValueError("reference AND type mismatch")
            result = all(values)
        elif node.op == "OR":
            if not values or any(type(value) is not bool for value in values):
                raise ValueError("reference OR type mismatch")
            result = any(values)
        elif node.op == "XOR":
            if not values or any(type(value) is not bool for value in values):
                raise ValueError("reference XOR type mismatch")
            result = sum(int(value) for value in values) % 2 == 1
        elif node.op == "LOOKUP":
            if not values or any(type(value) is not bool for value in values):
                raise ValueError("reference LOOKUP type mismatch")
            table = node.params.get("table")
            if not isinstance(table, Mapping):
                raise ValueError("reference LOOKUP table is invalid")
            key = "".join("1" if value else "0" for value in values)
            if key not in table or type(table[key]) is not bool:
                raise ValueError(f"reference LOOKUP lacks Boolean key {key}")
            result = table[key]
        elif node.op == "COUNT_GE":
            threshold = node.params.get("threshold")
            if (
                any(type(value) is not bool for value in values)
                or type(threshold) is not int
            ):
                raise ValueError("reference COUNT_GE type mismatch")
            result = sum(int(value) for value in values) >= threshold
        elif node.op == "ADD":
            if not values or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in values
            ):
                raise ValueError("reference ADD type mismatch")
            result = sum(values)
        elif node.op == "GREATER_THAN":
            threshold = node.params.get("threshold")
            if (
                len(values) != 1
                or not isinstance(values[0], (int, float))
                or isinstance(values[0], bool)
                or not isinstance(threshold, (int, float))
                or isinstance(threshold, bool)
            ):
                raise ValueError("reference GREATER_THAN type mismatch")
            result = values[0] > threshold
        else:
            raise ValueError(f"reference evaluator does not support {node.op}")

        visiting.remove(identifier)
        memo[identifier] = result
        return result

    outputs = [resolve(output) for output in program.outputs]
    if any(type(value) is not bool for value in outputs):
        raise TypeError("reference action outputs must be Boolean")
    return "".join("1" if value else "0" for value in outputs)


def all_assignments(facts: tuple[FactSpec, ...]) -> tuple[dict[str, bool], ...]:
    fact_ids = tuple(fact.fact_id for fact in facts)
    return tuple(
        dict(zip(fact_ids, values))
        for values in itertools.product((False, True), repeat=len(fact_ids))
    )


def action_algebraic_degrees(program: PolicyProgram) -> tuple[int, int, int]:
    """Return each action bit's algebraic-normal-form degree over GF(2)."""
    fact_ids = tuple(fact.fact_id for fact in program.facts)
    width = len(fact_ids)
    truth_tables = [[0] * (1 << width) for _ in range(3)]
    for mask in range(1 << width):
        assignment = {
            fact_id: bool(mask & (1 << index))
            for index, fact_id in enumerate(fact_ids)
        }
        vector = evaluate_action_vector_reference(program, assignment)
        for output_index, bit in enumerate(vector):
            truth_tables[output_index][mask] = int(bit)

    degrees: list[int] = []
    for values in truth_tables:
        coefficients = list(values)
        for bit_index in range(width):
            for mask in range(1 << width):
                if mask & (1 << bit_index):
                    coefficients[mask] ^= coefficients[mask ^ (1 << bit_index)]
        degrees.append(
            max(
                (mask.bit_count() for mask, active in enumerate(coefficients) if active),
                default=0,
            )
        )
    return degrees[0], degrees[1], degrees[2]


def validate_program_structure(program: PolicyProgram) -> None:
    if program.domain == "":
        raise ValueError("program domain must not be empty")
    expected_count = {"I3": 3, "I6": 6, "I9": 9}.get(program.information_level)
    if expected_count is None:
        raise ValueError(f"unsupported information level: {program.information_level}")
    if len(program.facts) != expected_count:
        raise ValueError(
            f"{program.information_level} requires {expected_count} facts, "
            f"got {len(program.facts)}"
        )
    owner_counts = {
        owner: sum(fact.owner_agent_index == owner for fact in program.facts)
        for owner in (1, 2, 3)
    }
    if len(set(owner_counts.values())) != 1:
        raise ValueError(f"facts are not equally allocated: {owner_counts}")

    fact_ids = [fact.fact_id for fact in program.facts]
    node_ids = [node.node_id for node in program.nodes]
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("fact IDs must be unique")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("policy node IDs must be unique")
    if set(fact_ids) & set(node_ids):
        raise ValueError("fact and node IDs must be disjoint")
    if len(program.outputs) != 3 or any(output not in node_ids for output in program.outputs):
        raise ValueError("policy must expose three valid action outputs")

    available = set(fact_ids)
    previous_layer = 0
    node_by_id: dict[str, PolicyNode] = {}
    for node in program.nodes:
        if node.op not in SUPPORTED_OPERATORS:
            raise ValueError(f"unsupported operator in program: {node.op}")
        if node.layer < previous_layer:
            raise ValueError("policy nodes must be ordered by nondecreasing layer")
        if not node.inputs:
            raise ValueError(f"node {node.node_id} has no inputs")
        missing = set(node.inputs) - available
        if missing:
            raise ValueError(f"node {node.node_id} has unavailable inputs: {sorted(missing)}")
        available.add(node.node_id)
        node_by_id[node.node_id] = node
        previous_layer = node.layer

    probe = evaluate_policy(
        program,
        {fact_id: False for fact_id in fact_ids},
    )
    missing_paths = [
        fact_id
        for fact_id, paths in probe.fact_to_action_paths.items()
        if not paths
    ]
    if missing_paths:
        raise ValueError(f"facts have no output dependency path: {missing_paths}")

    if program.reasoning_level == "LOOKUP":
        if any(node_by_id[output].op != "LOOKUP" for output in program.outputs):
            raise ValueError("LOOKUP outputs must be direct lookup nodes")
        for output in program.outputs:
            owners = {
                next(
                    fact.owner_agent_index
                    for fact in program.facts
                    if fact.fact_id == input_id
                )
                for input_id in node_by_id[output].inputs
            }
            if len(owners) != 1:
                raise ValueError("LOOKUP outputs must use one owner's local facts")
    elif program.reasoning_level == "RULE":
        if max(node.layer for node in program.nodes) > 2:
            raise ValueError("RULE programs may use at most two operator layers")
        for output_index, output in enumerate(program.outputs, start=1):
            owners = {
                fact.owner_agent_index
                for fact in program.facts
                if f"action_bit_{output_index}"
                in probe.fact_to_action_paths[fact.fact_id]
            }
            if len(owners) < 2:
                raise ValueError("each RULE output must depend on at least two agents")
    elif program.reasoning_level == "DERIVE":
        layers = {node.layer for node in program.nodes}
        if not {1, 2, 3, 4}.issubset(layers):
            raise ValueError("DERIVE programs require four operator layers")
        if not any(node.op in {"AND", "OR", "COUNT_GE"} for node in program.nodes):
            raise ValueError("DERIVE programs require a non-parity constraint node")
        if max(
            distance
            for paths in probe.fact_to_action_paths.values()
            for distance in paths.values()
        ) < 4:
            raise ValueError("DERIVE programs require a four-operation path")
        for output_index in range(1, 4):
            owners = {
                fact.owner_agent_index
                for fact in program.facts
                if f"action_bit_{output_index}"
                in probe.fact_to_action_paths[fact.fact_id]
            }
            if len(owners) < 2:
                raise ValueError("each DERIVE output must depend on multiple agents")
    else:
        raise ValueError(f"unsupported reasoning level: {program.reasoning_level}")

    degrees = action_algebraic_degrees(program)
    if program.reasoning_level in {"LOOKUP", "RULE"} and degrees != (1, 1, 1):
        raise ValueError(
            f"{program.reasoning_level} outputs must have degree one, got {degrees}"
        )
    if program.reasoning_level == "DERIVE" and any(degree < 2 for degree in degrees):
        raise ValueError(
            f"every DERIVE output must retain a nonlinear interaction, got {degrees}"
        )

    actual_hash = program_sha256(program)
    if program.program_sha256 != actual_hash:
        raise ValueError(
            f"program hash mismatch: expected {program.program_sha256}, got {actual_hash}"
        )


def _lookup_table(input_count: int) -> dict[str, bool]:
    table: dict[str, bool] = {}
    for values in itertools.product((False, True), repeat=input_count):
        table["".join("1" if value else "0" for value in values)] = _bool_xor(
            list(values)
        )
    return table


INFLUENCE_COLUMNS = {
    (1, 1): (1, 1, 0),
    (2, 1): (0, 1, 1),
    (3, 1): (1, 1, 1),
    (1, 2): (1, 0, 1),
    (2, 2): (1, 1, 0),
    (3, 2): (0, 1, 1),
    (1, 3): (1, 1, 1),
    (2, 3): (1, 0, 1),
    (3, 3): (1, 1, 0),
}


def _lookup_program_nodes(
    facts: tuple[FactSpec, ...],
) -> tuple[tuple[PolicyNode, ...], tuple[str, str, str], tuple[str, ...]]:
    nodes: list[PolicyNode] = []
    rules: list[str] = []
    outputs: list[str] = []
    for owner in (1, 2, 3):
        owner_facts = tuple(
            fact for fact in facts if fact.owner_agent_index == owner
        )
        node_id = f"action_bit_{owner}"
        table = _lookup_table(len(owner_facts))
        nodes.append(
            PolicyNode(
                node_id=node_id,
                op="LOOKUP",
                inputs=tuple(fact.fact_id for fact in owner_facts),
                layer=1,
                params={"table": table},
                description=f"Local lookup for information agent {owner}",
            )
        )
        rows = ", ".join(
            f"{key}->{int(value)}" for key, value in sorted(table.items())
        )
        rules.append(
            f"Action bit {owner} uses the local status order "
            f"{', '.join(fact.name for fact in owner_facts)} with table: {rows}."
        )
        outputs.append(node_id)
    return (
        tuple(nodes),
        (outputs[0], outputs[1], outputs[2]),
        tuple(rules),
    )


def _rule_program_nodes(
    facts: tuple[FactSpec, ...],
) -> tuple[tuple[PolicyNode, ...], tuple[str, str, str], tuple[str, ...]]:
    nodes: list[PolicyNode] = []
    outputs: list[str] = []
    rules: list[str] = []
    for bit_index in range(3):
        selected = tuple(
            fact
            for fact in facts
            if INFLUENCE_COLUMNS[(fact.owner_agent_index, fact.slot_index)][bit_index]
        )
        output_id = f"action_bit_{bit_index + 1}"
        nodes.append(
            PolicyNode(
                node_id=output_id,
                op="XOR",
                inputs=tuple(fact.fact_id for fact in selected),
                layer=1,
                params={},
                description=f"Direct cross-agent rule for action bit {bit_index + 1}",
            )
        )
        rules.append(
            f"Action bit {bit_index + 1} is 1 exactly when an odd number of "
            f"the following status bits are 1: "
            f"{', '.join(fact.name for fact in selected)}."
        )
        outputs.append(output_id)
    return (
        tuple(nodes),
        (outputs[0], outputs[1], outputs[2]),
        tuple(rules),
    )


def _derive_program_nodes(
    facts: tuple[FactSpec, ...],
) -> tuple[tuple[PolicyNode, ...], tuple[str, str, str], tuple[str, ...]]:
    nodes: list[PolicyNode] = []
    rules: list[str] = []
    local_ids: list[str] = []
    for owner in (1, 2, 3):
        owner_facts = tuple(
            fact for fact in facts if fact.owner_agent_index == owner
        )
        node_id = f"local_state_{owner}"
        nodes.append(
            PolicyNode(
                node_id=node_id,
                op="XOR",
                inputs=tuple(fact.fact_id for fact in owner_facts),
                layer=1,
                params={},
                description=f"Derived local state for information agent {owner}",
            )
        )
        rules.append(
            f"First derive local state {owner} as odd parity over "
            f"{', '.join(fact.name for fact in owner_facts)}."
        )
        local_ids.append(node_id)

    constraint_id = "joint_constraint"
    nodes.append(
        PolicyNode(
            node_id=constraint_id,
            op="AND",
            inputs=(local_ids[0], local_ids[1]),
            layer=2,
            params={},
            description="Joint constraint derived from local states 1 and 2",
        )
    )
    rules.append(
        "Then activate the joint constraint only when both local state 1 and "
        "local state 2 are 1."
    )

    resolved_ids = (
        "resolved_state_1",
        "resolved_state_2",
        "resolved_state_3",
    )
    nodes.extend(
        (
            PolicyNode(
                node_id=resolved_ids[0],
                op="IDENTITY",
                inputs=(local_ids[0],),
                layer=3,
                params={},
                description="Preserved local state 1 after constraint evaluation",
            ),
            PolicyNode(
                node_id=resolved_ids[1],
                op="IDENTITY",
                inputs=(local_ids[1],),
                layer=3,
                params={},
                description="Preserved local state 2 after constraint evaluation",
            ),
            PolicyNode(
                node_id=resolved_ids[2],
                op="XOR",
                inputs=(local_ids[2], constraint_id),
                layer=3,
                params={},
                description="Priority-adjusted local state 3",
            ),
        )
    )
    rules.extend(
        (
            "Preserve local state 1 as resolved state 1.",
            "Preserve local state 2 as resolved state 2.",
            "Apply the joint constraint as a priority override: resolved state "
            "3 is the odd-parity result over local state 3 and the joint constraint.",
        )
    )

    cross_inputs = (
        (resolved_ids[0], resolved_ids[2]),
        (resolved_ids[0], resolved_ids[1], resolved_ids[2]),
        (resolved_ids[1], resolved_ids[2]),
    )
    output_ids: list[str] = []
    for bit_index, inputs in enumerate(cross_inputs, start=1):
        node_id = f"action_bit_{bit_index}"
        nodes.append(
            PolicyNode(
                node_id=node_id,
                op="XOR",
                inputs=inputs,
                layer=4,
                params={},
                description=f"Final cross-agent action bit {bit_index}",
            )
        )
        rules.append(
            f"Finally set action bit {bit_index} to 1 exactly when an odd "
            f"number of {', '.join(inputs)} are 1."
        )
        output_ids.append(node_id)
    return (
        tuple(nodes),
        (output_ids[0], output_ids[1], output_ids[2]),
        tuple(rules),
    )


def build_policy_program(
    *,
    policy_program_id: str,
    semantic_family_id: str,
    domain: str,
    information_level: str,
    reasoning_level: str,
    facts: tuple[FactSpec, ...],
    action_names: tuple[str, str, str],
) -> PolicyProgram:
    if reasoning_level == "LOOKUP":
        nodes, outputs, public_rules = _lookup_program_nodes(facts)
    elif reasoning_level == "RULE":
        nodes, outputs, public_rules = _rule_program_nodes(facts)
    elif reasoning_level == "DERIVE":
        nodes, outputs, public_rules = _derive_program_nodes(facts)
    else:
        raise ValueError(f"unsupported reasoning level: {reasoning_level}")

    program = PolicyProgram(
        policy_program_id=policy_program_id,
        semantic_family_id=semantic_family_id,
        domain=domain,
        information_level=information_level,
        reasoning_level=reasoning_level,
        facts=facts,
        nodes=nodes,
        outputs=outputs,
        action_names=action_names,
        public_rules=public_rules,
        program_sha256="",
    )
    program = replace(program, program_sha256=program_sha256(program))
    validate_program_structure(program)
    return program
