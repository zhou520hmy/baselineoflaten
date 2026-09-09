"""Auditable task efficiency aggregation; no torch, model or filesystem access.

Latent positions are deliberately not included in discrete token consumption.
Input records represent complete attempts, including failed and capped attempts.
Warm-up attempts must be marked explicitly and retained in a separate raw file.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping


COUNTERS = (
    "prompt_tokens", "generated_text_tokens", "reasoning_tokens", "answer_tokens",
    "forced_text_tokens", "emitted_text_positions", "stop_token_decisions",
    "generated_latent_positions", "received_message_positions",
    "transmitted_positions", "transmitted_prompt_positions",
    "transmitted_latent_positions", "transmitted_bytes",
    "model_prefill_positions", "adapter_context_tokens",
)


def _number(value: object, name: str, *, integer: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative finite number")
    if not math.isfinite(value) or value < 0 or (integer and int(value) != value):
        raise ValueError(f"{name} must be a non-negative finite {'integer' if integer else 'number'}")
    return int(value) if integer else float(value)


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _stats(values: list[float]) -> dict:
    return {
        "count": len(values), "total": sum(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "p50": _quantile(values, .5), "p95": _quantile(values, .95),
        "min": min(values) if values else None, "max": max(values) if values else None,
    }


def native_decode_counts(*, reasoning_token_count: int, stop_reason: str,
                         close_token_count: int, answer_prefix_token_count: int,
                         separator_token_count: int, vector_length: int,
                         completed: bool) -> dict[str, int]:
    """Count NativeReceiverV4Critic.generate's selected and injected positions.

    An EOS decision is not appended by that decoder; its injected closing tag
    is present in reasoning_token_count. The constrained bit selections are
    model decisions, whereas answer framing and commas are protocol inputs.
    A cap preserves the produced reasoning and has no answer/framing positions.
    """
    args = dict(reasoning_token_count=reasoning_token_count,
                close_token_count=close_token_count,
                answer_prefix_token_count=answer_prefix_token_count,
                separator_token_count=separator_token_count, vector_length=vector_length)
    args = {k: _number(v, k, integer=True) for k, v in args.items()}
    if not isinstance(completed, bool):
        raise ValueError("completed must be boolean")
    if completed and vector_length < 1:
        raise ValueError("a completed vector must contain at least one bit")
    if completed and stop_reason not in {"thinking_close", "eos_mapped_to_thinking_close"}:
        raise ValueError("unexpected completed native decoding stop reason")
    mapped = completed and stop_reason == "eos_mapped_to_thinking_close"
    forced_close = close_token_count if mapped else 0
    if reasoning_token_count < forced_close:
        raise ValueError("reasoning trace is shorter than its injected closing tag")
    reasoning = reasoning_token_count - forced_close
    answer = vector_length if completed else 0
    forced = forced_close + (answer_prefix_token_count + (vector_length - 1) * separator_token_count if completed else 0)
    return dict(reasoning_tokens=reasoning, answer_tokens=answer,
                generated_text_tokens=reasoning + answer, forced_text_tokens=forced,
                emitted_text_positions=reasoning + answer + forced,
                stop_token_decisions=int(mapped))


def summarize_efficiency(records: Iterable[Mapping], *,
                         success_field: str = "information_vector_exact",
                         expected_case_count: int | None = None) -> dict:
    """Summarize observed attempts with explicit coverage and full failure costs.

    Throughput is per sequential worker's summed case wall time, not the
    multi-worker job makespan. An incomplete benchmark can report observed
    throughput, but pending_case_count prevents presenting it as a final score.
    Missing/unsynchronized wall times suppress throughput, not just those rows.
    Counter totals include observed values; mean_per_case requires full coverage.
    """
    source = list(records)
    active = []
    warmup_excluded = 0
    for index, record in enumerate(source):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be a mapping")
        efficiency = record.get("efficiency", {})
        if not isinstance(efficiency, Mapping):
            raise ValueError(f"record {index} efficiency must be a mapping")
        if not isinstance(efficiency.get("warmup", False), bool):
            raise ValueError(f"record {index} warmup must be boolean")
        if efficiency.get("warmup", False):
            warmup_excluded += 1
        else:
            active.append((record, efficiency))
    count = len(active)
    if expected_case_count is not None:
        expected_case_count = _number(expected_case_count, "expected_case_count", integer=True)
        if expected_case_count < count:
            raise ValueError("observed case count exceeds expected_case_count")
    walls, stages, counters = [], {}, {key: [] for key in COUNTERS}
    correct = failures = caps = synchronized = 0
    status_counts = Counter()
    warnings = []
    for index, (record, efficiency) in enumerate(active):
        status = str(record.get("status", "missing"))
        status_counts[status] += 1
        # A failed attempt cannot count as a correctly completed task even if
        # a caller accidentally leaves a stale success flag in its record.
        correct += status == "success" and record.get(success_field) is True
        failures += status != "success"
        caps += bool(record.get("reasoning_cap_hit", record.get("reasoning_capped", False)))
        synchronized += efficiency.get("cuda_synchronized") is True
        if efficiency.get("task_wall_seconds") is not None:
            walls.append(_number(efficiency["task_wall_seconds"], f"row {index} task_wall_seconds"))
        for key in COUNTERS:
            if efficiency.get(key) is not None:
                counters[key].append(_number(efficiency[key], f"row {index} {key}", integer=True))
        stage_data = efficiency.get("stage_seconds", {})
        if not isinstance(stage_data, Mapping):
            raise ValueError(f"row {index} stage_seconds must be a mapping")
        for stage, value in stage_data.items():
            if value is not None:
                stages.setdefault(str(stage), []).append(_number(value, f"row {index} stage {stage}"))
        for total_key, part_keys in (
            ("generated_text_tokens", ("reasoning_tokens", "answer_tokens")),
            ("emitted_text_positions", ("generated_text_tokens", "forced_text_tokens")),
            ("transmitted_positions", ("transmitted_prompt_positions", "transmitted_latent_positions")),
        ):
            if all(efficiency.get(key) is not None for key in (total_key, *part_keys)):
                if efficiency[total_key] != sum(efficiency[key] for key in part_keys):
                    raise ValueError(f"row {index} inconsistent {total_key}")
    complete_timing = count > 0 and len(walls) == count
    total_time = sum(walls)
    rate_valid = complete_timing and synchronized == count and total_time > 0
    if len(walls) != count:
        warnings.append("Incomplete case wall-time coverage; throughput withheld.")
    if synchronized != count:
        warnings.append("Some case boundaries lack CUDA synchronization; throughput withheld.")
    missing_counters = [key for key, values in counters.items() if len(values) != count]
    if missing_counters:
        warnings.append("Incomplete counter coverage: " + ", ".join(missing_counters))
    return {
        "schema_version": "v6_channel_efficiency_v1", "record_count": len(source),
        "warmup_excluded": warmup_excluded, "case_count": count,
        "expected_case_count": expected_case_count,
        "pending_case_count": expected_case_count - count if expected_case_count is not None else None,
        "success_field": success_field, "semantic_correct_count": correct,
        "failure_count": failures, "reasoning_cap_count": caps,
        "status_counts": dict(sorted(status_counts.items())),
        "task_wall_seconds": _stats(walls),
        "all_cases_timing_covered": complete_timing,
        "cuda_synchronized_case_count": synchronized,
        "tasks_per_second": count / total_time if rate_valid else None,
        "correct_tasks_per_second": correct / total_time if rate_valid else None,
        "seconds_per_correct_task": total_time / correct if rate_valid and correct else None,
        "counters": {key: {"count": len(values), "total": sum(values) if values else None,
                           "mean_per_case": sum(values) / count if count and len(values) == count else None}
                     for key, values in counters.items()},
        "stage_seconds": {key: _stats(values) for key, values in sorted(stages.items())},
        "timing_scope": "instrumented_warm_pipeline_includes_role_provenance_copies_and_hashes_excludes_model_load_warmup_and_diagnostic_action_scoring",
        "warnings": warnings,
    }
