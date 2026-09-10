# Protocol and evidence boundary

> Collection and continuation update (2026-09-10): [COLLECT_REPAIR.md](COLLECT_REPAIR.md) supersedes the old shared collection gate and fail-fast scheduling. Evaluation inputs, sampling and scorers remain unchanged; learned methods use versioned new training/evaluation directories.

> Current execution revision (2026-09-09): [SAMPLED_PARALLEL_PROTOCOL.md](SAMPLED_PARALLEL_PROTOCOL.md) controls evaluation scheduling and scope. General tasks use fixed ~10% subsets (591 total); LATEN remains full 648/486. Default inference concurrency is two, training stays serial. There are 80 cells / 12,390 evaluated tasks. Older full-general counts and serial timing descriptions below document the source protocol, not the current default execution. New paths: evaluation_sample10 and semantic_parallel.

This is a portable **paper-derived comparison suite**, not a claim of exact author-checkpoint reproduction. It uses Qwen/Qwen3-4B and Qwen/Qwen3-8B (the public chat/thinking releases, pinned commits), not an invented 4B-Base model ID. It runs on one physical B200, with no distributed or multi-GPU fallback.

## Source audit

- LatentMAS: Gen-Verse/LatentMAS, commit `9a9e4d331eb11430bd9e64754c6b252b06d73031`; public hierarchical prompts are vendored from this unmodified commit. The official hierarchy implementation carries past KV between sender calls. Hierarchical role names do not imply independent sender caches.
- Interlat: XiaoDu-flying/Interlat, commit `66a89cb4d4097b2f86cbe48ed9851d6e8578f821`; paper arXiv:2511.09149v5. Official experiments primarily use ALFWorld/MATH and Qwen2.5/LLaMA. This suite ports its receiver objective, adapter and learned autoregressive compression to Qwen3 and the shared general-task interface. No Qwen2.5 or V6 adapter is relabeled as Interlat.
- LatCom: author-provided LatCom manuscript (not redistributed), SHA256 `136736974403008fbc5e224f156bf20992d76eb22fdf4313a624f99f47ff4c0a`. No verified author repository or trained weights was found. The implementation follows its equations; unspecified settings are listed below.
- B200 runtime: PyTorch 2.7.1 CUDA 12.8 wheels, Transformers 4.57.6; HF/PyTorch runtime only, no vLLM. SDPA for prefill and answer generation; eager attention during latent rollout for all LatentMAS-derived methods, enabling exact headwise H2O attention collection.

## General-task matrix

Five methods × two Qwen3 scales × seven datasets = 70 evaluation cells. All cells use the same locked rows, prompts, deterministic per-example random seed, temperature 0.6 and top-p 0.95. Top-k is disabled because LatCom does not specify it. Three senders use math/science/code role prompts. The final receiver has the public question, with no gold answers or hidden test code in model prompts. A single sampled answer per example yields accuracy or code pass@1, not pass@k.

Dataset counts: GSM8K 1319; ARC-E 2376; ARC-C 1172; MedQA 300; GPQA-Diamond 198; MBPP+ 378; HumanEval+ 164. Total 5907. MedQA is the pinned LatentMAS bundled subset. Code tasks use the pinned HF expanded-test parquet releases, matching the existing LATEN dataset protocol; these are **not** silently replaced with newly regenerated EvalPlus v0.3.1 tests or base-only tests. Their versions/counts are recorded with hashes.

The final token caps are 2048 for GSM8K/ARC, 4096 for MedQA/code, 8192 for GPQA. No adaptive reduction after failure. Sender latent budget is 40 per source for LatentMAS/Hidden/H2O/LatCom. LatCom has 64 global slots. Interlat's trained autoregressive compressor emits 21 hidden states per sender, 63 total, following the budget in the LatCom comparison; it is not an untrained 21-step baseline.

## Explicit implementation decisions, not recovered author settings

1. Interlat is trained on the same new evidence/rationale corpus as LatCom, with an explicit question-to-general-task transfer setup. This is different from original Interlat ALFWorld/MATH training. Its core receiver training and compression are separate stages.
2. LatCom uses a same-size pretrained Qwen3 backbone without its LM head, trained fully, with q/role/separator embeddings and learned slots. Sender/receiver are frozen. Stage1 uses task CE, receiver-grounded mismatched-message margin and full-trajectory JS plus mean-vector L2 anchor. Stage2 uses task CE and gold-source-replacement contrast.
3. The paper gives coefficient ranges but no reproducible dynamic rule. Here coefficients use `clamp(0.1 * detached_task / abs(detached_aux), lower, upper)`. Steps, corpus sizes, stage2 auxiliary proportions and loss weights are explicit in the JSON. These defaults establish a reproducible port, **not convergence or equality to the paper's training budget**.
4. H2O retains 64 own-prompt positions per KV head per sender, plus all previously retained history and all actual latent positions; its unspecified paper budget is exposed in config. Latent attention is aggregated per KV head, with chronological ordering preserved after headwise selection. Absolute rotary positions are not reset after pruning. This is not a 64-position total message or an implementation of every mechanism in the original standalone H2O paper.
5. All text prompts are shared, but carrier injection follows the method: full KV reuse for LatentMAS/H2O; prompt embeddings + actual aligned latent input vectors inserted into the receiver user message for Hidden; compressed slots before the entire receiver prompt for LatCom; learned boundary embeddings around Interlat communication after the receiver question.
6. Interlat compression training recomputes differentiable prefixes with gradient checkpointing instead of retaining a large backward KV graph. Inference uses cached recurrence. This changes scheduling, not the intended recurrence. B200 full-model training uses FP32 trainable weights with BF16 autocast and optional saved-activation CPU offload, microbatch1 and explicit global-batch accumulation.

## Reporting rules

The report always carries `paper_derived_port` provenance for learned methods. Never mix its results with published author numbers as if runtime/checkpoints were matched. Record training cost separately from inference. Report whole-task synchronized warm latency, output token IDs across all agents, prompt tokens, model-prefill positions, generated latent positions, cumulative communication positions/bytes and peak allocated GPU memory. Communication volume and token output are separate quantities.

Infrastructure failures (OOM, corrupt data, Docker launch failure) are typed errors and stop that cell; they are not scored as incorrect answers. Normal model parse failures, wrong answers, generation cap and code timeouts remain in the scoring denominator. Incomplete cells are explicitly incomplete and cannot generate a final all-suite completed marker.

Sources: https://github.com/Gen-Verse/LatentMAS ; https://github.com/XiaoDu-flying/Interlat ; https://arxiv.org/abs/2511.09149 ; https://pytorch.org/blog/pytorch-2-7/ ; https://pytorch.org/get-started/previous-versions/ ; pinned HF repos in `evidence/hf_sources.json`.

## Exact efficiency accounting

`task_wall_seconds` starts after model loading/realignment setup and includes all three senders, the compressor/adapter, receiver prefill and receiver decoding. It is a synchronized single-process warm inference measurement; code-test execution, downloads, training and setup are separate. Sender/receiver stage timings and code-scoring seconds remain in raw records.

`generated_text_tokens` counts sampled token IDs (including a sampled EOS) across the task. These methods do not decode intermediate textual plans during evaluation. `generated_latent_positions` counts continuous recurrent positions and is **not** added to the text-token count. `prompt_tokens` counts all four agent prompts. `model_prefill_positions` adds injected embeddings and, for LatCom, the compressor's question/roles/trajectories/separators/slots. Recurrent positions are reported separately. Interlat's learned BOS is one of its recurrent decode inputs; its message contains 63 hidden states plus six learned boundary vectors.

`transmitted_positions` and `transmitted_bytes` sum logical agent-edge payloads, not physical network traffic (all agents are colocated). LatentMAS/H2O sum the full cache after each of the three sender calls; retransmitted history counts on every edge. Hidden and LatCom also count the two internal sender-to-sender KV relays used by the pinned hierarchical implementation, separately visible as `internal_cache_relay_*`. Hidden adds the three prompt+latent payloads to the final receiver. LatCom adds source-to-compressor trajectories and compressor-to-receiver slots. Interlat sums the three adapted messages with boundary vectors. KV bytes include both K and V for all layers/heads; embedding bytes use the actual runtime dtype. A KV position and an embedding position have different storage sizes, so compare byte counts, not positions alone.

## Training boundary tokens and validation limits

Interlat uses learned continuous begin/end vectors rather than extending Qwen3's token vocabulary. The source objective and adapter structure inform the implementation; this detail, training corpus, task prompts, curriculum schedule and budgets are port decisions. No V6 or LoRA checkpoint is used.

Portable contract tests use local fixtures. Numeric tests use randomly initialized tiny Qwen3 CPU models, proving basic interface and gradient behavior only. Neither the full models, B200 memory headroom, training convergence nor benchmark accuracy have been measured in the preparation environment. The target-server pipeline performs a BF16 CUDA check and three short real-model interface smokes before evaluating a family. A learned method is evaluated only after its training export exists with a matching run identity and weight hash.

## Added project Benchmark

The 70 general-task cells remain intact. The frozen LATEN Benchmark adds 10 cells with 648 variants each. The original full-general matrix had 65,550 attempts; the current default has 80 cells and 12,390 attempts after general-task sampling. See [LATEN_BENCHMARK_PROTOCOL.md](LATEN_BENCHMARK_PROTOCOL.md) for immutable data, original prompt/scorer reuse, serial communication adaptations, 2048-token native decoding, efficiency scope and previous-release export compatibility. Its semantic metrics are reported separately from the seven-task macro.
