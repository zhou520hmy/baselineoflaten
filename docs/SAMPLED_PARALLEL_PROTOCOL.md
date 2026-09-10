# Sampled parallel evaluation revision

> Collection and continuation update (2026-09-10): [COLLECT_REPAIR.md](COLLECT_REPAIR.md) supersedes the old shared collection gate and fail-fast scheduling. Evaluation inputs, sampling and scorers remain unchanged; learned methods use versioned new training/evaluation directories.

Authorized on 2026-09-09: retain approximately one tenth of every general evaluation dataset, preserve complex tasks, and run inference concurrently on one B200. Training remains serial and unchanged. Existing method/source/metric scans remain in PROTOCOL.md and LATEN_BENCHMARK_PROTOCOL.md; this is a scheduling and evaluation-scope change, not a new method.

## Fixed selection, before model outcomes

Seven general datasets use round-half-up 10% counts: GSM8K 132, ARC-E 238, ARC-C 117, MedQA 30, MBPP+ 38, HumanEval+ 16, GPQA-Diamond 20 (591 total). Sort by a declared complexity proxy, divide into three equal-size rank strata, draw nearly equal counts with seed 42. For non-code tasks the proxy is public question length; for code it is the number of branch/loop/comprehension nodes in the reference program followed by question length. These are complexity proxies, NOT verified difficulty labels. No model correctness, output, cap hit or runtime is used for selection. ARC-C and GPQA remain present at the same sampling fraction. All ten method/model combinations share the same saved IDs and selected input records. Full source files and train/test exclusion remain intact.

LATEN remains FULL: 162 cases, 648 variants, 486 BASE/CF pairs. The user clarified that only the seven general datasets are reduced. Original G4/G5/G6, I3/I6/I9, LOOKUP/RULE/DERIVE and dev/test proportions are unchanged. Total evaluation volume is (591 + 648) x 5 x 2 = 12,390 tasks across 80 cells.

## Scheduling and measurement

Two independent inference workers process disjoint fixed shards of the same method/model on the same visible GPU. Each owns its model, RNG and journals; no shared mutable KV or decoding RNG. Training and collection remain serial. GPU model loading/realignment is serialized by a local setup lock. Each concurrent worker has a PyTorch allocator limit of 45% of total device memory; serial OOM retries use 90%. CUDA context and non-PyTorch allocations are outside this allocator limit, so headroom is intentional. OOM exits are typed infrastructure failures, never incorrect answers. Failed OOM shards resume serially after all active workers exit; an OOM in that retry stops with saved results retained. No automatic token-cap reduction or example dropping.

Each row records configured workers, runtime concurrency bound and retry mode. End-to-end task timings remain diagnostics under contention, not isolated latency comparisons. Token/latent/prefill/communication counters retain existing whole-path definitions. Summed overlapping task durations are labelled accumulated task time; reciprocals of this sum are not reported as real parallel throughput. The coordinator separately records stage wall time including model loading, retries and scoring.

## Validation and release boundary

Validate deterministic membership across input ordering and seeds, all strata, whole BASE/CF groups, exact denominators, shard disjointness/union, merging, resume after OOM, stop on other failures, and tiny CPU inference protocols. Full source models and B200 execution remain target-server work. Use fresh LATEN_STORE for a new training run. Completed prior exports may be used by eval-all with exact known release/config/hash checks; old optimizer state is not migrated. Sample outputs have separate directory names and cannot mix with previous full evaluation rows.
