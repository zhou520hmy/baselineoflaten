# B200 latent-communication baseline suite

**Start with the [Chinese operator handoff / 中文接手说明](README_ZH.md).** It contains the complete resume procedure, directory map, metrics, disk requirements and failure handling.

Five methods (LatentMAS, LatentMAS-H2O, LatentMAS-Hidden, LatCom, Interlat), Qwen3-4B / 8B. Seven general tasks use a fixed stratified ~10% sample; the in-house LATEN Benchmark remains **648 variants / 486 pairs in full**. Total: 80 cells / 12,390 task evaluations, plus collection and training. LatCom / Interlat are explicitly nonofficial `paper_derived_port` implementations, not original author checkpoints or V6.

## Resume an existing installation

Wait for previous coordinators to stop, retain the original store and configuration, then update the checkout. Do not delete models, datasets, scores or checkpoints.

```bash
git pull --ff-only
GPU_ID=3 LATEN_STORE=/data/laten_baselines bash run_remaining.sh --family 4b
```

On the second GPU, in a separate terminal/store:

```bash
GPU_ID=5 LATEN_STORE=/data/laten_baselines_8b bash run_remaining.sh --family 8b
```

The store name does **not** select model size. Always use `--family 8b` for that branch. With one GPU, omit `--family` to run both families sequentially. Append `--config configs/local.json` if the original run used that configuration. Never run overlapping coordinators against one store.

Verified environments/assets/reference tests are reused. Completed evaluation identities, IDs and shards are validated and skipped before model inference. Partial evaluations and valid v2 training checkpoints resume. The collection repair uses versioned `training_cache_v2` / `training_collect_v2`; rejected legacy caches are retained and recollected under the corrected eligibility rules. Output-only updates preserve known execution identities after checking unchanged compute sources. Unknown source/config mismatches stop without overwriting prior work.

The reported 90 GiB free is insufficient for safe 8B training. Resume preflight requires at least 150 GiB, and per-stage checkpoint checks can require more. For two training stores on one filesystem, plan approximately 400–500 GiB free; checks do not reserve space against other jobs. Nothing is automatically deleted to make room.

## Deliver only statistics TXT

Export offline, with system Python and no model loading, GPU, download or repeated evaluation:

```bash
LATEN_STORE=/data/laten_baselines bash run.sh export-txt
LATEN_STORE=/data/laten_baselines_8b bash run.sh export-txt
```

`status` and `report` refresh the same output. **Return only `LATEN_STORE/STATS_TXT/*.txt`**, keeping the two stores separate. Wait for export to finish before copying.

Read `00_README.txt`, then `01_INDEX.txt`; the latter lists part order, byte counts and SHA256. Every UTF-8 file is at most **88,000 bytes**, below 90 KB, including Chinese text and headers. Long sections split into `.p001.txt`, `.p002.txt`, etc.

| Prefix | Statistics |
|---|---|
| 02 | Coverage, model revisions, seed, output caps |
| 03 | General-task counts, accuracy, macro/micro aggregates, full-path tokens |
| 04 | Overall LATEN metrics, CF pairs, efficiency |
| 05 | Per-model/method LATEN stratification and conditional fact errors |
| 06 | Training stage progress and numerical losses |
| 07 | Collection scan/retention and method-specific gates |
| 08 | Historical branch failure counts and exception types |
| 09 | Recorded scheduler attempt durations including setup/scoring/retries |

No per-question prompts, answers, generated code, reasoning, token IDs, raw exceptions or traces are exported. New evaluation journals omit generated text. Runtime model/data/training tensors and targets, weights, optimizer state, identity manifests and minimal scoring journals must remain local for continuation; they are **not deliverables**. Existing historical raw files are neither deleted nor exported. An unfinished general sample interrupted before its score was committed may regenerate; committed scores are reused.

Scores are fractions from 0 to 1; `NA` is unavailable, not zero. Macro accuracy is produced only after all seven datasets finish. Partial micro accuracy covers only completed rows. Full-path generated text tokens, latent positions, prompt/prefill positions and transmitted bytes remain distinct. Parallel wall times are contention diagnostics, not isolated-GPU speed comparisons. See [TXT policy](docs/TXT_HANDOFF.md), [collection repair](docs/COLLECT_REPAIR.md), [method protocol](docs/PROTOCOL.md), and [LATEN metric protocol](docs/LATEN_BENCHMARK_PROTOCOL.md).

## Fresh installation

Linux, Python 3.10–3.12 with venv, Git, working Docker, a B200 with at least 170 GiB VRAM, 256 GB RAM, and 500 GiB free disk (1 TB recommended). PyTorch is pinned to 2.7.1 / CUDA 12.8. Downloads need Hugging Face, GitHub, PyTorch and Docker access. Configure credentials/proxies locally.

```bash
GPU_ID=0 LATEN_STORE=/data/laten_baselines bash run.sh
```

Training is serial. Evaluation defaults to two processes on one GPU with disjoint shards and 45% allocator allowance each. OOM shards retry serially at 90%, without dropping tasks or reducing token caps. Other failed branches continue independently; a blocked matrix exits nonzero and is never reported as complete.

## Verification

```bash
bash run.sh validate
.venv/bin/python -m unittest discover -s tests -p 'test_tiny*.py' -v
```

43 static/scheduler/continuation/TXT tests and 16 tiny CPU tests pass locally; see `evidence/validation.json`. These are engineering checks, not a new B200 run or evidence that collection retention, training convergence or benchmark scores have succeeded on the target server.
