# Collection repair, 2026-09-10

## Evidence and boundaries

User reports zero retained MAIN examples on two remote Qwen3-4B runs. Remote raw artifacts are unavailable locally. The existing 64 .pt files cannot be assumed to be 64 filtered main candidates: default auxiliary math/code total 64 and only retained rows create .pt assets. No claims about observed rejection distributions are independently verified.

Confirmed implementation defects: decoded EOS/chat controls were preserved in exact-answer matching (Answer: Paris<|im_end|> is falsely rejected); the full-latent filter tested concatenated multi-source gold trajectories whereas stage-one distillation uses a fresh single gold trajectory; rejection rows lacked model outputs; one global latent-readability gate blocked Interlat despite its trainable Receiver and text-plan reference. The supplied LatCom appendix A.1 lines 805-846 retains pre-compression answerability and defines stage one using a gold-evidence trajectory. Interlat's collection/training source uses text-plan supervision and a trainable receiver. Fixes preserve LatCom's strict gate; Interlat has an explicit, independent verified-text-plan gate. This is a documented corpus-selection change to the nonofficial Interlat port, not an assertion of exact author reproduction.

## Implementation

Remove only known chat controls before final-span comparison, require a closed final-answer region, normalize punctuation/articles for QA exact matching, never accept answer occurrence in arbitrary reasoning. Store raw text, parsed answer, gold, stop reason and each gate decision. LatCom tests exactly [single_gold_Z; q] and excludes q-only-solvable main cases. Interlat requires the evidence-conditioned text plan to answer correctly, without requiring a frozen receiver to read raw Z first. Both retain a minimum of 32 method-eligible main examples; auxiliary examples cannot satisfy the threshold. All pools share frozen train-only candidates and never use evaluation outcomes.

Use new training_cache_v2 and training_collect_v2 directories; old failures and any weights stay untouched. Resume within v2 validates candidate/data hashes, unique IDs, retained tensor hashes and live eligibility counts. Per-method readiness allows an unavailable LatCom pool to be reported as blocked while Interlat and later model families proceed. A failed branch is never marked complete or assigned an accuracy.

Evaluation methods, sampling, prompts, outputs and scorers stay unchanged. Known 3272ff7 inference code/config identities are reconstructed for the three training-free methods, with exact core inference hashes checked before reusing existing journals. Learned methods use the new run identity and a versioned evaluation path. An independently modified remote code identity is not silently accepted; it produces a clear mismatch and preserves old results.

Default all uses a new orchestration identity. --family explicitly selects 4b or 8b; naming a store *_8b does not select the model. Completed training-free journals are validated and skipped; incomplete ones resume. Independent failed branches are logged and do not prevent later families; the final command exits nonzero if any requested cells remain blocked.

## Disk and validation

The reported 90 GiB free is insufficient headroom for full 8B Adam training checkpoint replacement. Keep model/optimizer checkpoint safety and add a per-stage byte estimate before loading GPU models, rather than lowering the disk gate. No automatic deletion of source models, old results or user data. CPU checks reproduce the original parser failure, strict negative matches, separate gates, saved probes, stage pool selection, resumed caches, immutable inference identities and branch continuation. These checks do not establish B200 retention or convergence.

The report's 0% code results cannot be attributed to sample count alone. Existing raw generations/test_status need inspection on the remote host; code prompts/scorers are intentionally unchanged by this collection repair. Parallel summed task seconds are not isolated-GPU throughput. No new performance claims are made.

## Target-server commands

Update only after previous processes have stopped. Keep the original runtime config (pass --config if it differs from configs/default.json) and LATEN_STORE. After making enough disk space:

```bash
git pull --ff-only
GPU_ID=3 LATEN_STORE=/data/laten_baselines bash run_remaining.sh --family 4b
```

On the second GPU, explicitly choose 8B rather than relying on the store name:

```bash
GPU_ID=5 LATEN_STORE=/data/laten_baselines_8b bash run_remaining.sh --family 8b
```

Alternatively one GPU can run both families with `bash run_remaining.sh`. Do not launch overlapping coordinators against the same store. Two stores sharing one filesystem must budget for both training checkpoint peaks; no per-process check reserves disk against unrelated jobs. Old training_cache is not deleted or rewritten. Re-evaluating rejected candidates in v2 is necessary because old records did not preserve the raw outputs required to re-score them. The script does not silently expand candidates, drop the 32-main threshold, or train random fallback modules.
