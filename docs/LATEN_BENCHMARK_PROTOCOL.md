# LATEN semantic Benchmark extension

> Current execution revision (2026-09-09): [SAMPLED_PARALLEL_PROTOCOL.md](SAMPLED_PARALLEL_PROTOCOL.md) controls evaluation scheduling and scope. General tasks use fixed ~10% subsets (591 total); LATEN remains full 648/486. Default inference concurrency is two, training stays serial. There are 80 cells / 12,390 evaluated tasks. Older full-general counts and serial timing descriptions below document the source protocol, not the current default execution. New paths: evaluation_sample10 and semantic_parallel.

## Source and benchmark audit

This is an evaluation integration, not a new benchmark or new training objective. The immutable V2.2 data artifacts contain V2.1.0 items: 162 cases, 648 BASE/CF variants, 486 matched BASE/CF pairs; dev 108 and previously exposed test 540 variants. The prompt source is V2.3 deliberative readout. The latest project channel evaluation authorizes 2048 greedy reasoning tokens, then constrained comma-separated I3/I6/I9 bits. The prompt's historical soft request for fewer than 384 reasoning tokens is preserved verbatim. Original file hashes and scoring source provenance are in `vendor/laten/data/manifest.json` and `evidence/laten_sources.json`. No TaMAS regeneration or external API judge is needed.

Producer prompts, ownership, role order (G4/G5/G6), private fact values, public schema, policy and gold remain unchanged. Only owners see realized facts in their own prompts. Receiver/compressor public queries contain definitions and questions, never the realized gold. Public action policy is used only by the downstream action scorer. Evaluation does not train on these 648 variants. Both learned methods reuse their completed general-corpus training exports, including compatible original-release exports.

## Baseline adaptations to the serial role graph

- LatentMAS: cumulative KV with 10 actual realigned latent inputs per producer.
- LatentMAS-H2O: same serial KV, headwise prompt-attention retention plus all retained history and actual latent positions; absolute rotary positions continue across pruning.
- LatentMAS-Hidden: fresh per-role cache; incoming message is all upstream original prompt input embeddings plus only the latest producer's 10 aligned latents. This is the existing channel-v2 module-off carrier protocol. Old latents are not accumulated.
- LatCom: each receiving role obtains 64 slots from the trained compressor applied to the preceding producer's 10-step trajectory and the recipient's public context; the frozen recipient produces a new 10-step trajectory. No hidden KV bypass crosses these edges.
- Interlat: each producer uses the trained autoregressive compressor (21 hidden states). Each receiving compressed Sender conditions on the preceding states through the trained receiver adapter; the final receiver uses its trained full LM and adapter. This explicit multi-hop port extends compressed-Sender input with an incoming message; intermediate compressor weights are not mislabeled as the full receiver LM. No earlier states or prompt embeddings bypass the learned channel.

The latent rollout depth 10 follows the project's matched Benchmark setting, whereas the seven general datasets retain 40. LatCom's 64 slots and Interlat's 21 learned recurrence steps match their trained modules, so this is not an equal-byte budget experiment and no compression-rate advantage is presumed. These serial-graph learned-method adaptations are not claimed to be author-implemented or trained multi-hop baselines. All five methods are evaluated, and receive identical task prompts/decoding and metric definitions, while preserving method-specific carriers.

## Metrics and failure rules

Reuse the original deterministic policy evaluator, native reasoning/bit decoding rules, original V6 summary counts/pairing and efficiency aggregator. Report vector exact, bit accuracy, deterministic functional action exact, frozen-base A-H model-action diagnostic, decoding success and cap count, three CF pair metrics, and split/domain/G/I/reasoning-level breakdowns. Add P(fact error | correct functional action) using the same denominator definition. A wrong or capped answer counts in the full fixed denominator. Infrastructure errors stop and remain unscored/pending, never become synthetic model errors.

Time the entire warm producer-to-final-vector path including policy evaluation and provenance; frozen A-H model-action scoring stays outside it. Retain synchronized per-task seconds and mean/p50/p95 as contention diagnostics; summed overlapping seconds are accumulated task time, not stage wall time. Do not report inverse accumulated time as real parallel throughput; count selected reasoning/bit tokens separately from forced delimiters, latent positions, prefill and logical communication bytes. KV payloads preserve separate prompt/latent counts after pruning. Both sent and received counts cover every serial edge. LatCom also reports compressor prefill cost. Byte volume is logical payload size, not measured network traffic.

The LATEN table is separate from the seven-dataset accuracy macro. Its 10 model/method cells bring the current run to 80 cells (12,390 task attempts including the sampled general tasks). No clean unseen-test claim is supported by the previously exposed project test split.

## Feasibility and verification gate

Remote single-B200 execution only, with two inference workers and serial training; no current-server model download/training or benchmark scores. CPU checks must cover all frozen hashes, 648/486 denominators, original prompt and metric parity, gold-policy consistency, private visibility, partial/failed pairs, all five G4/G5/G6 communication paths, and native close/EOS/cap handling before upload. Tests with injected gold or tiny random models are explicitly engineering fixtures, not performance evidence.
