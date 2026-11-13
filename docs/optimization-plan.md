# Inkling 2 Small optimization campaign

Baseline (frozen, `performance` branch evidence/): 16/16 API gate,
c1 80 / c8 560 / c16 974 / c32 2138 output tok/s (1024/256 random),
natural oracle sweep complete with all exact-JSON answers correct,
per-request decode 70–85 tok/s, TTFT p50 ≤ 730ms through c32.
Engine: vLLM 0.1.dev25249+gb974b306f, TP4 B300, CC on, no speculation.

## Rules (inherited from kimi-k31-cc-lab and deepseek-flash-cc-lab)

- One controlled change per experiment; measure with the three instruments
  (scripts/run-qualification.sh, run-bench-curve.sh, run-natural.sh) against
  the frozen baseline. Keep rejected trials in the ledger below.
- Engine replacement only via scripts/switch-engine.sh (graceful 60s stop,
  120s GPU-retirement gate, restart=no). A crashed engine load poisons the
  guest NVIDIA driver (D-state, unkillable) → full CVM recovery (~25 min).
- Score with CC on and DevTools off. Never run CUDA diagnostics while a
  serving model owns GPU contexts.

## Phases

0. Recover baseline CVM after any poisoning event (automated redeploy+patch).
1. Upstream recon: our vLLM is months newer than the labs' 0.31. For each
   lab patch class, check whether it is upstreamed, renamed, or regressed:
   UVA/device mirrors (VLLM_DISABLE_UVA is an UNKNOWN env var in this build —
   find the successor), sampling dirty-refresh, pageable non-blocking H2D,
   metadata mirrors, consumer readback, text staging (deepseek exp 028/035:
   improved every concurrency point). Also audit new defaults that may hurt
   CC (NVLS/multicast paths, autotune defaults); deepseek exp 022 showed a
   bare upgrade LOSES throughput at c32.
2. Cheap flag screens: (a) enable legacy custom all-reduce (non-multicast;
   deepseek exp 069 qualified the component on identical CC B300s, exp 074
   rejected at c32 — model-specific, rescreen here); (b) CUDA-graph capture
   coverage around the c16 per-request-decode dip (kimi step 3); (c) one
   max-num-batched-tokens screen (negative prior: deepseek exp 113/117).
3. Reasonable revisits of deepseek rejections: fix flashinfer globaltimer
   (missing cusparse.h headers in image) → requalify native timer → screen
   autotune ON (deepseek exp 061/084 rejected with a broken timer); screen a
   compressed NVFP4 attention cache analog if this build exposes one
   (deepseek exp 110: +4.98%, narrowly missed guard). Skip: CPU/weight
   offload (UVA-dependent, we have VRAM headroom), frontend/logprob tricks.
4. DSpark CC fallback (separate project): Inkling2 DSpark hard-requires
   NVLink multicast for its cross-rank draft buffer
   (vllm/models/inkling2/ops/dspark.py: `assert is_multicast_supported`,
   world_size>1 branch); multicast is unavailable under CC (confirmed here
   and in deepseek exp 050). Fallback = per-rank symmetric buffers +
   explicit exchange. Prize per deepseek static-5: +46.7% c1 / +83.6% c8.
   Note: num_speculative_tokens must be 7 (checkpoint block_size 8 − 1).

## Ledger

| # | change | disposition |
| --- | --- | --- |
| 1 | DSpark depth 5 (runai loader) | blocked: runai loader lacks index filtering / allow_patterns_overrides / local ignore_patterns → fuse_weight; upstream report due |
| 2 | DSpark depth 5 (default loader) | blocked: depth must be 7 (block_size − 1) |
| 3 | DSpark depth 7 (default loader) | blocked under CC: `assert is_multicast_supported` in draft buffer (ops/dspark.py:277); needs CC fallback (phase 4) |

## Known platform quirks (report upstream / to infra)

- Engine crash mid-CUDA-load poisons guest NVIDIA driver (D-state ioctl,
  unkillable containers, docker exec setns failures). Recovery: replace CVM.
- First boot after CVM teardown fails GPU attestation ("NVIDIA bootstrap
  reported failure") ~always; immediate redeploy succeeds.
- flashinfer globaltimer JIT cannot build: image lacks cusparse.h (same as
  kimi inf22). CUDA-event fallback active; autotuning must stay off until
  repaired and requalified.
- vLLM runai streamer loader ignores the safetensors index,
  allow_patterns_overrides, and ignore_patterns for local paths (three-part
  upstream bug; default loader handles all three).
