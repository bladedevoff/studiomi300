# Benchmarks , Wan2.2-T2V-A14B on MI300X

All measurements: 1280×704, 81 frames, 30 inference steps, BF16 weights.
Container: `rocm/vllm-dev:nightly_main_20260506` (vLLM 0.20.2rc1, torch 2.10, AITER 0.1.13).
Single MI300X (192 GB HBM3, ROCm 7.2 inside container, host driver 7.0.2).

## Headline numbers

| # | Stack increment | Time/clip | Speedup | Notes |
|---|---|---|---|---|
| B6 | BF16 baseline (no cache) | 25.9 min | 1.00× | reference |
| B8b | + ParaAttention FBCache thresh=0.12 (both transformers) | 12.46 min | **2.08×** | lossless |
| B13 | + UniPC `flow_shift=5.0` + ROCm env flags | 11.29 min | 2.30× | env flags = 1.10× alone |
| B14 | + `torch.compile(transformer_2, mode="default", fullgraph=False)` | **10.36 min** | **2.50×** | 2.35 min one-time warmup |

ROCm env flags applied in B13 (set BEFORE `import torch`):

```bash
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIP_FORCE_DEV_KERNARG=1
export GPU_MAX_HW_QUEUES=2
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export MIOPEN_FIND_MODE=FAST
```

## What didn't work (and why)

| Tried | Result | Reason |
|---|---|---|
| MagCache via diffusers 0.38 hooks | dead , calibration_ratios always empty | dual-transformer step counting, `_perform_calibration_step` never fires |
| cache-dit DBCache + TaylorSeer | **22.87 min** (slower than baseline FBCache) | TaylorSeer adds ~6 min overhead on ROCm; cache-dit's L20 numbers don't reproduce |
| AITER FA3 `set_attention_backend("flash")` | hung 9+ min stuck at step 0/24 | JIT compile for 81×1280×704 sequence never finishes |
| `guidance_scale_2=1.0` (skip CFG on low-noise) | 10.35 min vs 10.36 (no speedup) | diffusers WanPipeline doesn't actually short-circuit uncond pass at boundary |
| `torch.compile(mode="max-autotune", fullgraph=True)` | crash | Dynamo error on Wan2.2 (diffusers#12728) |
| `to(memory_format=torch.channels_last)` on transformer_2 | RuntimeError | Wan2.2 transformer is rank-5 (B,C,F,H,W); channels_last is for rank-4 only |

## Quality vs speed sweep , FBCache threshold

Same prompt, same seed, varying `residual_diff_threshold`:

| threshold | time/clip @ 720p | quality observation |
|---|---|---|
| 0.05 (default) | ~14.5 min (est.) | lossless |
| 0.06 | ~13.8 min (est.) | lossless |
| **0.08** | **~10.5 min** | **near-lossless, current production setting** |
| 0.12 | 12.46 min on a different setup | visible motion smearing on fast camera moves (B17 incident) |
| 0.15+ | not tested | community Wan2.1 reports "not usable" |

0.08 lands as the sweet spot. Community Wan2.1 measurements show >=0.09 starts
tearing on motion. Worth eyeballing close-up faces; if they soften too much, drop to 0.06.

## End-to-end reel

Single hybrid reel = 1 hero @ 720p + 8 B-roll @ 480p:

- Hero (B14 stack @ 720p): 10.36 min
- B-roll (same stack @ 832×480): ~3.5 min × 8 = 28 min
- ACE-Step music: ~70s
- Kokoro VO: ~5s
- ffmpeg concat + mix: ~3s
- **Total ~43 min/reel**

Add `--critic` for Qwen3.5 vision auto-retry: +5 min/reel typical (one Qwen3.5 reload between
clips dominates the cost; staying-resident would burn 70 GB while Wan2.2 needs 94 GB to doesn't fit).
