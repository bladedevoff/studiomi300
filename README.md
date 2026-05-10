# studiomi300

One prompt → 30s cinematic reel. End-to-end on a single AMD Instinct MI300X.
Built solo for the AMD Developer Hackathon, May 2026.

## what it does

```
python generate.py --prompt "a young woman walks through neon-lit Tokyo at night and meets two friends" --out outputs/demo --critic
```

~45 minutes later you get `outputs/demo/reel_final.mp4` - six 5-second shots,
character-consistent, with music and per-shot voice-over, mixed.

## how it works

Eight stages, all on the same GPU:

1. **Director** - `Qwen3.5-35B-A3B` via vLLM. Plans 6 shots, character portraits,
   music brief, per-shot VO script, and the language to narrate in.
2. **Masters** - `FLUX.2 [klein] 4B` text-to-image, one canonical frame per character.
3. **Per-shot keyframes** - same klein, reference editing, conditioned on the master.
4. **Animation** - `Wan2.2-I2V-A14B` with FBCache (lossless 2x) + selective
   `torch.compile`. FLF2V mode on `cut: false` continuation arcs locks identity at
   both ends.
5. **Vision critic** - Qwen3.5 re-loads, scores 4 frames per clip with structured
   labels (`STYLIZED_AI_LOOK`, `CHARACTER_DRIFT`, `CAMERA_IGNORED`, ...). Bumps
   seed and re-renders if `overall < 7`. Up to 3 attempts.
6. **Music** - `ACE-Step v1` 3.5B, 30s instrumental from the brief.
7. **Voice-over** - `Kokoro-82M`, 9 languages, one wav per shot, ffmpeg `adelay`s
   them onto the music bed at clip-start offsets.
8. **Mix** - `ffmpeg` concat + lanczos upscale + loudnorm.

The Director also doubles as the vision critic. Same checkpoint, two roles.

## stack

| Stage | Model | License |
|---|---|---|
| Planner / critic | Qwen3.5-35B-A3B | Apache 2.0 |
| Image | FLUX.2 [klein] 4B | Apache 2.0 |
| Video | Wan2.2-I2V-A14B | Apache 2.0 |
| Music | ACE-Step v1 3.5B | Apache 2.0 |
| TTS | Kokoro-82M | Apache 2.0 |
| Serving | vLLM 0.17 | Apache 2.0 |
| Cache | ParaAttention FBCache | Apache 2.0 |
| AMD kernels | AITER | MIT |

Outputs are commercially usable. No NC weights anywhere.

## why a single MI300X

192 GB HBM3 lets four very different architectures share one card sequentially.
On 24 GB consumer hardware you'd need 4-5 separate machines.

| Phase | Peak VRAM |
|---|---|
| Director (Qwen3.5-35B BF16) | ~70 GB |
| FLUX.2 klein 4B | ~8 GB |
| Wan2.2-I2V-A14B | ~94 GB |
| Critic (Qwen3.5 reload) | ~70 GB |
| ACE-Step v1 | ~12 GB |
| Kokoro-82M | <1 GB |

Each phase unloads cleanly via `gc.collect()` + `torch.cuda.empty_cache()`
before the next one loads. Director runs in-process for planning then `del`s
itself before Wan2.2 loads, otherwise OOM.

## run it

Tested inside `rocm/vllm-dev:nightly_main_20260506` (vLLM 0.20.2rc1, torch 2.10,
ROCm 7.2 in-container) on AMD Developer Cloud.

```bash
# inside the rocm container
pip install -r requirements.txt --no-deps
export STUDIOMI_AITER_FP8=0
export VLLM_ROCM_USE_AITER=1
python generate.py --prompt "your reel idea" --out outputs/myreel --critic
```

ROCm env is set in `generate.py` before any torch import - `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`,
`TORCH_BLAS_PREFER_HIPBLASLT=1`, `MIOPEN_FIND_MODE=FAST`, `GPU_MAX_HW_QUEUES=2`,
`HIP_FORCE_DEV_KERNARG=1`, `HSA_ENABLE_SDMA=0`. If you change those after import
you get a silent allocator-profile mismatch and lose ~3 GB.

### multi-GPU

Stage routing via env vars (default `cuda:0` for everything):

```bash
STUDIOMI_GPU_FLUX=cuda:1 STUDIOMI_GPU_WAN=cuda:0 python generate.py ...
```

Tested only on single-MI300X. Hooks are there for 2+ card setups.

### API server

```bash
STUDIO_API_TOKEN=secret uvicorn server:app --host 0.0.0.0 --port 8000
```

`POST /jobs` with `{"prompt": "...", "use_critic": true}` returns `{"job_id"}`.
`GET /jobs/{id}/stream` is an SSE feed of every stage event (plan, masters,
keyframes, clip render, critic verdicts, music, VO, final). Granular artifact
endpoints under `/jobs/{id}/{plan,master,keyframe,clip,music,vo,video}`.

## what's optimised

| Knob | Speedup | Note |
|---|---|---|
| ParaAttention FBCache (threshold 0.05) | 2.00x | lossless |
| `torch.compile(transformer_2, mode="default")` | 1.20x | 2.35 min one-time warmup |
| ROCm env flags | 1.10x | hipBLASLt, expandable_segments, MIOpen FAST mode |
| flow_shift=5 hero / 8 b-roll | quality | upstream wan_i2v_A14B.py default; I used 12 first and got plastic skin |
| FLUX.2 klein 4B vs FLUX.1-schnell | ~15x | sub-second keyframes |

End-to-end: 25.9 min → 10.4 min per 720p clip on a single MI300X.

## what's not used (and why)

- **AITER FP8 on Wan2.2** - `gemm_a8w8_CK` segfaults on the cross-attn shape
  (M=512, K=4096, N=5120) on ROCm 7.0; closed standalone on 7.2 but still crashes
  inside the full pipeline graph (matches `ROCm/aiter#2187`). Code stays behind
  `STUDIOMI_AITER_FP8=1` env flag; production ships BF16.
- **MagCache** - diffusers 0.38 calibration-step counter doesn't fire on Wan2.2's
  dual-transformer schedule.
- **cache-dit + TaylorSeer** - slower than baseline FBCache on ROCm.
- **Wan2.2-Lightning I2V LoRA** - V1 only (Aug 2025), V2.0 phased-DMD never landed
  for I2V; quality drop on hero shots vs full-step.
- **AITER FA3** - JIT compile for 81×1280×704 attention never finishes.
- **`torch.compile(mode="max-autotune", fullgraph=True)`** - Dynamo error on
  Wan2.2 (diffusers#12728).
- **`channels_last`** - Wan2.2 transformer is rank-5; channels_last is rank-4 only.

## files

```
generate.py        cli entry, runs the pipeline
director.py        director agent + vision critic (one Qwen3.5-35B serves both)
utils.py           pipeline core: masters, keyframes, render_clips, music, vo, mix
aiter_linear.py    fp8 nn.Linear drop-in for wan2.2 transformer (off by default)
events.py          stage event emit (jsonl + EVENT:: stdout marker)
server.py          fastapi wrapper with sse + per-artifact endpoints
app.py             gradio app: showcase + stub generate
incidents.md       running journal of failures, root causes, fixes
benchmarks/        wan2.2 speedup table on mi300x
space/             slim showcase-only gradio app for huggingface space
```

`incidents.md` is honest. Headless violinist and AITER segfault are in there.

## license

MIT for project code. All upstream models are Apache 2.0 / MIT - see the stack table.
