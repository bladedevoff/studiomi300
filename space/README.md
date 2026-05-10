---
title: StudioMI300
emoji: 🎬
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: true
license: mit
short_description: One prompt → 30s cinematic reel on a single AMD MI300X
thumbnail: thumbnail.png
tags:
  - amd
  - amd-hackathon-2026
  - mi300x
  - rocm
  - video-generation
  - wan2.2
  - flux
  - qwen
  - text-to-video
  - text-to-film
  - cinematic
  - gradio
---

# StudioMI300

**One prompt → 30-second cinematic reel.** Built for the AMD Developer Hackathon 2026
on a single AMD Instinct MI300X (192 GB HBM3, ROCm 7.2).

## What it does

You write one sentence. The pipeline plans a six-shot story, paints character
keyframes, animates them, scores the music, narrates the voice-over, and stitches
everything into a 30-second `mp4`. No setup. No LoRA training. No per-shot prompting.

```
"A young woman walks through neon-lit Tokyo at night and meets two friends."
                                  ↓
                 [ ~45 minutes on a single MI300X ]
                                  ↓
                    30s cinematic reel.mp4 + audio
```

## How it works (single MI300X, sequential)

1. **Director Agent** — Qwen3.5-35B-A3B (BF16, vLLM, AITER MoE) plans 6 shots,
   character portraits, music brief, VO script, language tag.
2. **Per-shot keyframes** — FLUX.2 [klein] 4B reference editing seeds each
   shot from a single canonical character master, pinning identity.
3. **Animation** — Wan2.2-I2V-A14B with ParaAttention FBCache (2× lossless)
   and selective `torch.compile` on `transformer_2` (1.2× compile win).
4. **Vision Critic** — the same Qwen3.5 looks at four sampled frames per clip,
   labels failure modes (`STYLIZED_AI_LOOK`, `CHARACTER_DRIFT`, `EXTRAS_INVADE_FRAME`...)
   and triggers a re-render with a bumped seed if the score is below threshold.
5. **Music** — ACE-Step v1 3.5B generates a 30-second instrumental from the
   Director's music brief.
6. **Voice-over** — Kokoro-82M narrates the Director's script in any of 9
   languages (Director picks the language to match the setting).
7. **Mix** — `ffmpeg` concat-and-loudnorm into the final `mp4`.

## The full open-source stack (Apache 2.0 / MIT throughout)

| Stage | Model | License |
|---|---|---|
| Planner / Critic | Qwen3.5-35B-A3B | Apache 2.0 |
| Image | FLUX.2 [klein] 4B | Apache 2.0 |
| Video | Wan2.2-I2V-A14B | Apache 2.0 |
| Music | ACE-Step v1 3.5B | Apache 2.0 |
| TTS | Kokoro-82M | Apache 2.0 |
| Serving | vLLM 0.17 | Apache 2.0 |
| Caching | ParaAttention FBCache | Apache 2.0 |
| AMD kernels | AITER 0.1.13 | MIT |
| Project code | StudioMI300 | MIT |

Every output you generate from this stack is yours to use commercially.

## Why a single MI300X

Most cinematic generation pipelines assume you have a multi-GPU cluster: one GPU
for the planner, one for the image model, one for the video model, etc. On 192 GB
HBM3 the pipeline runs them all sequentially on the same card. That's the project's central
constraint and also its main flex:

- Qwen3.5-35B planner loads / unloads cleanly between Director and Critic phases.
- Wan2.2-I2V-A14B (~80 GB BF16) leaves headroom for FLUX.2 [klein] 4B (~8 GB)
  and ACE-Step (~12 GB) to live alongside in subprocess scope.
- AITER MoE for the planner. AITER FA / FP8 was evaluated for Wan2.2 — results
  documented in `incidents.md` of the GitHub repo (FP8 path crashes mid-pipeline
  on ROCm 7.2, AITER/issues#2187, BF16 ships).

## Live demo

This Space hosts the showcase. Live generation requires an MI300X (45 minutes
per reel is too long for a casual visitor anyway). The full pipeline is on
GitHub — clone, point it at your MI300X, and it generates.

## Credits

AMD Developer Hackathon 2026 entry. Built solo over six days on one AMD
Developer Cloud MI300X droplet.

Made with the open-source ecosystem: Black Forest Labs, Wan-AI, Alibaba Qwen,
StepFun, hexgrad/Kokoro, vLLM, ParaAttention, diffusers, AMD ROCm + AITER.
