# StudioMI300 incident log

A running journal of failure modes, root causes, and fixes during dev (May 6-10, 2026).
Kept here because half of these don't show up in commit messages.

## reel_v3 (May 6)
Character drift across 9 shots , woman has different hair color in shots 0/4/7,
different eye color in 5.
Root cause: each FLUX call re-rolls the character, even with same seed (different prompts move
the latent enough). Verbatim portrait inlining helped but didn't fully fix it.
Fix: introduce master keyframe + img2img per shot. `flux_master` now hardcoded `seed=7777`.

## reel_v5 (May 7, headless violinist incident)
Clip 7: Wan2.2 generated a second violinist in the alley **without a head**. User noticed it
on first watch ("a third violinist appeared, headless, ahaha").
Root cause: Wan2.2 sometimes hallucinates secondary characters from the prompt's compound
clauses. "Busker plays violin nearby" to it adds another violin-holder, sometimes incomplete.
Fix: added `"two heads, headless, extra people, ghost figures, duplicate character"` to
`NEG_VIDEO`. Hasn't recurred over 12 reels since.

## reel_v6 (May 7, late , woman-with-violin)
Woman ends up holding a violin in clips 4-8 even though prompt says she's just walking past
the busker. The busker is on the right, she's in the middle, but Wan2.2 puts a violin in her hand.
Root cause: master keyframe baked "near violin" into the protagonist embedding because the
master prompt mentioned violin in setting context.
Fix: stripped instrument refs from `master_prompt` v2 (utils.py:flux_master). Now the master
shows protagonist alone, in setting baseline only. Setting context goes via per-shot prompts.
Trade-off: slightly less rich master, but character not contaminated.

## B11 (May 8, FBCache jitter)
FBCache `threshold=0.12` to motion jitter on fast camera pans, especially in B-roll wides.
User feedback: "looks like missing frames in some clips."
Root cause: at threshold ≥0.09, Wan2.1 community had reported tearing on fast motion. Same on Wan2.2.
Fix: backed off to 0.08. 0.05 also tested , lossless but ~10% slower; 0.08 is the sweet spot.

## qwen-tts compatibility nightmare (May 8)
Tried to use Qwen3-TTS-12Hz-0.6B-CustomVoice for VO. Hit four cascading issues:
1. `qwen-tts 0.1.1` hard-pins `transformers==4.57.3` but rest of stack needs `>=5.x`.
2. `check_model_inputs()` decorator API changed to patched out via `sed` on installed package.
3. `Qwen3TTSTalkerConfig` missing `pad_token_id` to manually patched config.json to mirror `codec_pad_id=2148`.
4. `ROPE_INIT_FUNCTIONS["default"]` removed in transformers 5.x to wrote shim returning standard rope init.

Even after all four shims, hit a deeper SDPA shape mismatch in the talker's attention forward.
Gave up after ~1.5 hrs and switched to Kokoro-82M (Apache 2.0, standalone, no transformers dependency).
Kokoro is English-only but Director's VO scripts are English by default , fine.

## ACE-Step circular import (May 8)
`from acestep.pipeline_ace_step import ACEStepPipeline` fails because `pipeline_ace_step.py`
imports `music_vocoder` by absolute name, but the file is at `acestep/music_dcae/music_vocoder.py`.
Fix: `sys.path.insert(0, "/root/ace-step/ACE-Step/acestep/music_dcae")` before the import.
Also `pip install --no-deps` to keep transformers 5.8 (ACE-Step pins 4.50; the pin is too tight).

## Container migration drama (May 7)
Original container `rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210` (Dec 2025) didn't have
`Qwen3_5MoeForConditionalGeneration` registered. Switched to nightly
`rocm/vllm-dev:nightly_main_20260506` (vLLM 0.20.2rc1, torch 2.10).
Side effects: had to reinstall `para_attn`, `diffusers`, `accelerate` with `--no-deps`
to avoid torch version stomp. Also re-cloned ACE-Step. Lost ~30 min on env setup.

## FP8 evaluation (May 9, defer)
Tried two FP8 paths on `rocm/vllm-dev:nightly_main_20260506` (torch 2.10.0+git8514f05,
ROCm 7.0):

1. `torchao.quantization.Float8DynamicActivationFloat8WeightConfig` , Float8Tensor weights
   are created but `torch._scaled_mm` raises `HIPBLAS_STATUS_NOT_SUPPORTED`. The Python
   path falls through to a BF16-equivalent emulation. Smoke on a 480p clip: 5.45 min FP8
   vs 5.51 min BF16 (1.01x, within noise). VRAM drops 79.0 -> 71.6 GB which is real but
   uninteresting on a 192 GB MI300X.

2. `aiter.gemm_a8w8` and `aiter.gemm_a8w8_CK` , benchmarked at 1.17-1.42x BF16 on dense
   Wan2.2 shapes (FFN, QKV, out_proj at M >= 4096). But BOTH variants segfault with
   "Memory access fault by GPU node-1" on the cross-attention shape **M=512, N=5120, K=4096**
   that appears in Wan2.2 transformer blocks. Default config (no tuned entry in
   `a8w8_tuned_gemm.csv`) doesn't handle that shape.

Workaround would be running `aiter.gemm_a8w8_tune` ahead of time for every Wan2.2 shape,
~30 min one-time sweep. Deferred post-submission.

Update on `rocm/vllm-openai-rocm:v0.17.1` (ROCm 7.2 / torch 2.9.1 / aiter shipped):
the standalone `aiter.gemm_a8w8_CK` call on (M=512, N=5120, K=4096) now succeeds.
ROCm 7.2 closed that specific shape segfault. Win recorded for the AITER team.

But inside the full Wan2.2 + FBCache + torch.compile pipeline, the same call
crashes with "Memory access fault by GPU node-1" after a sequence of shapes
including M=75600, K=5120, N=5120 (the high-noise expert batched cross-attention).
This matches the multi-shape pattern in ROCm/aiter#2187 (FP8 MoE crashing on
specific cudagraph capture sizes, mitigated only by disabling AITER MoE).
Disabling torch.compile and using `gemm_a8w8` non-CK didn't help. Pre-tune sweep
across all observed shapes is the next step (~50 min one-time cost), deferred.

Production shipped on BF16 + FBCache + selective compile.

`aiter_linear.py` and `STUDIOMI_AITER_FP8=1` env-toggle stay in the repo for future
experiments. Production stack remains BF16 + FBCache + selective `torch.compile`
(measured 2.5x lossless).

## known limitations (still open)
- Wan2.2 sometimes adds a 2nd pedestrian in Tokyo-style night scenes. Critic catches ~70%
  but not all (compound prompts confuse the critic too sometimes).
- Kokoro English-only , Arabic / Chinese / Russian VO is on the wishlist (Bark is the
  fallback, RTF is worse but multilingual).
- ACE-Step occasionally clips at the end (last 0.5s). ffmpeg `-shortest` masks it.
- I2V keyframe locks the character pose more than ideal. Bumping `strength` to 0.7
  helps but increases drift. Better fix would be Wan2.2-FLF (first/last frame) variant
  if it ships in time.
- LoRA training (Sharon Zhou's r=4 alpha=4 pattern) deferred to post-submission , would
  fix character drift completely but costs 2 hrs/character of GPU time. Not feasible
  for a "type any prompt" UI.
