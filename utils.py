# utils, every model load + render helper used by generate.py.
# kept flat (no submodules) because hackathon code shouldn't pretend to be a library.
import os
import sys
import gc
import json
import time
import logging
import subprocess
from pathlib import Path

log = logging.getLogger("studiomi300.utils")

# multi-GPU routing: each stage can pin to its own device via env vars.
# all default to cuda:0 -> single-MI300X behaviour unchanged. on a 2x rig
# Wan2.2 can sit on one card while FLUX/ACE run on another in parallel.
GPU_FLUX = os.environ.get("STUDIOMI_GPU_FLUX", "cuda:0")
GPU_WAN  = os.environ.get("STUDIOMI_GPU_WAN",  "cuda:0")
GPU_ACE  = os.environ.get("STUDIOMI_GPU_ACE",  "cuda:0")
GPU_TTS  = os.environ.get("STUDIOMI_GPU_TTS",  "cuda:0")

try:
    import events as _events
except ImportError:
    class _events:
        @staticmethod
        def emit(*a, **kw): pass


# verbatim trained Chinese negative from shared_config.py (umT5 was tuned against
# these tokens, English is weaker) + photoreal anti-style additive for "AI look"
NEG_VIDEO = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    "，plastic skin, smooth skin, airbrushed skin, perfect skin, beauty filter, "
    "retouched, oversaturated, neon bloom, AI artifact, uncanny valley, waxy face, "
    "melted face, 3d render, CGI, cartoon, anime, illustration, digital painting, "
    "overprocessed, HDR halo, sharpening halo, posterization, color banding"
)

# critic threshold. picked 7/10 after eyeballing five reels in a row - 6 lets
# obvious face drift through, 8 rejects everything and the retry loop bottoms out.
PASS_THRESHOLD = 7


def _free_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _build_wan_i2v_pipe():
    # Wan2.2-I2V-A14B + FBCache + selective torch.compile.
    # diffusers config ships boundary_ratio=0.9 (matches upstream wan_i2v_A14B.py),
    # so the LAST 90% of steps go through transformer_2; both still need patching.
    # channels_last is broken on Wan2.2 (rank-5 tensor), skip it.
    # STUDIOMI_AITER_FP8=1 / STUDIOMI_FP8=1 toggle the FP8 paths; incidents.md
    # explains why neither is the default on ROCm 7.0 + torch 2.10.
    import torch
    from diffusers import WanImageToVideoPipeline, AutoencoderKLWan, UniPCMultistepScheduler
    from para_attn.first_block_cache.diffusers_adapters import (
        apply_cache_on_pipe, apply_cache_on_transformer,
    )

    mid = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    log.info(f"loading {mid}")
    vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(mid, vae=vae, torch_dtype=torch.bfloat16).to(GPU_WAN)

    if os.environ.get("STUDIOMI_AITER_FP8") == "1":
        # AITER gemm_a8w8_CK path. Validated 1.17-1.42x vs torch BF16 matmul on
        # Wan2.2-shaped matrices (scripts/fp8_probe.py). Chosen over torchao because
        # torch._scaled_mm hits HIPBLAS_STATUS_NOT_SUPPORTED on ROCm 7.0 + torch 2.10.
        from aiter_linear import patch_linears_to_fp8
        t = time.perf_counter()
        n1 = patch_linears_to_fp8(pipe.transformer)
        n2 = patch_linears_to_fp8(pipe.transformer_2)
        torch.cuda.synchronize()
        log.info(f"AITER FP8 patched {n1+n2} Linear layers in {time.perf_counter()-t:.1f}s")
    elif os.environ.get("STUDIOMI_FP8") == "1":
        # torchao stock path. Confirmed inactive on ROCm 7.0 + torch 2.10 (no cpp ext,
        # _scaled_mm not supported in hipBLASLt). Kept as a no-cost reference.
        from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
        cfg = Float8DynamicActivationFloat8WeightConfig()
        t = time.perf_counter()
        quantize_(pipe.transformer, cfg)
        quantize_(pipe.transformer_2, cfg)
        torch.cuda.synchronize()
        log.info(f"torchao FP8 quantize_: {time.perf_counter()-t:.1f}s")

    # FBCache 0.05 = lossless (0.12 jitters fast motion per B11; 0.08 had choppy
    # cuts on v8). Apply AFTER any quantization so the cache hooks the final
    # output of each block, not the intermediate.
    apply_cache_on_pipe(pipe, residual_diff_threshold=0.05)
    apply_cache_on_transformer(pipe.transformer_2)

    # torch.compile only when AITER FP8 is OFF. With FP8 patched Linears the
    # compiled graph drives the JIT-loaded gemm_a8w8 op into shape configs the
    # default kernel can't handle, and you get the multi-shape memory access
    # fault from ROCm/aiter#2187. BF16 path keeps the 2.5x compile win.
    if os.environ.get("STUDIOMI_AITER_FP8") != "1":
        pipe.transformer_2 = torch.compile(
            pipe.transformer_2, mode="default", fullgraph=False, dynamic=False
        )
    else:
        log.info("torch.compile skipped (AITER FP8 active, see ROCm/aiter#2187)")
    return pipe


# keyframe strategy: anchor masters + per-shot Img2Img.
# v7-v9 used one woman-only master and Img2Img couldn't add the dog/busker on its
# own. v10 went pure text2img per shot but characters re-rolled identity between
# cuts. v11 splits the difference: cache one master per character + a "group"
# master that has everyone together, then Img2Img every shot off the matching
# anchor at strength=0.5 so faces stay locked across the reel.

def flux_masters(plan, outdir, *, seed=7777):
    # one master per character + master_ABC for multi-char shots + master_scene
    # for inserts. Group master is the trick: it gives multi-char shots a base
    # that already contains all three faces, so editing just rearranges them
    # rather than inventing new ones each time.
    import torch
    from diffusers import Flux2KleinPipeline

    chars = plan.get("characters") or {}
    logline = plan.get("story_logline", "")

    targets = {}  # name -> (prompt, w, h)
    for k, portrait in chars.items():
        targets[k] = (
            f"{portrait}. Portrait shot, neutral standing pose, looking ahead, "
            f"the subject fills the centre. Setting: {logline}. "
            f"shot on Arri Alexa, anamorphic, 35mm film grain, photorealistic.",
            832, 480,
        )
    if len(chars) >= 2:
        # group composition. Spell out spatial layout; klein 4B handles 3
        # subjects at 832x480 well enough as long as the prompt is explicit.
        keys = sorted(chars.keys())
        slots = ["LEFT", "CENTER", "RIGHT", "FAR-RIGHT"]
        slot_lines = [f"{slots[i]} third: {chars[k]}" for i, k in enumerate(keys[:4])]
        targets["ABC"] = (
            "Group photograph, wide framing. " + " ".join(slot_lines)
            + f" Setting: {logline}. All faces visible. "
            f"shot on Arri Alexa, anamorphic, 35mm film grain, photorealistic.",
            832, 480,
        )
    targets["scene"] = (
        f"Establishing background plate. Setting: {logline}. "
        f"No people in frame. shot on Arri Alexa, anamorphic, 35mm film grain, photorealistic.",
        832, 480,
    )

    todo = []
    masters = {}
    for name, (prompt, w, h) in targets.items():
        path = Path(outdir) / f"master_{name}.png"
        if path.exists():
            log.info(f"master_{name} exists, reusing")
            masters[name] = str(path)
        else:
            todo.append((name, path, prompt, w, h))

    if not todo:
        return masters

    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16
    ).to(GPU_FLUX)
    for offset, (name, path, prompt, w, h) in enumerate(todo):
        log.info(f"master_{name}: {prompt[:90]}...")
        t = time.perf_counter()
        # klein 4B is step-distilled, 4 steps is the sweet spot. guidance_scale
        # is silently ignored on distilled klein (see "Guidance scale 4.0 is
        # ignored for step-wise distilled models" warning).
        out = pipe(
            prompt=prompt, width=w, height=h, num_inference_steps=4,
            generator=torch.Generator(device=GPU_FLUX).manual_seed(seed + offset * 11),
        )
        out.images[0].save(path)
        log.info(f"master_{name} done in {time.perf_counter()-t:.1f}s")
        masters[name] = str(path)
        _events.emit("master_ready", name=name, path=str(path), seconds=round(time.perf_counter()-t, 2))
    del pipe
    _free_gpu()
    return masters


def _which_chars_in_shot(shot, chars_map):
    # match via the first few portrait words against the shot prompt. Director v4
    # writes "the young woman with chestnut hair..." style refs so the first 3-5
    # words of each portrait usually appear verbatim in the shot prompt.
    text = shot.get("prompt", "").lower()
    found = []
    for k, portrait in chars_map.items():
        head = " ".join(portrait.split()[:6]).lower()
        if head and (head[:30] in text or any(
            w in text for w in head.split()[1:5] if len(w) > 4
        )):
            found.append(k)
    return found


def _build_keyframe_prompt(shot, plan):
    # append full portraits for every character mentioned in the shot, so FLUX
    # has enough detail to paint all of them. Director's shot prompt itself
    # only carries short refs to keep the Wan2.2 prompt budget under 120 words.
    chars = plan.get("characters") or {}
    base = shot["prompt"].replace(", photorealistic, 24fps cinematography", "").strip()
    referenced = _which_chars_in_shot(shot, chars) or [shot.get("dominant_subject", "A")]
    referenced = [k for k in referenced if k in chars]
    if referenced:
        portraits_block = " Characters: " + " | ".join(
            f"{k}: {chars[k]}" for k in referenced
        )
        base = base + portraits_block
    return base


def _build_flux_pipes():
    # FLUX.2 [klein] 4B is a single distilled pipeline that does both text2img
    # (when image=None) and reference editing (when image is a PIL.Image).
    # Apache 2.0, ~8 GB VRAM, 4 steps, sub-second per call after warmup.
    # Replaced FLUX.1-schnell + FluxImg2ImgPipeline because klein editing is
    # trained for the editing task and preserves identity better than schnell's
    # noise-injection Img2Img (verified via smoke test on master_A).
    import torch
    from diffusers import Flux2KleinPipeline
    klein = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16
    ).to(GPU_FLUX)
    # return same pipe twice so call sites that distinguish text/img stay the
    # same; the difference is whether the `image=` arg is passed.
    return klein, klein


def _flux_text2img_keyframe(txt_pipe, prompt, w, h, *, seed):
    import torch
    out = txt_pipe(
        prompt=prompt, width=w, height=h, num_inference_steps=4,
        generator=torch.Generator(device=GPU_FLUX).manual_seed(seed),
    )
    return out.images[0]


def _flux_chained_keyframe(img_pipe, base_path, prompt, w, h, *, seed, strength=0.45):
    # klein 4B reference editing. `strength` is unused (klein decides edit
    # magnitude internally from the prompt), kept for call-site compatibility.
    import torch
    from PIL import Image
    ref = Image.open(base_path).convert("RGB").resize((w, h))
    out = img_pipe(
        image=ref, prompt=prompt, width=w, height=h, num_inference_steps=4,
        generator=torch.Generator(device=GPU_FLUX).manual_seed(seed),
    )
    return out.images[0]


def _save_last_frame(frames, path):
    # last frame is the next shot's chained Img2Img base when chained mode kicks in
    from PIL import Image as PIL_Image
    import numpy as _np
    frame = frames[-1]
    if not isinstance(frame, PIL_Image.Image):
        arr = _np.asarray(frame)
        if arr.dtype.kind == "f":
            arr = (_np.clip(arr, 0, 1) * 255).astype(_np.uint8)
        frame = PIL_Image.fromarray(arr)
    frame.save(path)
    return str(path)


def _pick_anchor(shot, plan, masters):
    # pick the master image to use as Img2Img base. For shots with >=2 characters
    # in frame use the group master so faces are sourced from one canonical
    # composition; for solo shots use the single-character master.
    chars = plan.get("characters") or {}
    refs = _which_chars_in_shot(shot, chars)
    refs = [k for k in refs if k in chars]
    dom = shot.get("dominant_subject", "A")
    if len(refs) >= 2 and "ABC" in masters:
        return masters["ABC"], "ABC"
    if dom in masters:
        return masters[dom], dom
    if dom == "scene" and "scene" in masters:
        return masters["scene"], "scene"
    return next(iter(masters.values())), "fallback"


def render_clips(plan, outdir, *, use_critic=False):
    # main loop. keyframes are FLUX Img2Img on a per-shot anchor master (single
    # char or ABC group). Continuations with same dominant fall back to chained
    # Img2Img off the previous clip's last frame.
    import torch
    from PIL import Image
    from diffusers import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    masters = flux_masters(plan, outdir)
    # 6-shot plan -> 5.04 sec/clip (121 frames @ 24fps), 9-shot plan -> 3.375 sec
    # (81 frames). Wan2.2 num_frames must be 4k+1.
    n_shots = len(plan["shots"])
    num_frames = 121 if n_shots == 6 else 81

    flux_txt, flux_img = _build_flux_pipes()
    pipe = _build_wan_i2v_pipe()

    chars = plan.get("characters") or {}
    base_seed = 5000
    flux_seed = 8888

    out_paths = []
    prev_last_frame_path = None  # filled after each clip for chained transitions
    prev_dominant = None         # which character anchored the previous clip

    for shot in plan["shots"]:
        i = shot["index"]
        # Uniform 832x480 + 18 steps + flow_shift=12.0 (vLLM-Omni recipe).
        # The hero used to render at 1280x720 / 24 steps. On a 6-shot plan
        # that doubles the slowest clip's wall time without a clear win:
        # mix_reel already upscales every clip to 1280x704 with lanczos.
        # The hero shot is still wide / establishing in the prompt, just at
        # B-roll resolution.
        # shift=5 hero (upstream wan_i2v_A14B.py) preserves detail; shift=8 b-roll
        # (SwarmUI default) favours motion. v17/v20 used 12.0 -> plastic skin.
        w, h, n_steps = 832, 480, 24
        flow_shift = 8.0
        if shot["is_hero"]:
            n_steps = 30
            flow_shift = 5.0

        clip_path = Path(outdir) / f"clip_{i:02d}.mp4"
        last_path = Path(outdir) / f"clip_{i:02d}_last.png"
        kf_path = Path(outdir) / f"keyframe_{i:02d}.png"

        # resume support: skip already-rendered clip but make sure last frame is
        # available for the next shot's chained mode.
        if clip_path.exists():
            log.info(f"[clip {i}] exists, skip")
            out_paths.append(str(clip_path))
            if last_path.exists():
                prev_last_frame_path = str(last_path)
            prev_dominant = shot.get("dominant_subject", "A")
            continue

        # keyframe: chained Img2Img off previous clip when the shot continues the
        # same dominant character without a hard cut; otherwise Img2Img off the
        # matching anchor master (group ABC for multi-char shots, single-char
        # master for solo shots). strength=0.5 keeps the master's faces intact
        # and lets FLUX rearrange pose/composition per shot prompt.
        if not kf_path.exists():
            dom = shot.get("dominant_subject", "A")
            chained = (
                i > 0
                and not shot.get("cut", True)
                and prev_last_frame_path is not None
                and prev_dominant == dom
            )
            kf_prompt = _build_keyframe_prompt(shot, plan)
            t = time.perf_counter()
            if chained:
                log.info(f"[kf {i}] {w}x{h} chained from {Path(prev_last_frame_path).name}")
                kf_img = _flux_chained_keyframe(
                    flux_img, prev_last_frame_path, kf_prompt, w, h,
                    seed=flux_seed + i, strength=0.45,
                )
            else:
                anchor_path, anchor_tag = _pick_anchor(shot, plan, masters)
                log.info(f"[kf {i}] {w}x{h} anchor={anchor_tag} src={Path(anchor_path).name}")
                kf_img = _flux_chained_keyframe(
                    flux_img, anchor_path, kf_prompt, w, h,
                    seed=flux_seed + i, strength=0.5,
                )
            kf_img.save(kf_path)
            log.info(f"[kf {i}] done in {time.perf_counter()-t:.1f}s")
            _events.emit("keyframe_ready", shot=i, path=str(kf_path),
                         chained=chained, seconds=round(time.perf_counter()-t, 2))

        # FLF2V end keyframe for cut:false continuations - locks identity at both ends
        end_kf_path = Path(outdir) / f"keyframe_{i:02d}_end.png"
        last_image_pil = None
        if i + 1 < len(plan["shots"]) and not plan["shots"][i + 1].get("cut", True):
            if not end_kf_path.exists():
                log.info(f"[kf {i} end] FLF2V end frame -> next shot start")
                t_end = time.perf_counter()
                end_prompt = _build_keyframe_prompt(plan["shots"][i + 1], plan)
                end_img = _flux_chained_keyframe(
                    flux_img, str(kf_path), end_prompt, w, h,
                    seed=flux_seed + 1000 + i,
                )
                end_img.save(end_kf_path)
                log.info(f"[kf {i} end] done in {time.perf_counter()-t_end:.1f}s")
            last_image_pil = Image.open(end_kf_path).convert("RGB").resize((w, h))

        # Wan2.2 I2V (or FLF2V when last_image_pil is set) + critic-driven retry
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=flow_shift
        )
        ref = Image.open(kf_path).convert("RGB").resize((w, h))

        max_attempts = 3 if use_critic else 1
        clip_path_str = str(clip_path)
        for attempt in range(max_attempts):
            seed = base_seed + i + attempt * 1000
            log.info(f"[clip {i+1}/9] {w}x{h} attempt {attempt+1}/{max_attempts}, seed={seed}")
            _events.emit("clip_started", shot=i, attempt=attempt+1,
                         max_attempts=max_attempts, seed=seed,
                         flow_shift=flow_shift, n_steps=n_steps,
                         flf2v=last_image_pil is not None)
            t = time.perf_counter()
            # CFG (3.5, 3.0): high-noise matches upstream sample_guide_scale=3.5;
            # low-noise dropped from 3.5 to reduce over-rendering
            pipe_kwargs = dict(
                image=ref, prompt=shot["prompt"], negative_prompt=NEG_VIDEO,
                width=w, height=h, num_frames=num_frames, num_inference_steps=n_steps,
                guidance_scale=3.5, guidance_scale_2=3.0,
                generator=torch.Generator(device=GPU_WAN).manual_seed(seed),
            )
            if last_image_pil is not None:
                pipe_kwargs["last_image"] = last_image_pil
                if attempt == 0:
                    log.info(f"[clip {i}] FLF2V mode")
            out = pipe(**pipe_kwargs)
            dt = time.perf_counter() - t
            export_to_video(out.frames[0], clip_path_str, fps=24)
            log.info(f"  done in {dt/60:.2f} min")
            _events.emit("clip_rendered", shot=i, attempt=attempt+1,
                         path=clip_path_str, minutes=round(dt/60, 2))

            # always persist last frame, even on retries (later attempts overwrite)
            prev_last_frame_path = _save_last_frame(out.frames[0], last_path)

            if not use_critic or attempt + 1 == max_attempts:
                break

            # 4 frames sampled across the clip (start, mid-early, mid-late, end).
            # Single midframe missed motion (reel_v7: clip 4 had busker not bow because mid
            # was a transition frame). 4 frames let the critic judge the OVERALL motion arc.
            from PIL import Image as PIL_Image
            import numpy as _np
            n = len(out.frames[0])
            sample_idxs = [0, n // 3, (2 * n) // 3, n - 1]
            sample_paths = []
            for s_idx, fi in enumerate(sample_idxs):
                sp = Path(outdir) / f"clip_{i:02d}_frame{s_idx}.png"
                f = out.frames[0][fi]
                if not isinstance(f, PIL_Image.Image):
                    arr = _np.asarray(f)
                    if arr.dtype.kind == "f":
                        arr = (_np.clip(arr, 0, 1) * 255).astype(_np.uint8)
                    f = PIL_Image.fromarray(arr)
                f.save(sp)
                sample_paths.append(str(sp))

            # critic: free Wan2.2 (94 GB peak + Qwen3.5 70 GB > 192 GB HBM headroom).
            # FLUX img2img stays resident (~24 GB) so the next shot skips the reload.
            log.info(f"[critic {i}] freeing Wan2.2 for Qwen3.5 critique...")
            _events.emit("critic_starting", shot=i, attempt=attempt+1,
                         frames=sample_paths)
            del pipe
            _free_gpu()

            from director import DirectorAgent
            d = DirectorAgent()
            try:
                # actual chars in shot, not hardcoded char A (caused false drift on Kenji+Mei shots)
                expected = _which_chars_in_shot(shot, chars) or [shot.get("dominant_subject", "A")]
                expected = [k for k in expected if k in chars]
                if expected:
                    expected_portrait = " | ".join(f"{k}: {chars[k]}" for k in expected)
                else:
                    expected_portrait = "(scene shot, no specific characters expected)"
                score = d.critique(sample_paths, shot["prompt"], expected_portrait)
                log.info(f"[critic {i}] {score}")
            except (json.JSONDecodeError, RuntimeError, KeyError) as e:
                log.warning(f"[critic {i}] failed: {e}")
                score = {"overall": 10}
            _events.emit("critic_verdict", shot=i, attempt=attempt+1,
                         expected=expected, score=score)
            d.unload()
            _free_gpu()

            log.info(f"[clip {i}] reloading Wan2.2 after critic")
            pipe = _build_wan_i2v_pipe()

            if score.get("overall", 10) >= PASS_THRESHOLD:
                log.info(f"[clip {i}] passed critic at attempt {attempt+1}")
                _events.emit("clip_passed", shot=i, attempts=attempt+1, score=score)
                break
            log.info(f"[clip {i}] critic flagged, retrying with new seed")
            _events.emit("clip_retrying", shot=i, attempt=attempt+1, score=score)

        out_paths.append(clip_path_str)
        prev_dominant = shot.get("dominant_subject", "A")

    del pipe
    del flux_txt, flux_img
    _free_gpu()
    return out_paths


def gen_music(plan, outdir, duration_s=30):
    # ACE-Step v1 3.5B, 30s instrumental. Two sys.path hacks for its circular
    # import: pipeline_ace_step references `music_vocoder` as an absolute name
    # but the file lives at acestep/music_dcae/music_vocoder.py.
    music_path = Path(outdir) / "music.wav"
    if music_path.exists() and music_path.stat().st_size > 1000:
        log.info(f"music exists, skip: {music_path}")
        return str(music_path)

    sys.path.insert(0, "/root/ace-step/ACE-Step")
    sys.path.insert(0, "/root/ace-step/ACE-Step/acestep/music_dcae")

    import torch
    import torchaudio, soundfile as sf

    # ACE-Step's pipeline calls torchaudio.save which requires torchcodec on ROCm.
    # torchcodec isn't in the rocm container so torchaudio.save monkey-patches
    # to soundfile.write below.
    def _patched_save(uri, src, sample_rate, channels_first=True, **kwargs):
        arr = src.detach().cpu().to(torch.float32).numpy()
        if channels_first and arr.ndim == 2:
            arr = arr.T
        sf.write(str(uri), arr, int(sample_rate))
    torchaudio.save = _patched_save

    from acestep.pipeline_ace_step import ACEStepPipeline

    pipe = ACEStepPipeline(dtype="bfloat16", torch_compile=False, cpu_offload=False)
    log.info(f"music_style: {plan['music_style'][:80]!r}")
    t = time.perf_counter()
    pipe(
        audio_duration=duration_s,
        prompt=plan["music_style"],
        lyrics="[instrumental]",
        infer_step=27, guidance_scale=15.0, scheduler_type="euler", cfg_type="apg",
        omega_scale=10.0, manual_seeds="42",
        guidance_interval=0.5, guidance_interval_decay=0.0, min_guidance_scale=3.0,
        use_erg_tag=True, use_erg_lyric=False, use_erg_diffusion=True,
        oss_steps="", guidance_scale_text=0.0, guidance_scale_lyric=0.0,
        save_path=str(music_path),
    )
    log.info(f"music done in {time.perf_counter()-t:.1f}s")
    del pipe
    _free_gpu()
    return str(music_path)


# Kokoro voice defaults per language. See https://huggingface.co/hexgrad/Kokoro-82M
# for the full voice catalog (each lang has a small set of trained voices).
KOKORO_VOICES = {
    "a": "af_heart",       # American English, female warm
    "b": "bf_emma",        # British English, female
    "e": "ef_dora",        # Spanish, female
    "f": "ff_siwis",       # French, female
    "h": "hf_alpha",       # Hindi, female
    "i": "if_sara",        # Italian, female
    "j": "jf_alpha",       # Japanese, female
    "p": "pf_dora",        # Brazilian Portuguese, female
    "z": "zf_xiaobei",     # Mandarin Chinese, female
}


def _kokoro_say(pipe, text, voice):
    import numpy as np
    chunks = []
    for _, _, audio in pipe(text, voice=voice):
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        chunks.append(audio)
    return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]


def gen_voiceover(plan, outdir):
    # per-shot wavs aligned to clip beats (vo_NN.wav); single vo.wav as fallback
    try:
        from kokoro import KPipeline
        import soundfile as sf
    except ImportError as e:
        log.warning(f"kokoro not installed ({e}), skipping VO")
        return None

    lang = (plan.get("vo_lang") or "a").lower()[:1]
    if lang not in KOKORO_VOICES:
        log.warning(f"unknown vo_lang {lang!r}, fallback to a")
        lang = "a"
    voice = KOKORO_VOICES[lang]

    per_shot = plan.get("vo_script_per_shot")
    if isinstance(per_shot, list) and per_shot:
        outpaths = []
        n = len(per_shot)
        all_done = all((Path(outdir) / f"vo_{i:02d}.wav").exists() for i in range(n))
        if all_done:
            log.info(f"per-shot VO exists, skip ({n} files)")
            return [str(Path(outdir) / f"vo_{i:02d}.wav") for i in range(n)]
        pipe = KPipeline(lang_code=lang)
        for i, line in enumerate(per_shot):
            wav = _kokoro_say(pipe, line, voice)
            p = Path(outdir) / f"vo_{i:02d}.wav"
            sf.write(str(p), wav, 24000)
            outpaths.append(str(p))
            log.info(f"vo_{i:02d} done: {len(wav)/24000:.2f}s '{line[:50]}'")
            _events.emit("vo_chunk_ready", shot=i, path=str(p),
                         seconds=round(len(wav)/24000, 2), text=line)
        _free_gpu()
        return outpaths

    # legacy single-string path
    vo_path = Path(outdir) / "vo.wav"
    if vo_path.exists() and vo_path.stat().st_size > 1000:
        log.info(f"vo exists, skip: {vo_path}")
        return str(vo_path)
    pipe = KPipeline(lang_code=lang)
    wav = _kokoro_say(pipe, plan["vo_script"], voice)
    sf.write(str(vo_path), wav, 24000)
    log.info(f"vo done: {len(wav)/24000:.2f}s @ 24kHz")
    _free_gpu()
    return str(vo_path)


CLIP_DURATION_S = 121 / 24.0  # one Wan2.2 clip at our 121-frame / 24fps config


def mix_reel(clip_paths, music_path, vo_path, output_path):
    # concat clips + music bed + (per-shot or single) VO; vo_path is list or str
    output_path = Path(output_path)
    video_only = output_path.parent / "reel_video_only.mp4"

    n = len(clip_paths)
    inputs = [arg for p in clip_paths for arg in ("-i", str(p))]
    # 1280x704 lanczos upscale + crop to keep 480p crops aspect-correct
    parts = [
        f"[{i}:v]scale=1280:-2:flags=lanczos,crop=1280:704:0:(ih-704)/2,setsar=1[v{i}]"
        for i in range(n)
    ]
    parts.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[outv]")
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(parts),
           "-map", "[outv]", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-pix_fmt", "yuv420p", str(video_only)]
    log.info(f"concat: {video_only}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error(f"ffmpeg concat failed: {r.stderr[-1500:]}")
        return None

    vo_list = vo_path if isinstance(vo_path, list) else ([vo_path] if vo_path else [])
    vo_list = [v for v in vo_list if v and Path(v).exists()]

    if not vo_list:
        cmd = ["ffmpeg", "-y", "-i", str(video_only), "-i", str(music_path),
               "-filter_complex", "[1:a]volume=0.50[mixed]",
               "-map", "0:v", "-map", "[mixed]",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output_path)]
    elif len(vo_list) == 1:
        # legacy single-track VO (stretched over whole reel)
        cmd = ["ffmpeg", "-y", "-i", str(video_only), "-i", str(music_path), "-i", str(vo_list[0]),
               "-filter_complex",
               "[1:a]volume=0.30,aresample=async=1[bg];"
               "[2:a]volume=1.0,aresample=async=1,apad[vo];"
               "[bg][vo]amix=inputs=2:duration=longest:weights=0.6 1.0[mixed]",
               "-map", "0:v", "-map", "[mixed]",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output_path)]
    else:
        # per-shot VO: each track delayed by i*CLIP_DURATION_S, then mixed with music bed
        nv = len(vo_list)
        vo_inputs = [arg for p in vo_list for arg in ("-i", str(p))]
        # input indexes: 0=video, 1=music, 2..2+N-1=vo tracks
        delay_filters = []
        for i in range(nv):
            delay_ms = int(i * CLIP_DURATION_S * 1000)
            ch = i + 2
            delay_filters.append(
                f"[{ch}:a]adelay={delay_ms}|{delay_ms},volume=1.0,aresample=async=1[v{i}]"
            )
        bg = "[1:a]volume=0.30,aresample=async=1[bg]"
        mix_inputs = "[bg]" + "".join(f"[v{i}]" for i in range(nv))
        weights = " ".join(["0.6"] + ["1.0"] * nv)
        amix = f"{mix_inputs}amix=inputs={nv+1}:duration=first:weights={weights}[mixed]"
        fc = ";".join([bg, *delay_filters, amix])
        cmd = ["ffmpeg", "-y", "-i", str(video_only), "-i", str(music_path), *vo_inputs,
               "-filter_complex", fc,
               "-map", "0:v", "-map", "[mixed]",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output_path)]

    log.info(f"mix: {output_path} (vo tracks={len(vo_list)})")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error(f"ffmpeg mix failed: {r.stderr[-1500:]}")
        return None
    return str(output_path)
