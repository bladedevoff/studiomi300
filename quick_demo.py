# 5-second single-clip demo: klein 4B keyframe + Wan2.2 I2V, no audio, no critic.
# Aimed for ~6-7 min wall on a single MI300X. Used by the API server's mode=demo path.
import os
import sys
import time
import logging
import argparse
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("GPU_MAX_HW_QUEUES", "2")
os.environ.setdefault("HIP_FORCE_DEV_KERNARG", "1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("HSA_ENABLE_SDMA", "0")

import events

log = logging.getLogger("studiomi300.quick_demo")

# minimal trained-on negative for photoreal demo (verbatim Chinese + anti-style additive)
NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    "，plastic skin, oversaturated, AI artifact, melted face, 3d render, cartoon, anime"
)

LENS_TAIL = "shot on Arri Alexa, anamorphic, 35mm film grain, photorealistic"


def run(prompt, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    events.init(out)
    events.emit("started", prompt=prompt, mode="demo")

    import torch

    # 1) klein 4B keyframe straight from the user prompt
    events.emit("klein_loading")
    t = time.perf_counter()
    from diffusers import Flux2KleinPipeline
    klein = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16
    ).to("cuda")
    log.info(f"klein loaded in {time.perf_counter()-t:.1f}s")

    events.emit("keyframe_starting")
    t = time.perf_counter()
    img = klein(
        prompt=f"{prompt}. {LENS_TAIL}",
        width=832, height=480, num_inference_steps=4,
        generator=torch.Generator(device="cuda").manual_seed(7777),
    ).images[0]
    kf_path = out / "keyframe.png"
    img.save(kf_path)
    events.emit("keyframe_ready", path=str(kf_path), seconds=round(time.perf_counter()-t, 2))

    del klein
    import gc; gc.collect(); torch.cuda.empty_cache()

    # 2) Wan2.2 I2V single 5-second clip, FBCache 0.08, 24 steps, flow_shift=8
    events.emit("wan_loading")
    t = time.perf_counter()
    from diffusers import WanImageToVideoPipeline, AutoencoderKLWan, UniPCMultistepScheduler
    from para_attn.first_block_cache.diffusers_adapters import (
        apply_cache_on_pipe, apply_cache_on_transformer,
    )
    from diffusers.utils import export_to_video
    from PIL import Image

    mid = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(mid, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
    apply_cache_on_pipe(pipe, residual_diff_threshold=0.08)
    apply_cache_on_transformer(pipe.transformer_2)
    pipe.transformer_2 = torch.compile(pipe.transformer_2, mode="default", fullgraph=False, dynamic=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
    log.info(f"wan loaded in {time.perf_counter()-t:.1f}s")

    events.emit("wan_rendering", n_steps=24, num_frames=81, fps=16, fbcache=0.08, flow_shift=8.0)
    t = time.perf_counter()
    ref = Image.open(kf_path).convert("RGB").resize((832, 480))
    out_pipe = pipe(
        image=ref,
        prompt=f"{prompt}. {LENS_TAIL}",
        negative_prompt=NEG,
        width=832, height=480, num_frames=81, num_inference_steps=24,
        guidance_scale=3.5, guidance_scale_2=3.0,
        generator=torch.Generator(device="cuda").manual_seed(5000),
    )

    silent = out / "demo_silent.mp4"
    export_to_video(out_pipe.frames[0], str(silent), fps=16)
    log.info(f"wan rendered in {(time.perf_counter()-t)/60:.2f} min")
    del pipe; gc.collect(); torch.cuda.empty_cache()
    events.emit("rendered", path=str(silent), minutes=round((time.perf_counter()-t)/60, 2))

    # 3) ACE-Step v1 music, 5 seconds, derived style from prompt
    events.emit("music_starting")
    try:
        import utils
        music_style = f"cinematic ambient score matching: {prompt[:120]}, atmospheric, 75 BPM, no drums"
        plan_stub = {"music_style": music_style}
        music_path = utils.gen_music(plan_stub, out, duration_s=5)
        if music_path:
            events.emit("music_ready", path=str(music_path))
        else:
            events.emit("music_skipped")
    except Exception as e:
        log.warning(f"music gen failed: {e}")
        events.emit("music_failed", error=str(e))
        music_path = None

    # 4) mux into final mp4
    events.emit("mix_starting")
    import subprocess
    final = out / "demo.mp4"
    if music_path and Path(music_path).exists():
        cmd = ["ffmpeg", "-y", "-i", str(silent), "-i", str(music_path),
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(final)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not final.exists():
            log.warning(f"ffmpeg mux failed: {r.stderr[-400:]}")
            silent.replace(final)
    else:
        silent.replace(final)
    events.emit("mix_done", path=str(final))
    events.emit("completed", final=str(final))
    return str(final)


def main():
    p = argparse.ArgumentParser(description="StudioMI300 quick demo (single 5-sec clip)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )
    final = run(a.prompt, a.out)
    log.info(f"done: {final}")


if __name__ == "__main__":
    main()
