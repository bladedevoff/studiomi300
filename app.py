# Gradio UI , showcase + (where possible) live generation.
# When deployed to HF Space without MI300X, the Generate tab gracefully shows
# a "self-host" CTA pointing at the GitHub repo's deploy guide.
import os
import subprocess
import logging
from pathlib import Path

import gradio as gr

log = logging.getLogger("studiomi300.app")
APP_ROOT = Path(__file__).parent

SHOWCASE = [
    {
        "title": "Busker encounter (golden hour)",
        "prompt": (
            "30-second cinematic mini-story: a street busker plays a violin in a sunset-lit "
            "European alley, a small dog dances on its hind legs nearby, a young woman walking "
            "by stops to listen, kneels to pet the dog, the dog wags its tail and licks her hand, "
            "golden hour light"
        ),
        "video": "showcase/busker_dog.mp4",
        "render_min": 43,
        "music_style": "warm acoustic guitar and cello duet, 85 BPM",
        "logline": "A solitary young woman pauses to connect with a dancing dog and the busker's music.",
        "notes": "first reel that survived critic-driven retry with score>=7 across all 9 shots",
    },
    {
        "title": "Tokyo neon walk",
        "prompt": (
            "30-second cinematic reel: a young woman walks through a neon-lit Tokyo street at "
            "night, reflections in puddles, atmospheric mood"
        ),
        "video": "showcase/tokyo_neon.mp4",
        "render_min": 33,
        "music_style": "moody synthwave instrumental, 90 BPM",
        "logline": "Solitary night walk under neon signs.",
        # less stable than busker , Wan2.2 sometimes adds a second pedestrian. retry helps.
    },
]


PIPELINE_MD = """
## How it works

One prompt -> 30s mp4. Single MI300X. ~45 min/reel.

### Pipeline

**Director Agent** -- Qwen3.5-35B-A3B BF16 via vLLM 0.20. Outputs a 9-shot plan
(character portraits, per-shot prompt, music brief, VO script). The same model
doubles as the vision critic when `--critic` is on.

**Per-shot keyframes** -- FLUX.1-schnell text2img with all referenced character
portraits inlined into the prompt. Continuation shots (same dominant character,
no hard cut) instead chain Img2Img off the previous clip's last frame so the
face doesn't reset between cuts.

**Wan2.2-I2V-A14B** + ParaAttention FBCache + selective `torch.compile`.
9 clips: ~13 min hero @ 1280x704, ~3.5 min B-roll @ 832x480. Wan2.2's
`boundary_ratio=0.875` routes ~87% of timesteps through `transformer_2`, so
that's the only branch under `torch.compile`. `max-autotune+fullgraph=True`
crashes Dynamo on Wan2.2 (diffusers#12728); `mode="default"` is what survives.

**ACE-Step v1** music (~30s instrumental) + **Kokoro-82M** VO + ffmpeg mix.

### Optional: vision critic loop

`--critic` re-uses Qwen3.5 to score four frames sampled across each clip. If
the overall score is <7/10, the clip regenerates with a bumped seed (3 attempts
max). Adds 5 to 10 min per reel: the Wan2.2 unload + Qwen3.5 reload between
clips dominates, because keeping both resident wants 94+70 = 164 GB and the
ROCm allocator caches another ~30 GB on top.
"""


def _has_mi300x():
    try:
        out = subprocess.run(["rocm-smi", "--showproductname"], capture_output=True, timeout=5)
        return b"MI300X" in out.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def generate_reel(user_prompt, use_critic, progress=gr.Progress()):
    if not _has_mi300x():
        raise gr.Error(
            "Live generation needs an AMD MI300X. This Space hosts the showcase. "
            "To run end-to-end, follow the README.md self-host guide."
        )
    if not user_prompt or len(user_prompt.strip()) < 20:
        raise gr.Error("Prompt must be at least 20 characters.")

    out_dir = "/tmp/studiomi300_run"
    os.makedirs(out_dir, exist_ok=True)

    # Spawn generate.py as a subprocess so its imports (vllm, diffusers etc) don't pollute
    # the gradio process. Gradio just polls log lines for progress.
    cmd = [
        "python", str(APP_ROOT / "generate.py"),
        "--prompt", user_prompt,
        "--out", out_dir,
    ]
    if use_critic:
        cmd.append("--critic")
    progress(0.0, desc="Director Agent planning…")
    log.info(f"running: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # rough progress mapping. percentages eyeballed from one full reel timing ,
    # Wan2.2 is the bottleneck so it owns [0.20, 0.90].
    # TODO: replace with actual per-stage time estimation
    for line in p.stdout:
        sys_line = line.strip()
        log.info(sys_line)
        if "loading Qwen" in sys_line:
            progress(0.03, desc="loading planner")
        elif "plan ready" in sys_line:
            progress(0.06, desc="story plan ready")
        elif "master prompt" in sys_line:
            progress(0.09, desc="FLUX master")
        elif "FLUX Img2Img" in sys_line:
            progress(0.13, desc="keyframes")
        elif "loading wan" in sys_line.lower():
            progress(0.18, desc="loading Wan2.2")
        elif "[clip " in sys_line:
            try:
                m = sys_line.split("[clip ")[1].split("]")[0]
                cur, total = m.split("/")
                # Wan2.2 owns [0.20, 0.90] , 70% of total wall-clock
                progress(0.20 + 0.70 * (int(cur) / int(total)), desc=f"clip {cur}/{total}")
            except (ValueError, IndexError):
                pass
        elif "music_style" in sys_line:
            progress(0.91, desc="music")
        elif "vo done" in sys_line:
            progress(0.95, desc="voiceover")
        elif "concat" in sys_line:
            progress(0.98, desc="mixing")
    p.wait()

    final = Path(out_dir) / "reel_final.mp4"
    if p.returncode != 0 or not final.exists():
        raise gr.Error(f"pipeline failed (exit {p.returncode}). Check logs.")
    progress(1.0, desc="done")
    return str(final)


# UI
with gr.Blocks(theme=gr.themes.Soft(), title="StudioMI300") as demo:
    gr.Markdown(
        "# StudioMI300\n"
        "**One prompt to 30s cinematic reel.** Single AMD Instinct MI300X. "
        "Apache 2.0 / MIT stack , see *How it works* tab for license breakdown."
    )

    with gr.Tabs():
        with gr.Tab("Showcase"):
            gr.Markdown("### Pre-rendered demo reels (real outputs from the pipeline)")
            for item in SHOWCASE:
                with gr.Row():
                    with gr.Column(scale=2):
                        vp = APP_ROOT / item["video"]
                        if vp.exists():
                            gr.Video(value=str(vp), label=item["title"], autoplay=False)
                        else:
                            gr.Markdown(f"*(showcase {item['video']} not bundled , see GitHub releases)*")
                    with gr.Column(scale=1):
                        gr.Markdown(f"### {item['title']}")
                        gr.Markdown(f"**Logline.** {item['logline']}")
                        gr.Markdown(f"**Music.** {item['music_style']}")
                        gr.Markdown(f"**Render time.** {item['render_min']} min on 1× MI300X")
                        if "notes" in item:
                            gr.Markdown(f"**Note.** {item['notes']}")
                        gr.Markdown(f"**Prompt.**\n```\n{item['prompt']}\n```")

        with gr.Tab("Generate"):
            gr.Markdown(
                "### Live generation (~45 min/reel on MI300X)\n"
                "Type a 30-second-reel concept. Director Agent plans 9 cinematic shots, FLUX "
                "anchors a master keyframe, Wan2.2-I2V-A14B animates each shot, ACE-Step scores "
                "the music, Kokoro narrates, ffmpeg mixes."
            )
            prompt = gr.Textbox(
                label="Reel concept",
                placeholder=(
                    "30-second cinematic mini-story: a child meets a stray dog in a rainy Berlin park, "
                    "the dog leads them to a hidden street performer playing accordion, golden hour light"
                ),
                lines=3,
            )
            critic = gr.Checkbox(
                label="Enable Qwen3.5 vision critic + auto-retry (slower, better quality)",
                value=False,
            )
            btn = gr.Button("Generate Reel", variant="primary")
            video = gr.Video(label="Your reel")
            btn.click(generate_reel, inputs=[prompt, critic], outputs=[video])

        with gr.Tab("How it works"):
            gr.Markdown(PIPELINE_MD)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    # max_size=4 to keep the queue from chewing MI300X for nine hours
    demo.queue(max_size=4).launch(server_name="0.0.0.0", server_port=7860, share=False)
