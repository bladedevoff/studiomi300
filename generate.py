# studiomi300, text-to-reel pipeline on AMD MI300X.
# AMD x lablab.ai hackathon, May 2026.
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")   # diffusers / transformers spam too much

# ROCm env BEFORE importing torch, otherwise hipBLASLt won't pick up.
# learned the hard way: PYTORCH_HIP_ALLOC_CONF has to land before torch import,
# otherwise hipBLASLt picks the wrong allocator profile and ~3 GB goes missing.
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("GPU_MAX_HW_QUEUES", "2")
os.environ.setdefault("HIP_FORCE_DEV_KERNARG", "1")
# inductor caches survive container restart if /workspace is mounted
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
# HSA_ENABLE_SDMA=0, workaround for hangs on long multi-clip runs (ROCm/ROCm#3892?)
os.environ.setdefault("HSA_ENABLE_SDMA", "0")

from director import DirectorAgent, expand_character_refs
import utils
import events

log = logging.getLogger("studiomi300")


def _setup_logging(verbose=False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def make_reel(user_prompt, outdir, *, use_critic=False, plan_path=None):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    events.init(out)
    events.emit("started", prompt=user_prompt, use_critic=use_critic, outdir=str(out))

    # plan first, fail fast before loading 50GB of weights
    director = DirectorAgent()
    if plan_path and Path(plan_path).exists():
        log.info(f"reusing plan from {plan_path}")
        events.emit("plan_reused", path=str(plan_path))
        with open(plan_path) as f:
            plan = json.load(f)
    else:
        events.emit("plan_starting")
        plan = director.plan(user_prompt)
        with open(out / "plan.json", "w") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)

    plan = expand_character_refs(plan)
    with open(out / "plan_expanded.json", "w") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    log.info(f"plan ready: {plan.get('story_logline', '')[:80]}")
    events.emit(
        "plan_ready",
        logline=plan.get("story_logline", ""),
        n_shots=len(plan.get("shots", [])),
        characters=list((plan.get("characters") or {}).keys()),
        music_style=plan.get("music_style", ""),
        vo_lang=plan.get("vo_lang", "a"),
        shots=[{"index": s["index"], "shot_type": s.get("shot_type", ""),
                "dominant_subject": s.get("dominant_subject", ""),
                "cut": s.get("cut", True),
                "prompt": s.get("prompt", "")} for s in plan["shots"]],
    )

    # free Director before Wan2.2 loads: 70 GB vLLM + 94 GB Wan2.2 + 30 GB
    # ROCm cache > 192 GB HBM. critic re-loads Director between clips.
    director.unload()
    del director

    clips = utils.render_clips(plan, out, use_critic=use_critic)

    events.emit("music_starting", style=plan.get("music_style", ""))
    music = utils.gen_music(plan, out)
    events.emit("music_ready", path=str(music) if music else None)

    events.emit("vo_starting", lang=plan.get("vo_lang", "a"))
    vo = utils.gen_voiceover(plan, out)
    if isinstance(vo, list):
        events.emit("vo_ready", paths=vo, mode="per_shot")
    else:
        events.emit("vo_ready", path=str(vo) if vo else None, mode="single")

    events.emit("mix_starting")
    final = utils.mix_reel(clips, music, vo, out / "reel_final.mp4")
    events.emit("mix_done", path=str(final) if final else None)
    events.emit("completed", final=str(final) if final else None)
    return final


def main():
    p = argparse.ArgumentParser(description="StudioMI300, text to 30s reel")
    p.add_argument("--prompt", help="reel concept (1-2 sentences)")
    p.add_argument("--plan", help="reuse a director plan json instead of re-planning")
    p.add_argument("--out", default="/root/outputs/reel_default")
    p.add_argument("--critic", action="store_true",
                   help="enable Qwen3.5 vision critic + auto-retry on bad clips")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    _setup_logging(args.verbose)
    if not args.prompt and not args.plan:
        p.error("either --prompt or --plan required")

    log.info(f"out dir: {args.out}")
    final = make_reel(args.prompt or "(reused plan)", args.out,
                      use_critic=args.critic, plan_path=args.plan)
    log.info(f"done: {final}")


if __name__ == "__main__":
    main()
