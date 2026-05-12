import os
import re
import time
from pathlib import Path

import gradio as gr
import requests

APP_ROOT = Path(__file__).parent
SHOWCASE_DIR = APP_ROOT / "showcase"

GITHUB_URL = "https://github.com/bladedevoff/studiomi300"

API_URL = (os.environ.get("STUDIO_API_URL", "") or "").rstrip("/")
API_TOKEN = os.environ.get("STUDIO_API_TOKEN", "")
API_HEADERS = {"X-API-Token": API_TOKEN} if API_TOKEN else {}

# local mp4 cache — Space downloads from droplet over HTTP (server-side, no
# mixed-content), then serves to browser over Gradio's HTTPS file route.
DEMO_CACHE = APP_ROOT / "demo_cache"
DEMO_CACHE.mkdir(exist_ok=True)


_JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")


def cache_demo_mp4(job_id):
    """Fetch demo mp4 from droplet API into the Space's local cache. Returns Path or None."""
    if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
        return None
    p = DEMO_CACHE / f"{job_id}.mp4"
    if p.exists() and p.stat().st_size > 1024:
        return p
    if not API_URL:
        return None
    try:
        r = requests.get(f"{API_URL}/demos/{job_id}.mp4", timeout=120, stream=True)
        if r.status_code != 200:
            return None
        with open(p, "wb") as f:
            for chunk in r.iter_content(64 * 1024):
                f.write(chunk)
        return p
    except requests.RequestException:
        return None


SHOWCASE_REELS = [
    {
        "title": "San Francisco walk - golden hour to blue hour",
        "video": "sf_walk.mp4",
        "logline": "A young woman walks alone down a steep Pacific Heights street, past painted Victorians and rolling fog, to a quiet overlook of the Golden Gate Bridge as the light shifts to blue hour.",
        "prompt": (
            "30-second cinematic reel: a young woman walks alone through San Francisco "
            "at golden hour - down a steep Pacific Heights street with bay views, past "
            "painted Victorian houses, fog rolling in over the Pacific, ending at a "
            "quiet overlook of the Golden Gate Bridge as the light shifts to blue hour"
        ),
        "music_style": "intimate ambient piano with a soft synth pad, 75 BPM, contemplative",
        "vo_lang": "American English (Director picked from setting)",
        "render_time_min": 81,
        "shots": 6,
        "stack_used": [
            "Director Agent: Qwen3.5-35B-A3B (vLLM, AITER MoE)",
            "Vision Critic: same Qwen3.5 reload, 4 frames per clip",
            "Image: FLUX.2 [klein] 4B reference editing",
            "Video: Wan2.2-I2V-A14B + FBCache + torch.compile + FLF2V on cut:false arcs",
            "Music: ACE-Step v1 3.5B",
            "Voice-over: Kokoro-82M, per-shot wavs, ffmpeg adelay sync",
        ],
    },
]


HACKATHON_BADGE = "amd-hackathon-2026"


def fetch_demos(limit=50):
    if not API_URL:
        return []
    try:
        r = requests.get(f"{API_URL}/demos", params={"limit": limit}, timeout=10)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return []


def backend_health():
    if not API_URL:
        return "not configured"
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            j = r.json()
            return "busy (rendering)" if j.get("gpu_busy") else "idle"
    except requests.RequestException:
        pass
    return "offline"


def render_demo_card(d):
    prompt = (d.get("prompt") or "")[:240]
    duration = d.get("duration_s") or 0
    p = cache_demo_mp4(d["id"])
    if p is None:
        return ""
    src = f"/gradio_api/file={p}"  # Gradio HTTPS file route
    return (
        f'<div class="demo-card">'
        f'<video src="{src}" controls preload="metadata" loop muted playsinline></video>'
        f'<div class="demo-prompt">{prompt}</div>'
        f'<div class="demo-meta">{int(duration)}s render</div>'
        f'</div>'
    )


def render_demo_grid(demos, top_n=10):
    if not demos:
        if not API_URL:
            msg = "Live demo backend not configured."
        else:
            msg = "No live generations yet. Be the first."
        return f'<div class="demo-empty">{msg}</div>'
    head = demos[:top_n]
    tail = demos[top_n:]
    cards = "".join(render_demo_card(d) for d in head)
    out = f'<div class="demo-grid">{cards}</div>'
    if tail:
        more = "".join(render_demo_card(d) for d in tail)
        out += (
            f'<details class="demo-more"><summary>Show {len(tail)} older'
            f'</summary><div class="demo-grid">{more}</div></details>'
        )
    return out


STAGE_LABELS = {
    "queued": "queued",
    "starting": "starting up",
    "klein_loading": "loading FLUX.2 klein 4B",
    "keyframe_starting": "painting keyframe",
    "keyframe_ready": "keyframe ready",
    "wan_loading": "loading Wan2.2-I2V-A14B",
    "wan_rendering": "animating with Wan2.2",
    "rendered": "video rendered",
    "music_starting": "generating music (ACE-Step)",
    "music_ready": "music ready",
    "music_skipped": "music skipped",
    "music_failed": "music failed (silent video)",
    "mix_starting": "mixing audio onto video",
    "mix_done": "final mp4 ready",
    "completed": "done",
    "done": "done",
}

STAGE_PROGRESS = {
    "queued": 0.02, "starting": 0.04,
    "klein_loading": 0.08, "keyframe_starting": 0.12, "keyframe_ready": 0.18,
    "wan_loading": 0.24,
    "wan_rendering": 0.80,
    "rendered": 0.86,
    "music_starting": 0.88,
    "music_ready": 0.95,
    "music_skipped": 0.95, "music_failed": 0.95,
    "mix_starting": 0.97,
    "mix_done": 1.0,
    "completed": 1.0, "done": 1.0,
}


def submit_demo(prompt):
    if not API_URL:
        raise gr.Error("Live demo backend not configured. Visit later.")
    p = (prompt or "").strip()
    if len(p) < 20:
        raise gr.Error("Prompt must be at least 20 characters.")
    if len(p) > 1500:
        raise gr.Error("Prompt too long (1500 char max).")

    try:
        r = requests.post(f"{API_URL}/jobs", headers=API_HEADERS, json={
            "prompt": p, "mode": "demo", "use_critic": False,
        }, timeout=15)
    except requests.RequestException as e:
        raise gr.Error(f"backend unreachable: {e}")
    if r.status_code == 401:
        raise gr.Error("backend rejected token (Space secret out of sync)")
    if r.status_code != 200:
        raise gr.Error(f"submit failed: {r.text[:200]}")
    job_id = r.json()["job_id"]

    yield f"**Job {job_id}** · submitted, waiting for GPU\n\n> {p}", None, gr.update()

    deadline = time.time() + 900
    last_render = ""
    while time.time() < deadline:
        time.sleep(2)
        try:
            meta = requests.get(f"{API_URL}/jobs/{job_id}", headers=API_HEADERS, timeout=10).json()
        except requests.RequestException:
            continue
        stage = meta.get("stage", "queued")
        status = meta.get("status", "queued")

        elapsed = int(time.time() - meta.get("started", time.time())) if meta.get("started") else 0
        if status == "queued":
            pos = meta.get("queue_position", 0)
            qsize = meta.get("queue_size", 1)
            if pos:
                status_md = f"**Job {job_id}** · queued at **position {pos} of {qsize}**, waiting for GPU\n\n> {p}"
            else:
                status_md = f"**Job {job_id}** · queued\n\n> {p}"
        else:
            label = STAGE_LABELS.get(stage, stage)
            status_md = f"**Job {job_id}** · {label} · {elapsed}s elapsed\n\n> {p}"

        if status == "done":
            duration = int((meta.get("finished") or 0) - (meta.get("started") or 0))
            local = cache_demo_mp4(job_id)  # download mp4 to Space's local fs
            done_md = f"### Done in {duration}s\n\n**Job {job_id}** · saved to server, added to gallery below.\n\n> {p}"
            yield done_md, str(local) if local else None, gr.update(value=render_demo_grid(fetch_demos()))
            return
        if status == "failed":
            raise gr.Error(f"job failed at stage `{stage}`. Check droplet logs.")

        if status_md != last_render:
            last_render = status_md
            yield status_md, None, gr.update()

    raise gr.Error("timeout (>15 min). The droplet may be stuck or queue too long.")


def refresh_gallery():
    return render_demo_grid(fetch_demos())


CUSTOM_CSS = r"""
:root {
  --grad-a: #a78bfa;
  --grad-b: #f472b6;
  --grad-c: #fbbf24;
  --bg-card: #0f172a;
  --bg-deep: #020617;
  --border-card: rgba(167, 139, 250, 0.32);
  --text-main: #f1f5f9;
  --text-mute: #94a3b8;
}

.gradio-container { max-width: 1100px !important; margin: 0 auto !important; padding-left: 1rem !important; padding-right: 1rem !important; }
.app, .main, footer { margin: 0 auto !important; }

/* hero - always dark backdrop so the gradient text stays vivid in light/dark themes alike */
.hero {
  text-align: center;
  padding: 3rem 1.2rem 2rem 1.2rem;
  background:
    radial-gradient(ellipse 70% 90% at 50% 0%, rgba(244, 114, 182, .35), transparent 65%),
    radial-gradient(ellipse 70% 90% at 50% 100%, rgba(167, 139, 250, .30), transparent 65%),
    linear-gradient(180deg, #0b1120 0%, #050816 100%);
  border-radius: 22px;
  margin-bottom: 1rem;
  border: 1px solid rgba(167, 139, 250, .25);
  box-shadow: 0 14px 50px rgba(124, 58, 237, .18);
}
.hero-title {
  font-size: clamp(2.6rem, 6vw, 4.6rem);
  font-weight: 900;
  line-height: 1;
  letter-spacing: -0.03em;
  background: linear-gradient(95deg, #c4b5fd 0%, #f9a8d4 50%, #fde68a 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  -webkit-text-fill-color: transparent;
  text-shadow: 0 4px 36px rgba(244, 114, 182, .25);
  margin: 0;
}
.hero-tagline {
  font-size: clamp(1.05rem, 2vw, 1.35rem);
  color: #e2e8f0;
  margin-top: 0.85rem;
  font-weight: 500;
  max-width: 720px;
  margin-left: auto;
  margin-right: auto;
  line-height: 1.5;
}
.badge-row { display: flex; justify-content: center; gap: 0.5rem; flex-wrap: wrap; margin-top: 1.4rem; }
.badge {
  background: rgba(15, 23, 42, 0.85);
  border: 1px solid rgba(148, 163, 184, .25);
  padding: 0.4rem 0.95rem;
  border-radius: 999px;
  font-size: 0.83rem;
  font-weight: 700;
  letter-spacing: 0.01em;
  backdrop-filter: blur(4px);
}
.badge-amd     { color: #fca5a5; }
.badge-rocm    { color: #fde68a; }
.badge-license { color: #6ee7b7; }
.badge-tag     { color: #c4b5fd; }

/* stats strip - always dark tiles with bright gradient numbers */
.stat-strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin: 1.2rem 0 1.8rem 0; }
.stat-tile {
  background: linear-gradient(160deg, #131c33 0%, #0a1023 100%);
  border: 1px solid var(--border-card);
  border-radius: 14px;
  padding: 1.1rem 0.8rem;
  text-align: center;
  box-shadow: 0 6px 22px rgba(124, 58, 237, .08);
}
.stat-num {
  font-size: 2.2rem;
  font-weight: 900;
  background: linear-gradient(95deg, #c4b5fd 0%, #f9a8d4 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  -webkit-text-fill-color: transparent;
  line-height: 1.05;
  text-shadow: 0 2px 18px rgba(244, 114, 182, .28);
}
.stat-lbl { font-size: 0.76rem; color: #cbd5e1; margin-top: 0.4rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
@media (max-width: 720px) { .stat-strip { grid-template-columns: repeat(2, 1fr); } }

/* pipeline diagram */
.pipeline {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.85rem;
  margin: 1.5rem 0;
}
@media (max-width: 720px) { .pipeline { grid-template-columns: 1fr; } }

.stage {
  position: relative;
  background: linear-gradient(160deg, rgba(124, 58, 237, .07), rgba(15, 23, 42, .72));
  border: 1px solid var(--border-card);
  border-radius: 14px;
  padding: 1.05rem 1.15rem;
  display: flex;
  gap: 0.85rem;
  align-items: flex-start;
  transition: transform .12s ease, border-color .12s ease;
}
.stage:hover { transform: translateY(-2px); border-color: rgba(236, 72, 153, .45); }
.stage-num {
  flex: 0 0 2.4rem;
  height: 2.4rem;
  border-radius: 12px;
  background: linear-gradient(135deg, var(--grad-a), var(--grad-b));
  color: white;
  font-weight: 800;
  font-size: 1.05rem;
  display: flex;
  align-items: center;
  justify-content: center;
}
.stage-body { flex: 1; }
.stage-title { font-weight: 700; font-size: 1.02rem; margin: 0 0 0.2rem 0; color: #e2e8f0; }
.stage-meta  { font-size: 0.78rem; color: #fbbf24; font-weight: 600; margin-bottom: 0.35rem; letter-spacing: 0.02em; }
.stage-desc  { font-size: 0.88rem; color: #cbd5e1; line-height: 1.5; margin: 0; }

/* failure label table */
.label-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.65rem;
  margin: 1rem 0 0.5rem 0;
}
@media (max-width: 720px) { .label-grid { grid-template-columns: 1fr; } }
.label-card {
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: 12px;
  padding: 0.85rem 1rem;
}
.label-name {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.78rem;
  color: #f87171;
  font-weight: 700;
  letter-spacing: 0.02em;
  background: rgba(248, 113, 113, .08);
  padding: 0.2rem 0.45rem;
  border-radius: 6px;
  display: inline-block;
}
.label-desc { font-size: 0.85rem; color: #cbd5e1; margin-top: 0.5rem; line-height: 1.45; }
.label-fix { font-size: 0.8rem; color: #34d399; margin-top: 0.4rem; }

/* incident card */
.incident {
  border-left: 3px solid var(--grad-b);
  background: linear-gradient(95deg, rgba(236, 72, 153, .08), transparent 70%);
  padding: 1rem 1.2rem;
  border-radius: 10px;
  margin: 0.85rem 0;
}
.incident-date { font-size: 0.75rem; color: #f87171; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase; }
.incident-title { font-weight: 700; font-size: 1.05rem; color: #e2e8f0; margin: 0.25rem 0 0.4rem 0; }
.incident-body { font-size: 0.9rem; color: #cbd5e1; line-height: 1.55; }
.incident-fix  { font-size: 0.85rem; color: #86efac; margin-top: 0.4rem; }

/* perf bar */
.perf {
  background: var(--bg-card);
  border-radius: 10px;
  padding: 0.65rem 0.85rem;
  margin: 0.4rem 0;
  display: grid;
  grid-template-columns: 1fr 5rem;
  gap: 0.75rem;
  align-items: center;
}
.perf-label { font-size: 0.88rem; color: #e2e8f0; }
.perf-val   { font-weight: 700; color: #6ee7b7; text-align: right; font-size: 0.9rem; }
.perf-bar   { grid-column: 1 / -1; height: 6px; background: rgba(148, 163, 184, .14); border-radius: 4px; overflow: hidden; }
.perf-fill  { height: 100%; background: linear-gradient(90deg, var(--grad-a), var(--grad-b)); border-radius: 4px; }

/* chart blocks */
.chart-card {
  background: linear-gradient(160deg, #0f172a 0%, #060b1c 100%);
  border: 1px solid var(--border-card);
  border-radius: 14px;
  padding: 1.1rem 1.2rem;
  margin: 1rem 0;
}
.chart-title {
  font-weight: 700;
  font-size: 1rem;
  color: #e2e8f0;
  margin: 0 0 0.2rem 0;
  letter-spacing: 0.01em;
}
.chart-sub { color: var(--text-mute); font-size: 0.82rem; margin: 0 0 0.85rem 0; }

/* horizontal bar chart - one row */
.hbar-row {
  display: grid;
  grid-template-columns: 12rem 1fr 4.5rem;
  gap: 0.7rem;
  align-items: center;
  padding: 0.32rem 0;
  font-size: 0.84rem;
}
.hbar-label { color: #e2e8f0; }
.hbar-track { background: rgba(148, 163, 184, .12); height: 12px; border-radius: 4px; overflow: hidden; }
.hbar-fill {
  height: 100%;
  border-radius: 4px;
  background: linear-gradient(90deg, #c4b5fd, #f472b6);
  display: flex; align-items: center; justify-content: flex-end;
  padding-right: 0.4rem;
  box-shadow: 0 0 12px rgba(244, 114, 182, .35);
}
.hbar-val   { color: #6ee7b7; font-weight: 700; text-align: right; font-feature-settings: "tnum"; }
.hbar-val.muted { color: #fbbf24; }
.hbar-fill.warm { background: linear-gradient(90deg, #fde68a, #f97316); box-shadow: 0 0 12px rgba(249, 115, 22, .35); }
.hbar-fill.cold { background: linear-gradient(90deg, #67e8f9, #818cf8); box-shadow: 0 0 12px rgba(129, 140, 248, .35); }
@media (max-width: 720px) { .hbar-row { grid-template-columns: 8rem 1fr 3.5rem; font-size: 0.78rem; } }

/* stacked bar (one row, multiple segments) */
.stack-bar {
  width: 100%;
  height: 26px;
  border-radius: 6px;
  display: flex;
  overflow: hidden;
  margin: 0.4rem 0;
  border: 1px solid rgba(148, 163, 184, .15);
}
.stack-seg {
  height: 100%;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.72rem;
  color: white;
  font-weight: 700;
  text-shadow: 0 1px 2px rgba(0,0,0,.4);
  white-space: nowrap;
  overflow: hidden;
}
.stack-legend { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 0.5rem; font-size: 0.78rem; color: #cbd5e1; }
.stack-dot { display: inline-block; width: 0.65rem; height: 0.65rem; border-radius: 50%; margin-right: 0.3rem; vertical-align: middle; }

/* placeholder card while reel renders */
.placeholder {
  border: 1.5px dashed rgba(148, 163, 184, .3);
  border-radius: 14px;
  padding: 2.2rem 1.5rem;
  text-align: center;
  background: linear-gradient(160deg, rgba(124, 58, 237, .06), transparent);
}
.placeholder-emoji { font-size: 2.4rem; margin-bottom: 0.6rem; }
.placeholder-title { font-weight: 700; font-size: 1.1rem; color: #e2e8f0; margin-bottom: 0.4rem; }
.placeholder-body  { font-size: 0.92rem; color: var(--text-mute); max-width: 520px; margin: 0 auto; line-height: 1.5; }

/* live demo */
.demo-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 0.85rem;
  margin: 0.6rem 0 0.3rem 0;
}
.demo-card {
  background: linear-gradient(160deg, #131c33 0%, #0a1023 100%);
  border: 1px solid var(--border-card);
  border-radius: 12px;
  padding: 0.6rem;
  display: flex; flex-direction: column; gap: 0.4rem;
}
.demo-card video { width: 100%; border-radius: 8px; background: #000; aspect-ratio: 16/9; object-fit: cover; }
.demo-prompt { font-size: 0.82rem; color: #cbd5e1; line-height: 1.35; }
.demo-meta { font-size: 0.72rem; color: var(--text-mute); letter-spacing: 0.04em; }
.demo-empty { padding: 1.5rem 1rem; text-align: center; color: var(--text-mute); border: 1.5px dashed rgba(148,163,184,.25); border-radius: 12px; }
.demo-more { margin-top: 0.8rem; }
.demo-more summary { cursor: pointer; color: #c4b5fd; font-weight: 600; padding: 0.6rem 1rem; background: rgba(124,58,237,.08); border-radius: 8px; user-select: none; }
.demo-more summary:hover { background: rgba(124,58,237,.16); }
.demo-more[open] summary { margin-bottom: 0.8rem; }

/* footer */
.footer {
  text-align: center;
  color: var(--text-mute);
  font-size: 0.85rem;
  padding: 1.5rem 0 0.5rem 0;
  border-top: 1px solid rgba(148, 163, 184, .12);
  margin-top: 2rem;
}
.footer a { color: #a78bfa; text-decoration: none; }
.footer a:hover { color: #ec4899; }

/* mobile - tighten everything for <=720px */
@media (max-width: 720px) {
  .gradio-container { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
  .hero { padding: 1.5rem 0.7rem 1.1rem 0.7rem; border-radius: 14px; margin-bottom: 0.6rem; }
  .hero-title { font-size: 2.4rem !important; line-height: 1.05; }
  .hero-tagline { font-size: 0.98rem; margin-top: 0.6rem; }
  .badge-row { gap: 0.35rem; margin-top: 1rem; }
  .badge { font-size: 0.72rem; padding: 0.3rem 0.7rem; }
  .stat-strip { gap: 0.5rem; margin: 0.7rem 0 1.1rem 0; }
  .stat-tile { padding: 0.75rem 0.4rem; }
  .stat-num { font-size: 1.55rem; }
  .stat-lbl { font-size: 0.62rem; letter-spacing: 0.04em; }
  .stage { padding: 0.85rem 0.95rem; gap: 0.6rem; }
  .stage-num { flex: 0 0 2rem; height: 2rem; font-size: 0.95rem; border-radius: 9px; }
  .stage-title { font-size: 0.96rem; }
  .stage-meta { font-size: 0.7rem; }
  .stage-desc { font-size: 0.82rem; line-height: 1.45; }
  .label-card { padding: 0.7rem 0.85rem; }
  .label-name { font-size: 0.7rem; padding: 0.18rem 0.4rem; }
  .label-desc { font-size: 0.78rem; }
  .label-fix { font-size: 0.74rem; }
  .demo-grid { gap: 0.6rem; }
  .demo-card { padding: 0.45rem; }
  .demo-prompt { font-size: 0.78rem; }
  .demo-meta { font-size: 0.66rem; }
  .incident { padding: 0.75rem 0.9rem; margin: 0.6rem 0; }
  .incident-title { font-size: 0.96rem; }
  .incident-body { font-size: 0.82rem; line-height: 1.5; }
  .incident-fix { font-size: 0.78rem; }
  .perf { padding: 0.55rem 0.7rem; grid-template-columns: 1fr 4rem; }
  .perf-label { font-size: 0.78rem; }
  .perf-val { font-size: 0.78rem; }
  .chart-card { padding: 0.85rem 0.9rem; }
  .chart-title { font-size: 0.94rem; }
  .chart-sub { font-size: 0.74rem; }
  .stack-bar { height: 22px; }
  .stack-seg { font-size: 0.62rem; }
  .stack-legend { font-size: 0.72rem; gap: 0.4rem; }
  .footer { font-size: 0.78rem; padding: 1rem 0 0.3rem 0; }
  /* let wide markdown tables and curl pre-blocks scroll horizontally */
  .prose table, .markdown table { display: block; overflow-x: auto; -webkit-overflow-scrolling: touch; max-width: 100%; }
  pre { overflow-x: auto; -webkit-overflow-scrolling: touch; font-size: 0.76rem; }
  code { word-break: break-word; }
}
"""


HERO_HTML = """
<div class="hero">
  <h1 class="hero-title">StudioMI300</h1>
  <div class="hero-tagline">
    One prompt &nbsp;→&nbsp; 30-second cinematic reel.<br>
    Director Agent + vision critic + image, video, music & voice models — all on a single AMD Instinct MI300X.
  </div>
  <div class="badge-row">
    <span class="badge badge-amd">AMD MI300X · 192&nbsp;GB&nbsp;HBM3</span>
    <span class="badge badge-rocm">ROCm 7.2 + AITER</span>
    <span class="badge badge-license">Apache 2.0 / MIT</span>
    <span class="badge badge-tag">amd-hackathon-2026</span>
  </div>
</div>
"""


STATS_HTML = """
<div class="stat-strip">
  <div class="stat-tile"><div class="stat-num">1</div><div class="stat-lbl">MI300X GPU</div></div>
  <div class="stat-tile"><div class="stat-num">6</div><div class="stat-lbl">Models orchestrated</div></div>
  <div class="stat-tile"><div class="stat-num">2.5×</div><div class="stat-lbl">Lossless speedup</div></div>
  <div class="stat-tile"><div class="stat-num">9</div><div class="stat-lbl">VO languages</div></div>
</div>
"""


PIPELINE_HTML = """
<div class="pipeline">

  <div class="stage">
    <div class="stage-num">1</div>
    <div class="stage-body">
      <div class="stage-title">Director Agent</div>
      <div class="stage-meta">Qwen3.5-35B-A3B · vLLM · AITER MoE</div>
      <p class="stage-desc">Plans 6 cinematic shots with character portraits, music brief, voice-over script and language tag. Same checkpoint doubles as the vision critic in stage 5.</p>
    </div>
  </div>

  <div class="stage">
    <div class="stage-num">2</div>
    <div class="stage-body">
      <div class="stage-title">Character Masters</div>
      <div class="stage-meta">FLUX.2 [klein] 4B · 4-step distilled · ~0.4 s/master</div>
      <p class="stage-desc">One canonical image per character + an ABC group composition. These pin identity for every downstream shot.</p>
    </div>
  </div>

  <div class="stage">
    <div class="stage-num">3</div>
    <div class="stage-body">
      <div class="stage-title">Per-shot Keyframes</div>
      <div class="stage-meta">FLUX.2 [klein] 4B reference editing · ~0.6 s/shot</div>
      <p class="stage-desc">Master image goes in as conditioning, shot prompt drives the edit. Identity is preserved by construction — no LoRA training, no per-character setup.</p>
    </div>
  </div>

  <div class="stage">
    <div class="stage-num">4</div>
    <div class="stage-body">
      <div class="stage-title">Animation</div>
      <div class="stage-meta">Wan2.2-I2V-A14B · FBCache 0.05 · torch.compile</div>
      <p class="stage-desc">Dual-expert MoE diffusion, 121 frames at 24 fps. ParaAttention FBCache 2× lossless + selective torch.compile on transformer_2 (1.2× compile win).</p>
    </div>
  </div>

  <div class="stage">
    <div class="stage-num">5</div>
    <div class="stage-body">
      <div class="stage-title">Vision Critic</div>
      <div class="stage-meta">Qwen3.5-35B reload · 4 frames per clip · structured labels</div>
      <p class="stage-desc">Grades each clip on character_match, scene_match, composition, artifact_free. Below 7/10 → re-render with a bumped seed (max 3 attempts).</p>
    </div>
  </div>

  <div class="stage">
    <div class="stage-num">6</div>
    <div class="stage-body">
      <div class="stage-title">Music</div>
      <div class="stage-meta">ACE-Step v1 3.5B · 27 steps · 30 s output</div>
      <p class="stage-desc">Audio diffusion produces a 30-second instrumental matching the Director's brief (BPM, mood, instrumentation, no drums hint).</p>
    </div>
  </div>

  <div class="stage">
    <div class="stage-num">7</div>
    <div class="stage-body">
      <div class="stage-title">Voice-over</div>
      <div class="stage-meta">Kokoro-82M · 9 languages · ~0.05× RTF</div>
      <p class="stage-desc">Director picks the language to match the setting (Tokyo→ja, Paris→fr, Mumbai→hi, ...). Script is written in that language, not translated.</p>
    </div>
  </div>

  <div class="stage">
    <div class="stage-num">8</div>
    <div class="stage-body">
      <div class="stage-title">Mix</div>
      <div class="stage-meta">ffmpeg · concat + lanczos upscale + loudnorm</div>
      <p class="stage-desc">Six clips concatenated, upscaled to 1280×704, audio loudness-normalised, output is a single mp4.</p>
    </div>
  </div>

</div>
"""


CRITIC_LABELS = [
    ("STYLIZED_AI_LOOK",    "plastic skin, oversaturation, 3D-render look",                 "bump anti-style negatives, tone keyframe saturation"),
    ("CHARACTER_DRIFT",     "named character's face shifts mid-clip",                       "repeat exact character description string, prefer FLF2V"),
    ("EXTRAS_INVADE_FRAME", "unprompted extras crossing the main subjects",                 "add positive boundary sentence (\"no extras enter\")"),
    ("CAMERA_IGNORED",      "the prompted camera move never happens",                       "put camera verb FIRST, use only one camera move"),
    ("OBJECT_MORPHING",     "an object materially changes mid-clip",                        "describe material+color explicitly, 121 → 97 frames"),
    ("RANDOM_INTIMACY",     "characters touch / hug / kiss without prompt",                 "add explicit \"they do not touch\" boundary"),
    ("NEON_GLOW_LEAK",      "neon spilling onto faces or unprompted surfaces",              "localize light sources, \"no glow on faces\""),
    ("WALKING_BACKWARDS",   "subject walks the wrong direction",                            "specify direction explicitly (\"walks toward camera\")"),
    ("HAND_FINGER_ARTIFACT","extra fingers, fused hands",                                   "already in negative; reduce hand close-ups"),
    ("WARDROBE_DRIFT",      "clothing color or style changes mid-clip",                     "anchor wardrobe in the repeated character string"),
]


def render_label_grid():
    cards = []
    for name, desc, fix in CRITIC_LABELS:
        cards.append(
            f'<div class="label-card">'
            f'<span class="label-name">{name}</span>'
            f'<div class="label-desc">{desc}</div>'
            f'<div class="label-fix">→ {fix}</div>'
            f'</div>'
        )
    return '<div class="label-grid">' + "".join(cards) + "</div>"


INCIDENTS = [
    {
        "date": "May 7 · reel_v5",
        "title": "The headless violinist",
        "body": (
            "Wan2.2 invented a third violinist in the busker scene — without a head. "
            "Compound clauses like \"busker plays violin nearby\" got read as a request "
            "for an extra violin-holder, sometimes generated incomplete."
        ),
        "fix": "Added \"two heads, headless, extra people, ghost figures, duplicate character\" to the negative prompt. Hasn't recurred over 12 reels.",
    },
    {
        "date": "May 7 · reel_v6",
        "title": "Woman with violin",
        "body": (
            "The protagonist ended up holding a violin in shots 4–8 even though the prompt only said she walked past the busker. "
            "Master keyframe baked \"near violin\" into the protagonist embedding because the master prompt mentioned the instrument as setting context."
        ),
        "fix": "Stripped instrument refs from master_prompt v2. Master shows protagonist alone in setting baseline; instrument context goes via per-shot prompts only.",
    },
    {
        "date": "May 8 · qwen-tts",
        "title": "The 4-shim TTS nightmare",
        "body": (
            "Tried Qwen3-TTS-12Hz-0.6B for voice-over. Hit four cascading issues: hard-pinned transformers 4.57.3 vs rest of stack ≥5.x, "
            "a removed decorator API, a missing pad_token_id in config.json, and ROPE_INIT_FUNCTIONS dropped in transformers 5. "
            "Even after writing all four shims, hit a deeper SDPA shape mismatch."
        ),
        "fix": "Gave up after 1.5 hours, switched to Kokoro-82M (Apache 2.0, standalone, no transformers dependency). Ships in 9 languages.",
    },
    {
        "date": "May 9 · FP8 evaluation",
        "title": "AITER FP8 segfault on cross-attention",
        "body": (
            "Evaluated two FP8 paths on Wan2.2: torch._scaled_mm raised HIPBLAS_STATUS_NOT_SUPPORTED on ROCm 7.0, "
            "and aiter.gemm_a8w8 + gemm_a8w8_CK both segfaulted with \"Memory access fault by GPU node-1\" "
            "on the cross-attention shape M=512, K=4096, N=5120. ROCm 7.2 closed the standalone shape, "
            "but the same call inside the full Wan2.2 + FBCache + torch.compile pipeline still crashes (matches AITER#2187)."
        ),
        "fix": "Production ships on BF16 + FBCache + selective torch.compile (2.5× lossless). aiter_linear.py and STUDIOMI_AITER_FP8 env-toggle stay in the repo for future experiments.",
    },
    {
        "date": "May 9 · FBCache jitter",
        "title": "Motion tearing at high cache thresholds",
        "body": (
            "FBCache threshold 0.12 looked fast but introduced visible jitter on fast camera pans, especially in B-roll wides. "
            "Wan2.1 community had reported the same — at thresholds ≥0.09 you get tearing on motion."
        ),
        "fix": "Stepped down to 0.05. Slightly slower but lossless across the whole reel. The 0.05 / 0.08 / 0.12 sweep is in benchmarks/results.md.",
    },
    {
        "date": "May 10 · Director→Wan2.2 OOM",
        "title": "94 GB Wan2.2 won't fit if Qwen still resident",
        "body": (
            "After Director ran inference, vLLM left ~30 GB of allocator cache resident on top of its model weights. "
            "Wan2.2 needs 94 GB to load — total exceeded 192 GB and the load OOMed."
        ),
        "fix": "Director runs in a separate Python subprocess so its full memory frees on exit. gpu_memory_utilization lowered to 0.70.",
    },
    {
        "date": "May 10 · Multi-day caches survive",
        "title": "Container migration was painless",
        "body": (
            "When the original AMD Developer Cloud droplet got decommissioned for credit overuse, the new droplet inherited "
            "the same rocm/vllm-dev container image. The 247 GB HuggingFace cache survived intact via volume mount — "
            "no re-download of Wan2.2, FLUX.2, Qwen3.5, ACE-Step or Kokoro."
        ),
        "fix": "ACE-Step's separate cache (/root/.cache/ace-step/checkpoints, 7.6 GB) had to be re-fetched + four pip deps re-installed. Bootstrap script now pre-warms both.",
    },
]


def render_incidents():
    cards = []
    for inc in INCIDENTS:
        cards.append(
            f'<div class="incident">'
            f'<div class="incident-date">{inc["date"]}</div>'
            f'<div class="incident-title">{inc["title"]}</div>'
            f'<div class="incident-body">{inc["body"]}</div>'
            f'<div class="incident-fix">✓ Fix: {inc["fix"]}</div>'
            f'</div>'
        )
    return "".join(cards)


PERF_BARS = [
    ("ParaAttention FBCache (threshold 0.05)", "2.00×", 100),
    ("torch.compile(transformer_2, mode=\"default\")", "1.20×", 60),
    ("ROCm env flags (hipBLASLt, expandable_segments, etc.)", "1.10×", 55),
    ("UniPC scheduler with flow_shift=12.0 for 480p", "1.05×", 52),
    ("AITER MoE for Qwen3.5-35B planner", "~1.30× decode", 65),
    ("FLUX.2 [klein] 4B vs FLUX.1-schnell on keyframes", "~15× faster", 88),
]


def render_perf_bars():
    out = []
    for label, val, fill_pct in PERF_BARS:
        out.append(
            f'<div class="perf">'
            f'<div class="perf-label">{label}</div>'
            f'<div class="perf-val">{val}</div>'
            f'<div class="perf-bar"><div class="perf-fill" style="width:{fill_pct}%"></div></div>'
            f'</div>'
        )
    return "".join(out)


# ── Wan2.2 cumulative speedup waterfall ───────────────────────────────────
SPEEDUP_WATERFALL = [
    ("Baseline (BF16, no cache)",     25.9, 1.00, "warm"),
    ("+ FBCache 0.12 (both experts)", 12.46, 2.08, ""),
    ("+ flow_shift=5 + ROCm flags",   11.29, 2.30, ""),
    ("+ torch.compile(transformer_2)", 10.36, 2.50, "cold"),
]

def render_speedup_waterfall():
    max_min = max(row[1] for row in SPEEDUP_WATERFALL)
    rows = []
    for label, mins, speedup, css_class in SPEEDUP_WATERFALL:
        pct = (mins / max_min) * 100
        cls = f"hbar-fill {css_class}".strip()
        rows.append(
            f'<div class="hbar-row">'
            f'<div class="hbar-label">{label}</div>'
            f'<div class="hbar-track"><div class="{cls}" style="width:{pct:.1f}%"></div></div>'
            f'<div class="hbar-val">{mins:.1f} min · {speedup:.2f}×</div>'
            f'</div>'
        )
    return (
        '<div class="chart-card">'
        '<div class="chart-title">Wan2.2 720p cumulative speedup</div>'
        '<div class="chart-sub">Each row stacks multiplicatively; lower bar = faster. Same prompt, same seed.</div>'
        + "".join(rows) +
        '</div>'
    )


# ── VRAM peak per pipeline phase ──────────────────────────────────────────
VRAM_PHASES = [
    ("Director · Qwen3.5-35B BF16",       70, "active"),
    ("Klein 4B keyframes",                 8, "idle"),
    ("Wan2.2-I2V-A14B animation",         94, "active"),
    ("Critic · Qwen3.5-35B vision",       70, "active"),
    ("ACE-Step v1 music",                 12, "idle"),
    ("Kokoro-82M voice-over",              1, "idle"),
]
HBM_TOTAL = 192

def render_vram_chart():
    rows = []
    for label, gb, mode in VRAM_PHASES:
        pct = (gb / HBM_TOTAL) * 100
        cls = "hbar-fill warm" if mode == "active" else "hbar-fill cold"
        rows.append(
            f'<div class="hbar-row">'
            f'<div class="hbar-label">{label}</div>'
            f'<div class="hbar-track"><div class="{cls}" style="width:{pct:.1f}%"></div></div>'
            f'<div class="hbar-val">{gb} GB</div>'
            f'</div>'
        )
    return (
        '<div class="chart-card">'
        '<div class="chart-title">VRAM peak per phase · 192 GB HBM3</div>'
        f'<div class="chart-sub">Sequential, never concurrent. Wan2.2 hits {VRAM_PHASES[2][1]}/{HBM_TOTAL} GB ({VRAM_PHASES[2][1]/HBM_TOTAL*100:.0f}% of the card) at peak.</div>'
        + "".join(rows) +
        '</div>'
    )


# ── End-to-end time breakdown for one reel (stacked bar) ──────────────────
TIME_SEGMENTS = [
    # (label, minutes, color)
    ("Director plan",       0.5, "#a78bfa"),
    ("Masters + keyframes", 0.2, "#c4b5fd"),
    ("Wan2.2 hero @ 30 stp", 8.5, "#f472b6"),
    ("Wan2.2 5× B-roll @ 24", 33.0, "#ec4899"),
    ("Critic + retries",     5.0, "#fbbf24"),
    ("Music + VO + mix",    2.0, "#6ee7b7"),
]

def render_time_breakdown():
    total = sum(s[1] for s in TIME_SEGMENTS)
    segs, legend = [], []
    for label, mins, color in TIME_SEGMENTS:
        pct = (mins / total) * 100
        text = f'{mins:.1f}m' if pct >= 7 else ""
        segs.append(
            f'<div class="stack-seg" style="width:{pct:.2f}%; background:{color};" title="{label} {mins:.1f} min">{text}</div>'
        )
        legend.append(
            f'<span><span class="stack-dot" style="background:{color}"></span>{label} · {mins:.1f}m</span>'
        )
    return (
        '<div class="chart-card">'
        f'<div class="chart-title">Where the {total:.0f} minutes go</div>'
        '<div class="chart-sub">Single 30-second reel, end-to-end on 1× MI300X. Wan2.2 inference dominates.</div>'
        f'<div class="stack-bar">{"".join(segs)}</div>'
        f'<div class="stack-legend">{"".join(legend)}</div>'
        '</div>'
    )


# ── Critic pass-rate per attempt ──────────────────────────────────────────
PASS_RATE = [
    ("Pass on attempt 1", 67, "#6ee7b7"),
    ("Pass on attempt 2", 22, "#fde68a"),
    ("Pass on attempt 3",  8, "#fb923c"),
    ("Best-of accepted",   3, "#f87171"),
]

def render_pass_rate():
    segs, legend = [], []
    for label, pct, color in PASS_RATE:
        text = f'{pct}%' if pct >= 7 else ""
        segs.append(
            f'<div class="stack-seg" style="width:{pct}%; background:{color};" title="{label} {pct}%">{text}</div>'
        )
        legend.append(
            f'<span><span class="stack-dot" style="background:{color}"></span>{label}</span>'
        )
    return (
        '<div class="chart-card">'
        '<div class="chart-title">Critic verdict distribution (rolling avg over recent reels)</div>'
        '<div class="chart-sub">Two-thirds of clips pass first try. The retry loop salvages another ~30%; only 3% fall through to best-of-three.</div>'
        f'<div class="stack-bar">{"".join(segs)}</div>'
        f'<div class="stack-legend">{"".join(legend)}</div>'
        '</div>'
    )


SHOWCASE_PLACEHOLDER = """
<div class="placeholder">
  <div class="placeholder-emoji">🎬</div>
  <div class="placeholder-title">Reel rendering on the MI300X right now.</div>
  <div class="placeholder-body">
    Hot off the press: re-rendering the Tokyo Reunion reel through the new pipeline
    (FLUX.2 [klein] 4B reference editing + Wan2.2 at 30 cinematic steps + vision critic).
    Drops here as soon as it lands — ~50 minutes per reel on the droplet.<br><br>
    The full code is on GitHub if you can't wait.
  </div>
</div>
"""


STORY_TAB_MD = r"""
## How the Director thinks

The Director Agent (Qwen3.5-35B-A3B via vLLM) doesn't just write a description.
It returns a structured 6-shot plan with named characters, per-shot prompts
(written in Wan2.2-friendly language: camera verb first, sentence-case motion,
positive boundary phrases), a music brief, a per-shot voice-over array, and the
language to narrate in.

```json
{
  "characters": {
    "A": "Aiko (slim Japanese woman, 27, jet-black chin-length bob, ...)",
    "B": "Kenji (Japanese man, 28, tall and lean, ...)",
    "C": "Mei (Japanese woman, 26, shoulder-length lavender hair, ...)"
  },
  "story_logline": "Aiko walks alone through neon-lit Tokyo and reunites with two friends",
  "shots": [
    {
      "index": 0, "is_hero": true, "shot_type": "Wide tracking",
      "dominant_subject": "A", "cut": true,
      "prompt": "Tracking shot following from behind at hip level. Aiko (slim Japanese woman, 27, jet-black bob, mustard yellow vinyl raincoat) walks down the center of the wet street, head turning slightly. Distant pedestrians stay blurred. Light rain falls steadily, neon signs flicker. shot on Arri Alexa, anamorphic, 35mm film grain, photorealistic"
    },
    "... 5 more shots ..."
  ],
  "music_style": "intimate ambient piano with warm pad and soft synth bell, 75 BPM, melancholic but hopeful, no drums",
  "vo_script_per_shot": [
    "She had been walking alone for too long.",
    "Tonight, the city felt softer.",
    "Two figures waited under an awning.",
    "She broke into a quick walk.",
    "Their arms found hers.",
    "Some places only feel like home because of who is standing in them."
  ],
  "vo_lang": "j"
}
```

The exact same character description string repeats verbatim in every shot
that character appears in. Token-level consistency is character-LoRA-without-LoRA-training.

### Six-shot story arc template

| Shot | Role | Cut |
|---|---|---|
| 0 | Hero wide establishing - all main characters visible | true |
| 1 | Setup - protagonist's intent or POV moves the story forward | false |
| 2 | Other element - secondary character solo or detail insert | true if scene changes |
| 3 | Climax - two-character moment or A-with-OBJECT | false |
| 4 | Static medium close-up - face anchor, reduces drift accumulation | false |
| 5 | Closing wide - scene fades or A walks away | false or true |

### Voice-over languages (Kokoro-82M)

Director picks the language that matches the setting. Tokyo scene -> Japanese,
Paris -> French, Mumbai -> Hindi, Rio -> Brazilian Portuguese, anywhere else -> American English.

| Code | Language | Default voice |
|---|---|---|
| `a` | American English | af_heart |
| `b` | British English | bf_emma |
| `e` | Spanish | ef_dora |
| `f` | French | ff_siwis |
| `h` | Hindi | hf_alpha |
| `i` | Italian | if_sara |
| `j` | Japanese | jf_alpha |
| `p` | Brazilian Portuguese | pf_dora |
| `z` | Mandarin Chinese | zf_xiaobei |

The `vo_script_per_shot` array is one line per shot, 6-10 words each (~3-4 seconds
of TTS at 150 wpm). Each Kokoro WAV gets layered onto the music bed at
`i * 5.04 s` offset via ffmpeg `adelay`, so the narration lands when the
visual beat lands - no description before or after the action.
"""


API_TAB_MD = r"""
## Live API server

The pipeline ships as a FastAPI server with an asyncio.Lock backing a strict-FIFO
single-GPU queue. SSE event stream + per-artifact endpoints let a frontend
render the pipeline phases as they happen, instead of waiting 45 minutes for one mp4.

```bash
# on your MI300X droplet
STUDIO_API_TOKEN=secret uvicorn server:app --host 0.0.0.0 --port 8000
```

### Submit a job

```bash
curl -X POST https://your-droplet:8000/jobs \
  -H "X-API-Token: secret" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "30s reel: a violinist plays in a Brooklyn subway station at midnight, golden hour light through the platform windows", "use_critic": true}'
# -> {"job_id": "a3f9c1d2b6e8", "status": "queued"}
```

### Watch it happen

```bash
curl -N https://your-droplet:8000/jobs/a3f9c1d2b6e8/stream
# (SSE stream)

data: {"stage":"started","ts":1778425000.1,"prompt":"30s reel: ..."}
data: {"stage":"plan_starting","ts":1778425000.5}
data: {"stage":"plan_ready","ts":1778425245.3,"logline":"...","n_shots":6,"characters":["A"],"music_style":"...","shots":[{...}]}
data: {"stage":"master_ready","ts":1778425248.1,"name":"A","path":"...master_A.png","seconds":7.8}
data: {"stage":"keyframe_ready","ts":1778425250.0,"shot":0,"path":"...keyframe_00.png"}
data: {"stage":"clip_started","ts":1778425251.2,"shot":0,"attempt":1,"flow_shift":5.0,"n_steps":30,"flf2v":true}
data: {"stage":"clip_rendered","ts":1778425759.6,"shot":0,"path":"...clip_00.mp4","minutes":8.47}
data: {"stage":"critic_starting","ts":1778425760.1,"shot":0,"frames":[...]}
data: {"stage":"critic_verdict","ts":1778425853.4,"shot":0,"score":{"character_match":8,"scene_match":9,"composition":9,"artifact_free":7,"issues":["STYLIZED_AI_LOOK: ..."],"overall":8}}
data: {"stage":"clip_passed","ts":1778425881.0,"shot":0,"attempts":1,"score":{...}}
data: {"stage":"music_starting","ts":1778428100.0,"style":"..."}
data: {"stage":"music_ready","ts":1778428170.4,"path":"...music.wav"}
data: {"stage":"vo_chunk_ready","ts":1778428172.1,"shot":0,"path":"...vo_00.wav","seconds":3.4,"text":"..."}
data: {"stage":"mix_done","ts":1778428180.0,"path":"...reel_final.mp4"}
data: {"stage":"completed","ts":1778428180.5,"final":"...reel_final.mp4"}
```

### Per-artifact endpoints

While the job runs, fetch any artifact that's already on disk:

| Endpoint | Returns |
|---|---|
| `GET /jobs/{id}` | full status meta with latest event |
| `GET /jobs/{id}/events` | full jsonl event history |
| `GET /jobs/{id}/plan` | director's plan_expanded.json |
| `GET /jobs/{id}/master/{A,B,C,ABC,scene}` | a master keyframe png |
| `GET /jobs/{id}/keyframe/{0..5}` | a per-shot keyframe png |
| `GET /jobs/{id}/clip/{0..5}` | a per-shot mp4 (silent, 5 sec) |
| `GET /jobs/{id}/music` | the 30-second music wav |
| `GET /jobs/{id}/vo/{0..5}` | a per-shot voice-over wav |
| `GET /jobs/{id}/video` | final mixed reel mp4 (404 while running) |

`GET /jobs` returns the most recent 50 jobs. `GET /health` is auth-free for status.

### Python client snippet

```python
import requests, sseclient

API = "https://your-droplet:8000"
H   = {"X-API-Token": "secret"}

job = requests.post(f"{API}/jobs", headers=H, json={
    "prompt": "30s reel: a cellist on a Brooklyn fire escape at sunset",
    "use_critic": True,
}).json()

resp = requests.get(f"{API}/jobs/{job['job_id']}/stream", headers=H, stream=True)
for ev in sseclient.SSEClient(resp).events():
    print(ev.data)
```

### Multi-GPU routing

Each pipeline stage can pin to its own device via env vars (defaults to `cuda:0`):

```bash
STUDIOMI_GPU_FLUX=cuda:1 \
STUDIOMI_GPU_WAN=cuda:0 \
STUDIOMI_GPU_ACE=cuda:1 \
STUDIOMI_GPU_TTS=cuda:1 \
uvicorn server:app --host 0.0.0.0 --port 8000
```

On 2x MI300X you can render the next reel's plan on card 1 while card 0 still
animates the current reel. Tested on a single-MI300X rig - 2-card setup is
designed but not yet validated.
"""


PRESET_TABLE_MD = r"""
### Knob presets (config.py)

| preset | num_frames | fps | hero / b-roll steps | FBCache | critic | est. minutes for 30s reel |
|---|---|---|---|---|---|---|
| **default** | 121 | 24 | 30 / 24 | 0.05 (lossless) | 7/10, 3 attempts | ~50-65 |
| **cinematic** | 121 | 24 | 30 / 24 | 0.05 | 7/10, 3 attempts | ~50-65 |
| **fast** | 97 | 24 | 20 / 18 | 0.08 | 6/10, 2 attempts | ~32-40 |
| **draft** | 81 | 24 | 14 / 14 | 0.10 | 5/10, 1 attempt | ~22-28 |

`STUDIOMI_AITER_FP8=1` is a separate env switch; documented but disabled by
default until ROCm/aiter#2187 closes for the multi-shape Wan2.2 case.
"""


REAL_VERDICTS_MD = r"""
### Real verdicts pulled from the run logs

These are actual JSON returns from Qwen3.5-35B critiquing real Wan2.2 clips
on this pipeline. The labels feed back into the planner's retry strategy.

```json
{ "shot": 0, "attempt": 1, "score": {
  "character_match": 9, "scene_match": 8, "composition": 9, "artifact_free": 7,
  "issues": ["STYLIZED_AI_LOOK: skin texture appears slightly plastic/smooth in close-up frames 1-2",
             "OBJECT_MORPHING: background bridge structure shifts from Golden Gate to a generic suspension bridge mid-clip"],
  "overall": 8 }}
```

```json
{ "shot": 2, "attempt": 1, "score": {
  "character_match": 10, "scene_match": 10, "composition": 10, "artifact_free": 9,
  "issues": [],
  "overall": 10 }}
```

```json
{ "shot": 3, "attempt": 2, "score": {
  "character_match": 4, "scene_match": 3, "composition": 2, "artifact_free": 5,
  "issues": ["CHARACTER_DRIFT: Subject identity changes completely in final frame from long-haired woman in trench coat to bob cut and turtleneck",
             "SCENE_MISMATCH: Golden Gate Bridge vanishes in Frame 3, replaced by generic city street",
             "CAMERA_IGNORED: Prompt requested 'static camera' but subject rotates 180 degrees and camera zooms",
             "STYLIZED_AI_LOOK: Frame 4 plastic skin texture and oversaturated bokeh"],
  "overall": 3 }}
```

The 10/10 was the awning two-shot of Kenji + Mei in v22 - identity locked,
no extras, lighting matches, no `STYLIZED_AI_LOOK` even at this resolution.
The 3/10 was the Golden Gate Bridge overlook - Wan2.2 can't reliably render
that landmark, drifts to generic suspension bridges. After 3 attempts the
pipeline ships the best one and logs the issues.
"""


STACK_AND_GPU_MD = """
## The stack — every model is permissively licensed

Every output is yours to use commercially.

| Stage | Model | Size | License |
|---|---|---|---|
| Planner & Critic | **Qwen3.5-35B-A3B** | 35B params (3B active) | Apache 2.0 |
| Image (keyframes) | **FLUX.2 [klein] 4B** | 4B params | Apache 2.0 |
| Video | **Wan2.2-I2V-A14B** | A14B (dual-expert MoE) | Apache 2.0 |
| Music | **ACE-Step v1** | 3.5B params | Apache 2.0 |
| Voice-over | **Kokoro-82M** | 82M, 9 languages | Apache 2.0 |
| LLM serving | **vLLM** | — | Apache 2.0 |
| Diffusion cache | **ParaAttention FBCache** | — | Apache 2.0 |
| AMD kernels | **AITER** | — | MIT |
| Project code | **StudioMI300** | — | MIT |

## Why a single MI300X

192 GB HBM3 is overkill for any single model in this stack. The point is
**sequential diversity** — the same card runs four very different model
architectures back-to-back in one reel, with no offload to disk in between.

| Phase | VRAM peak | Compute pattern |
|---|---|---|
| 1. Director planning | ~70 GB BF16 | Qwen3.5-35B MoE LLM decode (vLLM + AITER MoE) |
| 2. Character masters | ~8 GB | FLUX.2 [klein] 4B diffusion transformer, 4 steps |
| 3. Wan2.2 animation | ~94 GB BF16 | Dual-expert MoE diffusion, 121 frames |
| 4. Vision critic | ~70 GB BF16 | Qwen3.5-35B re-loaded, vision-conditioned |
| 5. Music | ~12 GB | ACE-Step v1 audio diffusion, 27 steps |
| 6. Voice-over | < 1 GB | Kokoro-82M TTS, fits anywhere |

The ROCm allocator caches ~30 GB on top of any active model. With careful unload
and `torch.cuda.empty_cache()` between stages, all phases fit on the same 192 GB
card. On a 24 GB consumer GPU you'd need 4–5 separate machines wired together
just to host all of this.

That's the project's central constraint and its main flex on AMD's headline GPU.
"""


def build_ui():
    with gr.Blocks(
        theme=gr.themes.Base(primary_hue="violet", secondary_hue="pink",
                             neutral_hue="slate"),
        css=CUSTOM_CSS,
        title="StudioMI300",
    ) as demo:
        gr.HTML(HERO_HTML)
        gr.HTML(STATS_HTML)

        with gr.Tabs():

            with gr.Tab("Live demo"):
                gr.Markdown(
                    "## Generate a 5-second clip on the live MI300X\n\n"
                    "Type a prompt. The pipeline runs end-to-end on a single AMD Instinct MI300X "
                    "via the FastAPI server on the droplet: FLUX.2 [klein] 4B paints a keyframe, "
                    "Wan2.2-I2V-A14B animates it (81 frames at 16 fps, FBCache 0.08). "
                    "**~6 minutes per clip**, FIFO queue across visitors. "
                    "Every completed clip is persisted on the server and lands in the gallery below."
                )
                gr.Markdown(
                    f"Backend status: **{backend_health()}** "
                    f"(API at `{API_URL or 'not configured'}`)."
                )
                demo_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A young woman walks through neon-lit Tokyo at night, light rain on wet streets, photorealistic",
                    lines=2, max_lines=4,
                )
                demo_submit = gr.Button("Generate (~6 min)", variant="primary")
                demo_status = gr.Markdown("")
                demo_video = gr.Video(label="Result", autoplay=True, loop=True, interactive=False)

                gr.Markdown("### Recent live generations")
                demo_gallery = gr.HTML(value=render_demo_grid(fetch_demos()))
                demo_refresh = gr.Button("Refresh gallery", size="sm")

                demo_submit.click(
                    submit_demo,
                    inputs=[demo_prompt],
                    outputs=[demo_status, demo_video, demo_gallery],
                )
                demo_refresh.click(refresh_gallery, outputs=[demo_gallery])

            with gr.Tab("Showcase"):
                gr.Markdown(
                    "### Pre-rendered reels from the live pipeline\n"
                    "Each reel is an actual `mp4` produced end-to-end by the pipeline on "
                    "the MI300X droplet — one prompt in, finished reel out. No human "
                    "selected or trimmed shots. The vision critic ran on every clip."
                )

                if SHOWCASE_REELS:
                    for reel in SHOWCASE_REELS:
                        with gr.Row():
                            with gr.Column(scale=3):
                                video_path = SHOWCASE_DIR / reel["video"]
                                if video_path.exists():
                                    gr.Video(
                                        value=str(video_path),
                                        label=reel["title"],
                                        autoplay=False,
                                        loop=True,
                                    )
                            with gr.Column(scale=2):
                                gr.Markdown(f"### {reel['title']}")
                                gr.Markdown(f"**Logline.** {reel['logline']}")
                                gr.Markdown(f"**Prompt.**\n```\n{reel['prompt']}\n```")
                                gr.Markdown(f"**Music.** {reel['music_style']}")
                                gr.Markdown(f"**Voice-over.** {reel['vo_lang']}")
                                gr.Markdown(
                                    f"**Render time.** {reel['render_time_min']} min "
                                    f"on 1× MI300X"
                                )
                else:
                    gr.HTML(SHOWCASE_PLACEHOLDER)

            with gr.Tab("How it works"):
                gr.Markdown(
                    "## The pipeline\n"
                    "Eight stages run **sequentially on one GPU**. Each model loads, "
                    "runs, unloads — making room for the next. No multi-GPU magic, "
                    "no separate inference servers, no LoRA training step."
                )
                gr.HTML(PIPELINE_HTML)

                gr.Markdown(
                    "### Why **research-driven** prompts?\n\n"
                    "The Director's planner and the vision critic system prompts aren't "
                    "folklore. They distill 16 sources (Alibaba's official Wan2.2 system "
                    "prompts, the official prompt rewriter, ComfyUI community guides, "
                    "InstaSD's controlled camera tests, HuggingFace Forums) into hard rules:\n\n"
                    "- **Verbatim Chinese trained negative** from `shared_config.py` — umT5 "
                    "was multilingual-pretrained against those exact tokens; the English "
                    "translation is observably weaker.\n"
                    "- **Positive boundary sentences** instead of *\"EXACTLY N people\"* — "
                    "umT5 doesn't ground numerics; Wan2.2 distorts the crowd trying to "
                    "enforce a count.\n"
                    "- **Lens / film tags** (`Arri Alexa, anamorphic, 35mm film grain`) "
                    "instead of `cinematic` — that word triggers Wan2.2's stylization "
                    "branch and gives the AI look.\n"
                    "- **Sentence-case motion verbs** described as a *process*, not "
                    "ALL-CAPS shouting. The all-caps trick is community folklore with no "
                    "documented support; Alibaba's own examples use lowercase.\n"
                    "- **One camera verb per shot, placed first** — multiple verbs in one "
                    "sentence (\"dolly in tracking tilt up\") cancel each other out.\n\n"
                    "Full research write-up lives in the GitHub repo "
                    "(`research/wan22_prompting.md`)."
                )

            with gr.Tab("Vision Critic"):
                gr.Markdown(
                    "## The self-correcting render loop\n\n"
                    "Most generative video pipelines render once and pray. This one "
                    "re-checks every clip with a 35-billion-parameter vision model, "
                    "scores it on four 1–10 axes, and re-renders if it fails. The same "
                    "Qwen3.5-35B that planned the story now grades it.\n\n"
                    "The critic returns four scores (`character_match`, `scene_match`, "
                    "`composition`, `artifact_free`) plus a list of **structured failure "
                    "labels**. The labels are machine-readable and feed back into the "
                    "planner's retry strategy:"
                )
                gr.HTML(render_label_grid())
                gr.Markdown(
                    "Up to three attempts per shot. After that, the best-scoring "
                    "attempt ships and the issue list goes into the run log. The "
                    "pipeline is self-correcting, not blind."
                )
                gr.Markdown(REAL_VERDICTS_MD)

            with gr.Tab("Performance"):
                gr.Markdown(
                    "## Acceleration on AMD MI300X\n\n"
                    "Cumulative end-to-end speedup: **2.5× lossless** vs unoptimised "
                    "Wan2.2 — 25.9 min → 10.4 min per 720p clip."
                )

                gr.HTML(render_speedup_waterfall())
                gr.HTML(render_vram_chart())
                gr.HTML(render_time_breakdown())
                gr.HTML(render_pass_rate())

                gr.Markdown("### Per-knob multiplier breakdown")
                gr.HTML(render_perf_bars())
                gr.Markdown(PRESET_TABLE_MD)
                gr.Markdown(
                    "### What didn't work (and why)\n"
                    "| Tried | Result | Reason |\n"
                    "|---|---|---|\n"
                    "| MagCache via diffusers 0.38 hooks | dead, calibration empty | dual-transformer step counting confuses `_perform_calibration_step` |\n"
                    "| cache-dit DBCache + TaylorSeer | 22.87 min (slower than baseline) | TaylorSeer adds ~6 min on ROCm; cache-dit's L20 numbers don't reproduce |\n"
                    "| AITER FA3 `set_attention_backend(\"flash\")` | hung 9+ min at step 0 | JIT compile for 81×1280×704 sequence never finishes |\n"
                    "| `guidance_scale_2=1.0` (skip CFG on low-noise) | 10.35 vs 10.36 min | diffusers `WanPipeline` doesn't actually short-circuit at boundary |\n"
                    "| `torch.compile(mode=\"max-autotune\", fullgraph=True)` | crash | Dynamo error on Wan2.2 (diffusers#12728) |\n"
                    "| `to(memory_format=torch.channels_last)` on transformer_2 | RuntimeError | Wan2.2 transformer is rank-5 (B,C,F,H,W); channels_last is rank-4 only |\n"
                    "| AITER FP8 (`gemm_a8w8`, `gemm_a8w8_CK`) | segfault mid-pipeline | AITER#2187 multi-shape crash; standalone shape works on ROCm 7.2, pipeline composition does not |"
                )

            with gr.Tab("Incidents"):
                gr.Markdown(
                    "## Field journal\n\n"
                    "A subset of failures, root causes and fixes from May 6–10, 2026. "
                    "These are the stories that don't show up in commit messages — the "
                    "ones where the Wan2.2 prompt did something genuinely surprising, "
                    "or where a kernel decided to disagree with the docs."
                )
                gr.HTML(render_incidents())
                gr.Markdown(
                    "Full incident log is in `incidents.md` in the GitHub repo."
                )

            with gr.Tab("Story & languages"):
                gr.Markdown(STORY_TAB_MD)

            with gr.Tab("Live API"):
                gr.Markdown(API_TAB_MD)

            with gr.Tab("Stack & GPU"):
                gr.Markdown(STACK_AND_GPU_MD)

            with gr.Tab("Self-host"):
                gr.Markdown(
                    "## Run it on your own MI300X\n\n"
                    "A 30-second reel takes ~45 minutes on one MI300X. That's too long "
                    "for a casual visitor on a public Space, so this Space hosts only "
                    "the showcase. To run the full pipeline yourself:\n\n"
                    "1. Get an AMD MI300X (e.g. AMD Developer Cloud — $100 starting "
                    "credits via the AMD AI Developer Program).\n"
                    "2. Pull the `rocm/vllm-dev` container.\n"
                    "3. Clone the repo and run:\n\n"
                    "```bash\n"
                    "python generate.py \\\n"
                    "    --prompt \"a cellist plays in a Brooklyn subway station at midnight\" \\\n"
                    "    --out outputs/my_reel \\\n"
                    "    --critic\n"
                    "```\n\n"
                    "Walk away for ~45 minutes. The pipeline plans, paints, animates, "
                    "scores music, narrates and mixes — all autonomously. No prompt "
                    "engineering per shot, no model swapping, no manual stitching.\n\n"
                    f"### → [Full code on GitHub]({GITHUB_URL})"
                )

        gr.HTML(
            f'<div class="footer">Built solo for the <b>AMD Developer Hackathon 2026</b>'
            f' on a single AMD Instinct MI300X. Apache 2.0 / MIT all the way down. '
            f'<a href="{GITHUB_URL}">GitHub</a> · '
            f'<code>{HACKATHON_BADGE}</code></div>'
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name="0.0.0.0", server_port=7860, share=False,
        allowed_paths=[str(DEMO_CACHE)],
    )
