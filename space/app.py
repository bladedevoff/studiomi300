from pathlib import Path

import gradio as gr

APP_ROOT = Path(__file__).parent
SHOWCASE_DIR = APP_ROOT / "showcase"

GITHUB_URL = "https://github.com/bladedevoff/studiomi300"


SHOWCASE_REELS = []  # gets repopulated when v18+ reels drop into showcase/


HACKATHON_BADGE = "amd-hackathon-2026"


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
        server_name="0.0.0.0", server_port=7860, share=False
    )
