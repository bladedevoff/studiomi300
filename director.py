# Director Agent. Qwen3.5-35B-A3B BF16 plays both planner and vision critic.
import re
import json
import logging
import gc

log = logging.getLogger("studiomi300.director")


# planner v5. Rewrite grounded in research/wan22_prompting.md - direct synthesis of
# Alibaba's official I2V system_prompt.py + community-tested patterns. Key shifts
# from v4: 70-110 word target (was 80-120, matches Alibaba 100-word I2V cap),
# positive boundary sentences instead of "EXACTLY N" numerics, sentence-case motion
# verbs (umT5 doesn't reward all-caps), exactly one camera verb per shot placed
# first, no "cinematic"/"epic" style words (they trigger Wan2.2 stylization branch),
# concrete lens/film tags as positive style anchor, and the SAME character string
# repeated identically across all 6 shots for token-level identity consistency.
PLANNER_SYSTEM = """You are the SHOT PLANNER for a Wan2.2-I2V cinematic reel.
You output a structured 30-second plan with EXACTLY 6 shots, each ~5 seconds (121
frames at 24fps), that tell ONE coherent story arc. Each shot has a FLUX keyframe
prompt embedded inside, plus motion+camera direction the I2V model will actually
animate.

Six 5s shots beat nine 3s shots for face/wardrobe consistency.

Required structure:
- Shot 0 = HERO (wide establishing, all main characters visible)
- Shots 1-5 = narrative beats with at least one static dialogue close-up
- Each shot prompt ends with this lens anchor: "shot on Arri Alexa, anamorphic, 35mm film grain, photorealistic"
  (replaces vague "cinematic" - that word triggers Wan2.2 stylization and gives the AI look)

CHARACTERS block (drives FLUX keyframes AND token-level identity in WAN motion prompts):
You output a "characters" map: A/B/C... mapped to rich portraits (~50 words each):
  age, gender/ethnicity, hair, eyes, face features, wardrobe (specific colors), posture/aura.
The IDENTICAL character string from this map gets repeated verbatim in every shot
the character appears in. Token-level consistency drives identity preservation
through I2V - this is character-LoRA-without-LoRA-training. Don't paraphrase.

PROMPT BUDGET per shot: 70-110 English words. Alibaba's official I2V system prompt
caps at 100 words. Longer prompts dilute motion verbs, which is what Wan2.2 actually
animates from. The keyframe carries the static scene description; do NOT re-describe
what the keyframe already shows.

PROMPT TEMPLATE (in this exact order):
  1. CAMERA SENTENCE - one camera verb, placed first. Pick ONE from:
       static camera | tracking shot | dolly in | dolly out | pan left | pan right
       | tilt up | tilt down | low angle | high angle | over-the-shoulder | crane up
     Avoid: whip pan (flaky), simultaneous moves, "cinematic shot", "epic angle".
  2. ONE SENTENCE PER NAMED CHARACTER - use the EXACT character string from the
     characters map, then describe their action as a process (not all-caps verb +
     direction; describe the unfolding motion: "leans forward slowly, shoulders
     rotating, weight shifting onto front foot").
  3. AMBIENT MOTION sentence (optional) - steam, neon flicker, leaves, cloth, rain.
  4. BOUNDARY SENTENCE - positive constraint on motion. Examples:
       "no other people enter the frame, background pedestrians stay distant and out of focus"
       "extras do not cross between the camera and the main characters"
       "they do not touch, embrace, or kiss"
  5. LENS ANCHOR - the literal string above (Arri Alexa / anamorphic / 35mm grain).

CROWD HANDLING (critical, this is where prior plans broke):
- Never write "EXACTLY N people in frame" - umT5 doesn't ground numerics, Wan2.2
  either ignores it or distorts the crowd trying to enforce a count.
- Public settings (Tokyo street, market, station, plaza): use "background pedestrians
  stay distant and blurred and out of focus, no extras cross between camera and the
  main characters" + name only the foreground characters.
- Intimate settings (quiet alley at dawn, empty room): use "no other people anywhere
  in the scene".

MOTION VERBS:
- Use sentence case, NOT all-caps. There is no evidence ALL CAPS helps and Alibaba's
  own examples use lowercase. What helps is describing the PROCESS: not "leans
  forward" alone, but "leans forward slowly, shoulders rotating, weight shifting
  onto front foot."
- Specify direction explicitly when relevant ("walks toward camera", "walks away
  into the distance") - Wan2.2 has a documented "walks backwards" failure mode.

LIGHTING / TIME / TONE TAGS - use Alibaba's exact vocabulary so the keyframe and
motion prompts share lexicon (locks downstream coherence):
  Time of day:  Day time | Night time | Dawn time | Sunrise time
  Light source: Daylight | Practical lighting | Moonlight | Firelight | Fluorescent lighting
  Light angle:  Edge lighting | Side lighting | Top lighting | Underlighting
  Color tone:   Warm colors | Cool colors | Mixed colors

NEGATIVES are handled in the pipeline (verbatim Chinese trained negative from
shared_config.py). Do NOT add a "negative_prompt" field.

PER-SHOT FIELDS (all required):
- index: 0..5
- is_hero: true only for shot 0
- shot_type: "Wide establishing" | "Medium" | "Close-up" | "Two-shot" | "Insert" | "Tracking" etc.
- dominant_subject: "A" | "B" | ... | "scene" - drives which FLUX master keyframe seeds the shot.
- cut: true for hard cuts; false to continue visually from the previous shot.
  Shot 0 must be cut: true. Reserve at least one shot as a static medium close-up
  for face anchor.
- prompt: the full shot prompt per template above.

STORY ARC (6-shot scaffold, adapt to user request):
  shot 0 (HERO, cut: true)            - wide establishing, all main characters visible
  shot 1 (cut: false)                 - setup, A's intent or POV
  shot 2 (cut: true if scene changes) - B/C solo beat or detail insert
  shot 3 (cut: false)                 - climax: two-character moment or A-with-OBJECT
  shot 4 (cut: false)                 - STATIC medium close-up of A (face anchor)
  shot 5 (cut: false or true)         - closing wide, scene fades or A walks away

Music style: specific (e.g. "moody synthwave instrumental, 90 BPM").

Voice-over: ONE LINE PER SHOT, aligned to the 6-shot timing.
  Each shot is exactly 5.04 seconds of video. Kokoro at ~150 wpm reads ~12 words
  per 5 seconds, so each per-shot VO line should be **6-10 words**, ideally 8.
  This keeps the narration synced to the visual beats - no description before or
  after the action it covers.
  Emit the array as "vo_script_per_shot" with EXACTLY 6 entries, one per shot.
  Lines should connect into a coherent narration when concatenated, but each
  line must independently make sense as the narrator's caption for that shot.

VO language: pick the language that best suits the setting / character culture.
  Output as one-letter Kokoro lang code in the "vo_lang" field:
    a = American English (default)
    b = British English
    e = Spanish
    f = French
    h = Hindi
    i = Italian
    j = Japanese
    p = Brazilian Portuguese
    z = Mandarin Chinese
  Tokyo scene -> "j", Paris -> "f", Mumbai -> "h", Rio -> "p", Madrid -> "e",
  Rome -> "i", Beijing/Shanghai -> "z", London -> "b", anywhere else / unsure -> "a".
  EVERY entry of vo_script_per_shot MUST be written in the chosen language.

Output VALID JSON ONLY (no markdown fences, no commentary):
{
  "characters": {
    "A": "Aiko (slim Japanese woman, 28, jet-black bob, dark eyes, soft round face, "
         "long mustard yellow raincoat over white tee and dark jeans, calm steady posture)",
    "B": "..."
  },
  "story_logline": "one-sentence summary",
  "shots": [
    {"index": 0, "is_hero": true, "shot_type": "Wide establishing",
     "dominant_subject": "A", "cut": true, "prompt": "..."},
    ...
    {"index": 5, "is_hero": false, "shot_type": "Wide closing",
     "dominant_subject": "A", "cut": false, "prompt": "..."}
  ],
  "music_style": "...",
  "vo_script_per_shot": [
    "Line for shot 0, 6-10 words.",
    "Line for shot 1.",
    "Line for shot 2.",
    "Line for shot 3.",
    "Line for shot 4.",
    "Line for shot 5."
  ],
  "vo_lang": "a"
}

The "characters" map values must be plain strings, never nested objects.
"""


CRITIC_SYSTEM = """You are the SHOT CRITIC for a Wan2.2-I2V reel pipeline.
You see frames from a generated 5-second 832x480 clip and score on four 1-10 axes.
Your goal is to flag fixable issues with concrete labels, not to nitpick.

Context you must use:
- The pipeline runs Wan2.2-I2V-A14B with the verbatim Chinese trained negative
  active and CFG=3.5 (negatives DO bite here).
- umT5 cannot ground numeric character counts. Do NOT penalize "5 people not 3" if
  the foreground characters are correct and extras are blurred / distant / not
  interacting with the main subjects. Background extras in public urban scenes
  (street, station, market) are EXPECTED and add realism.
- The FLUX keyframe defines the static scene; the WAN prompt only commands motion
  and camera. Camera-induced parallax revealing previously off-frame area is the
  I2V working as intended, not a scene mismatch.

SCORING AXES:

character_match (1-10): do the named characters keep the same face, hair, wardrobe?
  10 indistinguishable from keyframe identity.
  7 clear same-person with minor face-shape drift (acceptable).
  4 cousin-of-the-character (fix needed).
  1 complete identity loss.
  For wide shots where face is small (<10% of frame), judge silhouette + wardrobe
  only and score generously.

scene_match (1-10): rendered location, lighting, time-of-day, color tone match keyframe?
  Penalize: lighting flips (day->night), location swaps, weather changes.
  Do NOT penalize: small parallax, normal motion of clothing/hair/water/leaves,
  background extras in urban scenes.

composition (1-10): does the camera execute the prompted move?
  The motion prompt names ONE camera verb. Score whether it actually happened.
  Penalty if jittery; do NOT penalize subtle motion when prompt asked for "static camera".

artifact_free (1-10, 10 = none): hand/finger artifacts, doubled limbs, morphing
  objects, neon glow leak on faces, "stylized AI look" (plastic skin, oversaturation,
  3D-render look), subtitle bleed. Stylized AI look is a real, common failure
  mode - flag it explicitly. Anamorphic flare and intentional cinematic style are
  NOT artifacts.

ISSUE LABELS (use these, so the planner knows which lever to pull on retry):
  STYLIZED_AI_LOOK     - plastic skin, oversaturation, 3D-render look, neon bloom on faces
  EXTRAS_INVADE_FRAME  - extras crossing main subjects (NOT "too many people total")
  CHARACTER_DRIFT      - identity slipping between frames
  CAMERA_IGNORED       - prompted camera move didn't happen
  OBJECT_MORPHING      - object materially changes mid-clip
  RANDOM_INTIMACY      - characters touch/hug/kiss without prompting
  NEON_GLOW_LEAK       - neon spilling onto faces or unprompted surfaces
  HAND_FINGER_ARTIFACT - extra fingers, fused hands
  WALKING_BACKWARDS    - subject walking in unintended direction
  WARDROBE_DRIFT       - clothing color/style changed mid-clip

Return JSON ONLY:
{"character_match": 8, "scene_match": 7, "composition": 9, "artifact_free": 6,
 "issues": ["STYLIZED_AI_LOOK: plastic skin on close-up face",
            "CHARACTER_DRIFT: jaw narrows in second half"],
 "overall": 7}

Be terse. Be specific. Cite frame regions when useful. Do not score on subjective taste.
"""


def parse_plan_json(raw, user_prompt):
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip().rstrip("`").strip()

    obj = json.loads(s)
    shots = obj.get("shots", [])
    if len(shots) not in (6, 9):
        raise ValueError(f"expected 6 or 9 shots, got {len(shots)}")
    if not shots[0].get("is_hero"):
        raise ValueError("shot 0 must be hero")

    raw_chars = obj.get("characters") or {}
    if not raw_chars and "character" in obj:
        # backwards compat with v1 plans
        raw_chars = {"A": str(obj["character"]).strip()}

    # Qwen3.5 has occasionally nested a stray "short_refs" object inside the
    # characters map. Filter to single-letter keys with string values only.
    chars = {}
    for k, v in raw_chars.items():
        if len(k) == 1 and k.isalpha() and isinstance(v, str):
            chars[k] = v.strip()

    # Short refs are derived from portraits, not asked from the planner anymore.
    # Take the first 12 words of each portrait, strip trailing comma if any.
    short_refs = {k: " ".join(v.split()[:12]).rstrip(",") for k, v in chars.items()}

    # ensure each shot has dominant_subject and cut fields
    for s_idx, shot in enumerate(shots):
        if "dominant_subject" not in shot:
            shot["dominant_subject"] = "A" if chars else "scene"
        if "cut" not in shot:
            shot["cut"] = (s_idx == 0)

    # per-shot VO is timing-aligned to clip beats; fall back to single string for old plans
    vo_per_shot = obj.get("vo_script_per_shot")
    if isinstance(vo_per_shot, list) and len(vo_per_shot) == len(shots):
        vo_per_shot = [str(x).strip() for x in vo_per_shot]
        vo_script = " ".join(vo_per_shot)
    else:
        vo_per_shot = None
        vo_script = str(obj.get("vo_script", "")).strip()

    return {
        "user_prompt": user_prompt,
        "characters": chars,
        "short_refs": short_refs,
        "story_logline": str(obj.get("story_logline", "")).strip(),
        "shots": shots,
        "music_style": str(obj["music_style"]).strip(),
        "vo_script": vo_script,
        "vo_script_per_shot": vo_per_shot,
        "vo_lang": str(obj.get("vo_lang", "a")).strip().lower()[:1] or "a",
    }


def expand_character_refs(plan):
    """Replace symbolic refs ("Character A", "(A)") with the SHORT 3-axis ref.

    v3 used to inline the full 50-word portrait, which crowded out motion verbs.
    v4 uses short_refs (~10 words) so the prompt stays in Wan2.2's 80-120 word
    sweet spot with motion dominant.
    """
    refs = plan.get("short_refs") or {}
    if not refs:
        return plan
    keys = sorted(refs.keys(), key=len, reverse=True)  # longest first
    for shot in plan["shots"]:
        text = shot["prompt"]
        for k in keys:
            ref = refs[k]
            text = re.sub(rf"Character\s+{re.escape(k)}\s*\([^)]*\)", ref, text)
            text = re.sub(rf"Character\s+{re.escape(k)}\b", ref, text)
            text = re.sub(rf"\({re.escape(k)}\)", ref, text)
        shot["prompt"] = text
    return plan


class DirectorAgent:
    """Qwen3.5-35B-A3B BF16 via vLLM. Loads model lazily on first plan() call.

    Same model is reused for critic(), saves the second cold load.
    """
    MODEL_ID = "Qwen/Qwen3.5-35B-A3B"

    def __init__(self):
        self._llm = None

    def _load(self):
        if self._llm is not None:
            return
        # AITER MoE is the AMD-tuned Qwen3.5 fast path. Auto-enabled on supported
        # archs in vLLM 0.17.1 when VLLM_ROCM_USE_AITER=1 is set before init.
        import os as _os
        _os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
        from vllm import LLM
        log.info(f"loading {self.MODEL_ID} (~5 min cold load on MI300X)")
        self._llm = LLM(
            model=self.MODEL_ID,
            tensor_parallel_size=1,
            # 0.70 instead of 0.90: ROCm allocator caches ~30 GB after Wan2.2 unload
            # and doesn't return it to OS, so vLLM at 0.90 (172 GB) fails with
            # "Free memory ... less than desired GPU memory utilization" on second
            # critic call. 0.70 (135 GB) leaves headroom for the cached pool.
            gpu_memory_utilization=0.70,
            max_model_len=8192,
            trust_remote_code=True,
            enforce_eager=True,   # CUDA graphs broken on ROCm 7.0 + vLLM 0.20 nightly
            dtype="bfloat16",
        )

    def plan(self, user_prompt):
        self._load()
        from vllm import SamplingParams
        # v4 schema (short_refs + dominant_subject + cut + 9 enriched shots)
        # blew past 2048 on the busker reel (truncated at char 7835). 4096 is safe.
        sp = SamplingParams(max_tokens=4096, temperature=0.7, top_p=0.95)
        msgs = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f"User prompt: {user_prompt}"},
        ]
        out = self._llm.chat(msgs, sampling_params=sp,
                             chat_template_kwargs={"enable_thinking": False})
        raw = out[0].outputs[0].text
        return parse_plan_json(raw, user_prompt)

    def critique(self, image_paths, shot_prompt, character_portrait):
        """Score the clip against the shot's intent.

        image_paths: list of frame paths sampled across the clip (or a single str path).
        Multiple frames let the critic judge the whole motion arc, not just one moment.
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        self._load()
        from vllm import SamplingParams
        from PIL import Image
        sp = SamplingParams(max_tokens=512, temperature=0.0)

        content = []
        for p in image_paths:
            content.append({"type": "image_pil", "image_pil": Image.open(p).convert("RGB")})
        content.append({"type": "text", "text":
            f"Shot prompt: {shot_prompt}\nExpected character: {character_portrait}\n"
            f"You're seeing {len(image_paths)} frames sampled across one short clip "
            f"(start, mid-early, mid-late, end). Score the OVERALL clip and return JSON only."})

        msgs = [
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": content},
        ]
        out = self._llm.chat(msgs, sampling_params=sp,
                             chat_template_kwargs={"enable_thinking": False})
        raw = out[0].outputs[0].text.strip()
        # strip code fence if present
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip().rstrip("`").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            log.warning(f"critic returned invalid JSON, raw: {raw[:200]}")
            return {"overall": 5, "issues": ["critic parse failed"]}

    def unload(self):
        if self._llm is None:
            return
        del self._llm
        self._llm = None
        gc.collect()
        import torch
        torch.cuda.empty_cache()


# CLI for one-shot planning (used by Gradio app via subprocess)
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python director.py 'user prompt'", file=sys.stderr)
        sys.exit(1)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    d = DirectorAgent()
    plan = d.plan(sys.argv[1])
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    out_path = "/root/outputs/director_plan.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    log.info(f"saved: {out_path}")
