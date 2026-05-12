# FastAPI wrapper. asyncio.Lock = single-MI300X FIFO queue. token via STUDIO_API_TOKEN.
import os
import re
import json
import time
import uuid
import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

log = logging.getLogger("studiomi300.server")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).parent
JOBS_DIR = Path(os.environ.get("STUDIO_JOBS_DIR", "/root/outputs/api_jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)
TOKEN = os.environ.get("STUDIO_API_TOKEN", "")

GPU_LOCK = asyncio.Lock()
# manual FIFO tracker so each waiting job can see its queue position. asyncio.Lock
# doesn't expose its internal waiter list, so jobs append their id here on entry
# and remove it as soon as the lock is acquired.
QUEUE = []


class JobIn(BaseModel):
    prompt: str = Field(..., min_length=20, max_length=2000)
    use_critic: bool = True
    mode: str = Field("full", pattern="^(full|demo)$")


def auth(x_api_token: str = Header(default="")):
    if TOKEN and x_api_token != TOKEN:
        raise HTTPException(401, "bad token")


# job_ids are uuid4().hex[:12] — lock down all path-building to that exact format
# so user-supplied path components can't escape JOBS_DIR via traversal.
_JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")


def _check_job_id(job_id):
    if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
        raise HTTPException(400, "invalid job_id")


def _path(job_id):
    _check_job_id(job_id)
    return JOBS_DIR / f"{job_id}.json"


def _save(job_id, meta):
    _path(job_id).write_text(json.dumps(meta))


def _load(job_id):
    p = _path(job_id)
    return json.loads(p.read_text()) if p.exists() else None


def _job_dir(job_id):
    _check_job_id(job_id)
    d = JOBS_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _run_job(job_id, prompt, use_critic, mode):
    out = _job_dir(job_id)
    events_file = out / "events.jsonl"
    meta = _load(job_id) or {}

    QUEUE.append(job_id)
    async with GPU_LOCK:
        try: QUEUE.remove(job_id)
        except ValueError: pass
        meta.update({"status": "running", "started": time.time(), "stage": "starting", "queue_position": 0})
        _save(job_id, meta)
        log.info(f"job {job_id} started ({mode})")

        if mode == "demo":
            cmd = [
                "python", "-u", str(ROOT / "quick_demo.py"),
                "--prompt", prompt,
                "--out", str(out),
            ]
        else:
            cmd = [
                "python", "-u", str(ROOT / "generate.py"),
                "--prompt", prompt,
                "--out", str(out),
            ]
            if use_critic:
                cmd.append("--critic")

        env = os.environ.copy()
        env.setdefault("STUDIOMI_AITER_FP8", "0")
        env.setdefault("VLLM_ROCM_USE_AITER", "1")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(ROOT),
            env=env,
        )

        log_tail = []
        async for raw in proc.stdout:
            line = raw.decode("utf-8", "ignore").rstrip()
            if not line.strip():
                continue
            log_tail.append(line)
            log_tail = log_tail[-50:]
            if line.startswith("EVENT::"):
                # generate.py also writes events.jsonl directly, just update meta here
                try:
                    ev = json.loads(line[len("EVENT::"):])
                    meta["stage"] = ev.get("stage", meta.get("stage"))
                    meta["last_event"] = ev
                    _save(job_id, meta)
                except Exception:
                    pass
            else:
                meta["log_tail"] = log_tail
                _save(job_id, meta)
        await proc.wait()

        final = out / ("demo.mp4" if mode == "demo" else "reel_final.mp4")
        ok = proc.returncode == 0 and final.exists()
        meta["status"] = "done" if ok else "failed"
        meta["stage"] = "done" if ok else meta.get("stage", "failed")
        meta["video"] = str(final) if ok else None
        meta["finished"] = time.time()
        meta["log_tail"] = log_tail[-50:]
        _save(job_id, meta)
        log.info(f"job {job_id} {meta['status']} in {meta['finished']-meta['started']:.0f}s")


app = FastAPI(title="StudioMI300 API", version="0.2")


@app.get("/health")
async def health():
    return {"status": "ok", "gpu_busy": GPU_LOCK.locked(), "time": time.time()}


@app.post("/jobs", dependencies=[Depends(auth)])
async def submit(body: JobIn, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex[:12]
    meta = {
        "id": job_id,
        "prompt": body.prompt,
        "use_critic": body.use_critic,
        "mode": body.mode,
        "status": "queued",
        "submitted": time.time(),
        "stage": "queued",
    }
    _save(job_id, meta)
    bg.add_task(_run_job, job_id, body.prompt, body.use_critic, body.mode)
    return {"job_id": job_id, "status": "queued", "mode": body.mode}


@app.api_route("/demos/{job_id}.mp4", methods=["GET", "HEAD"])
async def demo_video(job_id: str):
    # public mp4 for <video> embed: inline disposition + HEAD support so Gradio
    # mime sniffing works
    meta = _load(job_id)
    if meta is None:
        raise HTTPException(404, "no such job")
    if meta.get("mode") != "demo" or meta.get("status") != "done":
        raise HTTPException(404, "not a completed demo")
    return FileResponse(
        meta["video"],
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="demo_{job_id}.mp4"'},
    )


@app.get("/demos")
async def demos(limit: int = 50):
    # public, no auth: list of completed demo jobs newest-first
    rows = []
    for p in sorted(JOBS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            j = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if j.get("mode") != "demo" or j.get("status") != "done":
            continue
        rows.append({
            "id": j["id"],
            "prompt": j.get("prompt", ""),
            "video": f"/demos/{j['id']}.mp4",
            "submitted": j.get("submitted"),
            "finished": j.get("finished"),
            "duration_s": round((j.get("finished") or 0) - (j.get("started") or 0), 1),
        })
        if len(rows) >= limit:
            break
    return rows


@app.get("/jobs/{job_id}", dependencies=[Depends(auth)])
async def status(job_id: str):
    meta = _load(job_id)
    if meta is None:
        raise HTTPException(404, "no such job")
    if meta.get("status") == "queued" and job_id in QUEUE:
        meta["queue_position"] = QUEUE.index(job_id) + 1
        meta["queue_size"] = len(QUEUE)
    return meta


@app.get("/jobs/{job_id}/events", dependencies=[Depends(auth)])
async def events(job_id: str):
    _check_job_id(job_id)
    p = JOBS_DIR / job_id / "events.jsonl"
    if not p.exists():
        raise HTTPException(404, "no events yet")
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


@app.get("/jobs/{job_id}/stream")
async def stream(job_id: str):
    _check_job_id(job_id)
    p = JOBS_DIR / job_id / "events.jsonl"

    async def gen():
        seen = 0
        # poll loop — yields each new event line as SSE; bails when job is done/failed
        while True:
            meta = _load(job_id)
            if meta is None:
                yield f"data: {json.dumps({'stage': 'unknown_job'})}\n\n"
                return
            if p.exists():
                with open(p) as f:
                    lines = f.readlines()
                for line in lines[seen:]:
                    line = line.strip()
                    if line:
                        yield f"data: {line}\n\n"
                seen = len(lines)
            if meta.get("status") in ("done", "failed"):
                yield f"data: {json.dumps({'stage':'_close','status':meta['status']})}\n\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")


def _file_or_404(path: Path, mime: str):
    if not path.exists():
        raise HTTPException(404, f"not ready: {path.name}")
    return FileResponse(str(path), media_type=mime, filename=path.name)


@app.get("/jobs/{job_id}/plan", dependencies=[Depends(auth)])
async def plan(job_id: str):
    _check_job_id(job_id)
    p = JOBS_DIR / job_id / "plan_expanded.json"
    if not p.exists():
        p = JOBS_DIR / job_id / "plan.json"
    if not p.exists():
        raise HTTPException(404, "plan not ready")
    return json.loads(p.read_text())


_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


@app.get("/jobs/{job_id}/master/{name}", dependencies=[Depends(auth)])
async def master(job_id: str, name: str):
    _check_job_id(job_id)
    if not _NAME_RE.match(name):
        raise HTTPException(400, "invalid name")
    return _file_or_404(JOBS_DIR / job_id / f"master_{name}.png", "image/png")


@app.get("/jobs/{job_id}/keyframe/{idx}", dependencies=[Depends(auth)])
async def keyframe(job_id: str, idx: int):
    _check_job_id(job_id)
    return _file_or_404(JOBS_DIR / job_id / f"keyframe_{idx:02d}.png", "image/png")


@app.get("/jobs/{job_id}/clip/{idx}", dependencies=[Depends(auth)])
async def clip(job_id: str, idx: int):
    _check_job_id(job_id)
    return _file_or_404(JOBS_DIR / job_id / f"clip_{idx:02d}.mp4", "video/mp4")


@app.get("/jobs/{job_id}/music", dependencies=[Depends(auth)])
async def music(job_id: str):
    _check_job_id(job_id)
    return _file_or_404(JOBS_DIR / job_id / "music.wav", "audio/wav")


@app.get("/jobs/{job_id}/vo/{idx}", dependencies=[Depends(auth)])
async def vo_chunk(job_id: str, idx: int):
    _check_job_id(job_id)
    p = JOBS_DIR / job_id / f"vo_{idx:02d}.wav"
    if not p.exists():
        # fallback to single-track legacy
        p = JOBS_DIR / job_id / "vo.wav"
    return _file_or_404(p, "audio/wav")


@app.get("/jobs/{job_id}/video", dependencies=[Depends(auth)])
async def video(job_id: str):
    meta = _load(job_id)
    if meta is None:
        raise HTTPException(404, "no such job")
    if meta.get("status") != "done":
        raise HTTPException(409, f"not ready, status={meta.get('status')}")
    return FileResponse(meta["video"], media_type="video/mp4", filename=f"{job_id}.mp4")


@app.get("/jobs", dependencies=[Depends(auth)])
async def list_jobs():
    rows = []
    for p in sorted(JOBS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            pass
    return rows[:50]
