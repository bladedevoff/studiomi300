# Stage events emitted from the pipeline to drive the API stream.
# Lines are written to <out>/events.jsonl AND echoed to stdout with EVENT:: prefix
# so the server can parse them live without tailing a file.
import json
import time
import threading
from pathlib import Path

_lock = threading.Lock()
_path = None


def init(out_dir):
    global _path
    _path = Path(out_dir) / "events.jsonl"
    _path.parent.mkdir(parents=True, exist_ok=True)
    _path.touch(exist_ok=True)


def emit(stage, **fields):
    ev = {"stage": stage, "ts": round(time.time(), 3), **fields}
    line = json.dumps(ev, ensure_ascii=False, default=str)
    print("EVENT::" + line, flush=True)
    if _path is not None:
        with _lock, open(_path, "a") as f:
            f.write(line + "\n")
