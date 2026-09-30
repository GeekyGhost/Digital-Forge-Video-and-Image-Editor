from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from .protocol import WorkerRequest, WorkerResponse


CAPABILITIES = {
    "ltx": ["keyframe-interpolation", "video-to-audio", "retake", "image-to-video"],
    "utilities": ["transcribe", "segment", "interpolate", "speech-enhance"],
    "editor": ["instruction-edit", "reference-edit"],
}


class Worker:
    def __init__(self) -> None:
        self.root = Path(os.environ.get("AIV_ROOT", Path.cwd()))
        self.models = Path(os.environ.get("AIV_MODELS", self.root / "models"))
        self.cancelled: set[str] = set()

    def dispatch(self, request: WorkerRequest) -> dict[str, Any]:
        if request.method == "health":
            return {
                "status": "ready",
                "python": platform.python_version(),
                "pid": os.getpid(),
                "modelsRoot": str(self.models),
            }
        if request.method == "capabilities":
            return {"capabilities": CAPABILITIES}
        if request.method == "cancel":
            job_id = str(request.params.get("jobId", ""))
            self.cancelled.add(job_id)
            return {"jobId": job_id, "state": "cancelled"}
        if request.method == "run":
            capability = str(request.params.get("capability", ""))
            if not any(capability in values for values in CAPABILITIES.values()):
                raise ValueError(f"unsupported capability: {capability}")
            # Model imports are deliberately lazy. Adapters own loading, offload,
            # telemetry, and cleanup so the worker can be recycled after each heavy job.
            return {
                "state": "accepted",
                "capability": capability,
                "adapter": request.params.get("adapter", "auto"),
                "vramLeaseMiB": min(int(request.params.get("vramLeaseMiB", 14336)), 14336),
            }
        if request.method == "shutdown":
            return {"shutdown": True}
        raise ValueError(f"unknown method: {request.method}")


def respond(response: WorkerResponse) -> None:
    sys.stdout.write(response.model_dump_json() + "\n")
    sys.stdout.flush()


def main() -> int:
    worker = Worker()
    for line in sys.stdin:
        try:
            request = WorkerRequest.model_validate_json(line)
            result = worker.dispatch(request)
            respond(WorkerResponse(request_id=request.request_id, ok=True, result=result))
            if request.method == "shutdown":
                return 0
        except Exception as error:  # Worker boundary must report, never leak a traceback to stdout.
            request_id = "unknown"
            try:
                request_id = json.loads(line).get("request_id", "unknown")
            except json.JSONDecodeError:
                pass
            respond(WorkerResponse(request_id=request_id, ok=False, error=str(error)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

