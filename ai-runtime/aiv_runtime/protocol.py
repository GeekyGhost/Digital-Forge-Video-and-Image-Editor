from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class WorkerRequest(BaseModel):
    request_id: str
    method: Literal["health", "capabilities", "run", "cancel", "shutdown"]
    params: dict[str, Any] = Field(default_factory=dict)


class WorkerResponse(BaseModel):
    request_id: str
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None

