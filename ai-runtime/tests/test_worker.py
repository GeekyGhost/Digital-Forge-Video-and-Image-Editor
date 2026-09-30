from aiv_runtime.protocol import WorkerRequest
from aiv_runtime.worker import Worker


def test_worker_reports_health() -> None:
    response = Worker().dispatch(WorkerRequest(request_id="1", method="health"))
    assert response["status"] == "ready"


def test_vram_lease_is_capped() -> None:
    response = Worker().dispatch(
        WorkerRequest(
            request_id="2",
            method="run",
            params={"capability": "transcribe", "vramLeaseMiB": 99_999},
        )
    )
    assert response["vramLeaseMiB"] == 14_336

