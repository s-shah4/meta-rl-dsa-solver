from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

server_app = importlib.import_module("server.app")


def main() -> None:
    client = TestClient(server_app.app)

    root = client.get("/")
    assert root.status_code == 200
    assert "training" in root.json()
    assert "model" in root.json()

    model_status = client.get("/model/status")
    assert model_status.status_code == 200
    assert "loaded" in model_status.json()

    train_status = client.get("/train/status")
    assert train_status.status_code == 200
    assert "status" in train_status.json()
    assert "completed_steps" in train_status.json()
    assert "remaining_steps" in train_status.json()
    assert "phase" in train_status.json()
    assert "started_at" in train_status.json()
    assert "finished_at" in train_status.json()
    assert "elapsed_seconds" in train_status.json()
    assert "timing_summary" in train_status.json()
    assert "run_manifest_path" in train_status.json()
    assert "events_path" in train_status.json()
    assert "latest_checkpoint_path" in train_status.json()
    assert "logs_deleted_from_space" in train_status.json()

    reset = client.post("/reset", json={"difficulty": "easy", "problem_id": "sum_even_numbers"})
    assert reset.status_code == 200
    session_id = reset.json()["session_id"]

    step = client.post(
        "/step",
        json={
            "session_id": session_id,
            "code": "n=int(input())\nnums=list(map(int,input().split()))\nprint(sum(x for x in nums if x % 2 == 0))",
        },
    )
    assert step.status_code == 200
    assert step.json()["done"] is True

    state = client.get("/state", params={"session_id": session_id})
    assert state.status_code == 200
    assert state.json()["session_id"] == session_id

    no_model = client.post("/run-trained-policy", json={"difficulty": "easy"})
    assert no_model.status_code == 409

    with patch.object(
        server_app.TRAINING_MANAGER,
        "generate_code",
        return_value={"code": "print(42)", "completion": "print(42)", "problem_id": "custom_problem"},
    ):
        generate = client.post(
            "/generate-code",
            json={
                "problem": "Print 42.",
                "input_format": "No input.",
                "constraints": "None",
            },
        )
        assert generate.status_code == 200
        assert generate.json()["code"] == "print(42)"

    with patch.object(server_app.TRAINING_MANAGER, "start_training", return_value={"status": "running", "run_id": "demo"}):
        train = client.post("/train", json={})
        assert train.status_code == 200
        assert train.json()["status"] == "running"

    with patch.object(
        server_app.TRAINING_MANAGER,
        "start_training",
        side_effect=RuntimeError("A training run is already in progress."),
    ):
        conflict = client.post("/train", json={})
        assert conflict.status_code == 409

    print("Space API smoke tests passed")


if __name__ == "__main__":
    main()
