from __future__ import annotations

from typing import Any

import httpx

from models import AdaptAction


class AdaptEnvClient:
    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=30.0)
        self.session_id: str | None = None

    def close(self) -> None:
        self._client.close()

    def reset(self, **params: Any) -> dict[str, Any]:
        response = self._client.post("/reset", json=params)
        response.raise_for_status()
        payload = response.json()
        self.session_id = payload.get("session_id")
        return payload

    def step(self, code: str) -> dict[str, Any]:
        if not self.session_id:
            raise RuntimeError("Call reset() before step() so the client has a session_id.")
        response = self._client.post(
            "/step",
            json=AdaptAction(session_id=self.session_id, code=code).model_dump(),
        )
        response.raise_for_status()
        return response.json()

    def state(self) -> dict[str, Any]:
        if not self.session_id:
            raise RuntimeError("Call reset() before state() so the client has a session_id.")
        response = self._client.get("/state", params={"session_id": self.session_id})
        response.raise_for_status()
        return response.json()

    def train(self, **params: Any) -> dict[str, Any]:
        response = self._client.post("/train", json=params)
        response.raise_for_status()
        return response.json()

    def train_status(self) -> dict[str, Any]:
        response = self._client.get("/train/status")
        response.raise_for_status()
        return response.json()

    def model_status(self) -> dict[str, Any]:
        response = self._client.get("/model/status")
        response.raise_for_status()
        return response.json()

    def run_trained_policy(self, **params: Any) -> dict[str, Any]:
        response = self._client.post("/run-trained-policy", json=params)
        response.raise_for_status()
        return response.json()

    def generate_code(self, **params: Any) -> dict[str, Any]:
        response = self._client.post("/generate-code", json=params)
        response.raise_for_status()
        return response.json()


__all__ = ["AdaptEnvClient"]
