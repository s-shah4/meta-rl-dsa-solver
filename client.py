from __future__ import annotations

from typing import Any

import httpx

from models import AdaptAction


class AdaptEnvClient:
    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def close(self) -> None:
        self._client.close()

    def reset(self, **params: Any) -> dict[str, Any]:
        response = self._client.post("/reset", json=params)
        response.raise_for_status()
        return response.json()

    def step(self, code: str) -> dict[str, Any]:
        response = self._client.post("/step", json={"action": AdaptAction(code=code).model_dump()})
        response.raise_for_status()
        return response.json()

    def state(self) -> dict[str, Any]:
        response = self._client.get("/state")
        response.raise_for_status()
        return response.json()


__all__ = ["AdaptEnvClient"]
