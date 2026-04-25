from __future__ import annotations

import argparse
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel

from env.adapt_env import AdaptEnvironment
from env.test_cases import load_problem_bank
from models import AdaptAction, AdaptObservation, AdaptState

ENV_NAME = "adapt_dsa_tutor"
ENV_DESCRIPTION = (
    "RL environment for DSA code generation with hidden tests, tiered problems, "
    "and verifier-aware reward shaping."
)
TASKS = [
    {
        "name": problem["problem_id"],
        "difficulty": problem["difficulty"],
        "description": problem["problem"],
    }
    for problem in load_problem_bank()
]

app = FastAPI(title="ADAPT DSA Tutor OpenEnv", version="0.2.0")
ENV = AdaptEnvironment()


class ResetRequest(BaseModel):
    seed: int | None = None
    episode_id: str | None = None
    problem_id: str | None = None
    difficulty: str | None = None


def _metadata() -> dict[str, Any]:
    return {
        "name": ENV_NAME,
        "description": ENV_DESCRIPTION,
        "version": "0.2.0",
        "tasks": TASKS,
        "mode": "simulation",
    }


@app.get("/")
def root() -> dict[str, Any]:
    payload = _metadata()
    payload["status"] = "ok"
    return payload


@app.get("/web", include_in_schema=False)
def web_root() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=307)


@app.get("/web/", include_in_schema=False)
def web_root_slash() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=307)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return _metadata()


@app.get("/tasks")
def list_tasks() -> dict[str, Any]:
    return {"tasks": TASKS}


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "action": AdaptAction.model_json_schema(),
        "observation": AdaptObservation.model_json_schema(),
        "state": AdaptState.model_json_schema(),
    }


@app.post("/mcp")
def mcp(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id"),
        "error": {
            "code": -32601,
            "message": "MCP methods are not implemented for this environment.",
        },
    }


@app.post("/reset")
def reset(request: ResetRequest | None = None) -> dict[str, Any]:
    effective_request = request or ResetRequest()
    observation = ENV.reset(
        seed=effective_request.seed,
        episode_id=effective_request.episode_id,
        problem_id=effective_request.problem_id,
        difficulty=effective_request.difficulty,
    )
    return observation.model_dump()


@app.post("/step")
async def step(request: Request) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object.")

    raw_action = payload.get("action", payload)
    try:
        effective_action = AdaptAction.model_validate(raw_action)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action payload: {exc}") from exc

    observation = ENV.step(effective_action)
    return {
        "observation": observation.model_dump(),
        "reward": float(observation.reward),
        "done": bool(observation.done),
        "info": {
            "feedback": observation.feedback,
            "pass_rate": observation.pass_rate,
            "execution_status": observation.execution_status,
        },
    }


@app.get("/state")
def state() -> dict[str, Any]:
    if not ENV.problem:
        ENV.reset()
    return ENV.state.model_dump()


def main(host: str | None = None, port: int | None = None) -> None:
    if host is None or port is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=7860)
        args = parser.parse_args()
        host = args.host if host is None else host
        port = args.port if port is None else port
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
