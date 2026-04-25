from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import uuid4

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel

from env.adapt_env import AdaptEnvironment
from env.test_cases import load_problem_bank
from models import AdaptAction, AdaptObservation, AdaptState
from server.runtime import SpaceTrainingManager

ENV_NAME = "adapt-dsa-tutor"
ENV_DESCRIPTION = (
    "Adversarial DSA Programming Tutor - RL environment for training LLMs to solve "
    "algorithmic problems through adaptive curriculum and self-repair."
)
ENV_VERSION = "0.4.0"
SESSION_TTL = timedelta(minutes=30)
SESSIONS: dict[str, AdaptEnvironment] = {}
SESSION_LAST_ACCESSED: dict[str, datetime] = {}
TRAINING_MANAGER = SpaceTrainingManager()
TASKS = [
    {
        "name": problem["problem_id"],
        "difficulty": problem["difficulty"],
        "description": problem["problem"],
    }
    for problem in load_problem_bank()
]

app = FastAPI(title="ADAPT DSA Tutor OpenEnv", version=ENV_VERSION)


class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    problem_id: Optional[str] = None
    difficulty: Optional[str] = None


class TrainRequest(BaseModel):
    preset: str = "l4"
    model_name: Optional[str] = None
    output_dir: Optional[str] = None
    dataset_size: Optional[int] = None
    max_steps: Optional[int] = None
    batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    num_generations: Optional[int] = None
    load_in_4bit: Optional[bool] = None
    gradient_checkpointing: Optional[bool] = None
    evaluation_episodes: Optional[int] = None
    baseline_eval: Optional[bool] = None
    generator_mode: Optional[str] = None
    use_dataset: bool = False
    dataset_name: str = "deepmind/code_contests"
    dataset_max_problems: int = 5000
    disable_wandb: Optional[bool] = None
    save_merged_model: Optional[bool] = None


class RunTrainedPolicyRequest(BaseModel):
    problem_id: Optional[str] = None
    difficulty: Optional[str] = None
    max_new_tokens: int = 512


class GenerateCodeRequest(BaseModel):
    problem: str
    input_format: str
    constraints: str
    feedback: Optional[str] = None
    problem_id: str = "custom_problem"
    problem_type: str = "custom"
    difficulty: str = "custom"
    attempt_number: int = 1
    max_steps: int = 1
    max_new_tokens: int = 512


def _metadata() -> dict[str, Any]:
    return {
        "name": ENV_NAME,
        "description": ENV_DESCRIPTION,
        "version": ENV_VERSION,
        "tasks": TASKS,
        "mode": "simulation",
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _cleanup_sessions() -> None:
    now = _utc_now()
    expired = [
        session_id
        for session_id, last_seen in SESSION_LAST_ACCESSED.items()
        if now - last_seen > SESSION_TTL
    ]
    for session_id in expired:
        SESSIONS.pop(session_id, None)
        SESSION_LAST_ACCESSED.pop(session_id, None)


def _touch_session(session_id: str) -> None:
    SESSION_LAST_ACCESSED[session_id] = _utc_now()


def _require_session(session_id: str) -> AdaptEnvironment:
    _cleanup_sessions()
    env = SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown or expired session_id: {session_id}")
    _touch_session(session_id)
    return env


@app.on_event("startup")
def startup() -> None:
    TRAINING_MANAGER.load_latest_model()


@app.get("/")
def root() -> dict[str, Any]:
    _cleanup_sessions()
    payload = _metadata()
    payload["status"] = "ok"
    payload["active_sessions"] = len(SESSIONS)
    payload["training"] = TRAINING_MANAGER.status_payload()
    payload["model"] = TRAINING_MANAGER.model_status_payload()
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
def health() -> dict[str, Any]:
    _cleanup_sessions()
    return {
        "status": "healthy",
        "active_sessions": len(SESSIONS),
        "training": TRAINING_MANAGER.status_payload()["status"],
        "model_loaded": TRAINING_MANAGER.model_status_payload()["loaded"],
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    _cleanup_sessions()
    return _metadata()


@app.get("/tasks")
def list_tasks() -> dict[str, Any]:
    _cleanup_sessions()
    return {"tasks": TASKS}


@app.get("/schema")
def schema() -> dict[str, Any]:
    _cleanup_sessions()
    return {
        "action": AdaptAction.model_json_schema(),
        "observation": AdaptObservation.model_json_schema(),
        "state": AdaptState.model_json_schema(),
    }


@app.get("/train/status")
def train_status() -> dict[str, Any]:
    return TRAINING_MANAGER.status_payload()


@app.get("/model/status")
def model_status() -> dict[str, Any]:
    return TRAINING_MANAGER.model_status_payload()


@app.post("/train")
def train(request: Optional[TrainRequest] = None) -> dict[str, Any]:
    try:
        return TRAINING_MANAGER.start_training((request or TrainRequest()).model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/run-trained-policy")
def run_trained_policy(request: Optional[RunTrainedPolicyRequest] = None) -> dict[str, Any]:
    effective_request = request or RunTrainedPolicyRequest()
    try:
        return TRAINING_MANAGER.run_trained_policy(
            problem_id=effective_request.problem_id,
            difficulty=effective_request.difficulty,
            max_new_tokens=effective_request.max_new_tokens,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"run-trained-policy failed: {exc}") from exc


@app.post("/generate-code")
def generate_code(request: GenerateCodeRequest) -> dict[str, Any]:
    try:
        return TRAINING_MANAGER.generate_code(
            problem=request.problem,
            input_format=request.input_format,
            constraints=request.constraints,
            feedback=request.feedback,
            problem_id=request.problem_id,
            problem_type=request.problem_type,
            difficulty=request.difficulty,
            attempt_number=request.attempt_number,
            max_steps=request.max_steps,
            max_new_tokens=request.max_new_tokens,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generate-code failed: {exc}") from exc


@app.post("/mcp")
def mcp(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    _cleanup_sessions()
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id"),
        "error": {
            "code": -32601,
            "message": "MCP methods are not implemented for this environment.",
        },
    }


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None) -> dict[str, Any]:
    _cleanup_sessions()
    effective_request = request or ResetRequest()
    session_id = effective_request.session_id or str(uuid4())
    env = AdaptEnvironment(session_id=session_id)
    SESSIONS[session_id] = env
    _touch_session(session_id)
    observation = env.reset(
        session_id=session_id,
        seed=effective_request.seed,
        episode_id=effective_request.episode_id,
        problem_id=effective_request.problem_id,
        difficulty=effective_request.difficulty,
    )
    return observation.model_dump()


@app.post("/step")
async def step(request: Request) -> dict[str, Any]:
    _cleanup_sessions()
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object.")

    raw_action = payload.get("action", payload)
    try:
        effective_action = AdaptAction.model_validate(raw_action)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action payload: {exc}") from exc

    if not effective_action.session_id:
        raise HTTPException(status_code=422, detail="`session_id` is required in the /step request body.")

    env = _require_session(effective_action.session_id)
    observation = env.step(effective_action)
    return {
        "observation": observation.model_dump(),
        "reward": float(observation.reward),
        "done": bool(observation.done),
        "info": {
            "session_id": observation.session_id,
            "feedback": observation.feedback,
            "pass_rate": observation.pass_rate,
            "visible_pass_rate": observation.visible_pass_rate,
            "execution_status": observation.execution_status,
        },
    }


@app.get("/state")
def state(session_id: str = Query(..., description="Session id returned from /reset.")) -> dict[str, Any]:
    env = _require_session(session_id)
    if not env.problem:
        env.reset(session_id=session_id)
    return env.state.model_dump()


def main(host: Optional[str] = None, port: Optional[int] = None) -> None:
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
