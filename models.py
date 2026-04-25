from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    class Action(BaseModel):
        model_config = {"extra": "forbid"}

    class Observation(BaseModel):
        reward: float = Field(default=0.0, ge=0.0, le=1.0)
        done: bool = False

    class State(BaseModel):
        episode_id: str = ""
        step_count: int = 0

from pydantic import Field


class AdaptAction(Action):
    code: str = Field(..., min_length=1, description="Python code to execute.")


class AdaptObservation(Observation):
    problem_id: str = Field(default="", description="Current problem identifier.")
    difficulty: str = Field(default="", description="Current curriculum difficulty tier.")
    problem: str = Field(default="", description="Problem statement shown to the agent.")
    input_format: str = Field(default="", description="Expected stdin format.")
    constraints: str = Field(default="", description="Problem constraints.")
    examples: list[dict[str, str]] = Field(default_factory=list)
    visible_tests: list[dict[str, str]] = Field(default_factory=list)
    feedback: str = Field(default="", description="Human-readable execution feedback.")
    pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    visible_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    hidden_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    syntax_valid: bool = Field(default=True)
    execution_status: str = Field(default="not_run")
    timeout_count: int = Field(default=0, ge=0)
    runtime_error_count: int = Field(default=0, ge=0)
    format_compliance: float = Field(default=0.0, ge=0.0, le=1.0)
    reward_components: dict[str, float] = Field(default_factory=dict)


class AdaptState(State):
    problem_id: str = Field(default="")
    difficulty: str = Field(default="")
    last_reward: float = Field(default=0.0)
    last_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    last_feedback: str = Field(default="")
    recent_metrics: dict[str, Any] = Field(default_factory=dict)
