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


class AdaptAction(Action):
    session_id: str = Field(
        default="",
        description="Environment session id for server-routed calls.",
    )
    code: str = Field(..., min_length=1, description="Python code to execute.")


class AdaptObservation(Observation):
    session_id: str = Field(default="", description="Session id for the active environment instance.")
    problem_id: str = Field(default="", description="Current problem identifier.")
    problem_type: str = Field(default="", description="Current generated problem family.")
    difficulty: str = Field(default="", description="Current curriculum difficulty tier.")
    attempt_number: int = Field(default=0, ge=0, description="1-indexed attempt number within the episode.")
    max_steps: int = Field(default=3, ge=1, description="Maximum attempts allowed for the episode.")
    problem: str = Field(default="", description="Problem statement shown to the agent.")
    input_format: str = Field(default="", description="Expected stdin format.")
    constraints: str = Field(default="", description="Problem constraints.")
    feedback: str = Field(default="", description="Human-readable execution feedback.")
    pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    visible_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    hidden_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    syntax_valid: bool = Field(default=True)
    execution_status: str = Field(default="not_run")
    timeout_count: int = Field(default=0, ge=0)
    runtime_error_count: int = Field(default=0, ge=0)
    invalid_output_count: int = Field(default=0, ge=0)
    wrong_answer_count: int = Field(default=0, ge=0)
    format_compliance: float = Field(default=0.0, ge=0.0, le=1.0)
    reward_components: dict[str, float] = Field(default_factory=dict)
    generator_reward_signal: float = Field(default=0.0)


class AdaptState(State):
    session_id: str = Field(default="")
    problem_id: str = Field(default="")
    problem_type: str = Field(default="")
    difficulty: str = Field(default="")
    generator_mode: str = Field(default="heuristic")
    max_steps: int = Field(default=3, ge=1)
    generated_problem: dict[str, Any] = Field(default_factory=dict)
    last_reward: float = Field(default=0.0)
    last_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    last_feedback: str = Field(default="")
    last_execution_status: str = Field(default="ready")
    generator_reward_signal: float = Field(default=0.0)
    history: dict[str, Any] = Field(default_factory=dict)
    recent_metrics: dict[str, Any] = Field(default_factory=dict)
