from __future__ import annotations

import json
import shutil
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class TraceArtifactLogger:
    run_id: str
    output_dir: Path
    training_config: dict[str, Any]
    model_identifiers: dict[str, Any]
    system_prompt: str
    checkpoint_interval_steps: int = 10
    schema_version: str = "1.0"
    logs_dir: Path = field(init=False)
    manifest_path: Path = field(init=False)
    events_path: Path = field(init=False)
    latest_checkpoint_path: Path = field(init=False)
    run_summary_path: Path = field(init=False)
    checkpoint_paths: list[Path] = field(default_factory=list, init=False)
    _last_checkpoint_step: int = field(default=0, init=False, repr=False)
    _latest_event: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _recent_rewards: deque[float] = field(default_factory=lambda: deque(maxlen=25), init=False, repr=False)
    _recent_pass_rates: deque[float] = field(default_factory=lambda: deque(maxlen=25), init=False, repr=False)
    _recent_efficiency_scores: deque[float] = field(default_factory=lambda: deque(maxlen=25), init=False, repr=False)
    _latest_progress: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.logs_dir / "run_manifest.json"
        self.events_path = self.logs_dir / "events.jsonl"
        self.latest_checkpoint_path = self.logs_dir / "latest_checkpoint.json"
        self.run_summary_path = self.logs_dir / "run_summary.json"
        manifest = {
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "started_at": _utc_now_iso(),
            "training_config": self.training_config,
            "model_identifiers": self.model_identifiers,
            "system_prompt": self.system_prompt,
            "checkpoint_interval_steps": int(max(self.checkpoint_interval_steps, 1)),
        }
        self.manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")

    def artifact_paths(self) -> dict[str, Any]:
        return {
            "logs_dir": str(self.logs_dir),
            "run_manifest_path": str(self.manifest_path),
            "events_path": str(self.events_path),
            "latest_checkpoint_path": str(self.latest_checkpoint_path),
            "checkpoint_paths": [str(path) for path in self.checkpoint_paths],
            "run_summary_path": str(self.run_summary_path),
        }

    def log_event(self, event: dict[str, Any]) -> None:
        dynamic_event = {
            "run_id": self.run_id,
            "timestamp": _utc_now_iso(),
            "phase": event.get("phase"),
            "step": event.get("step"),
            "train_episode_index": event.get("train_episode_index"),
            "problem_id": event.get("problem_id"),
            "problem_family": event.get("problem_family"),
            "difficulty": event.get("difficulty"),
            "teacher_prompt": event.get("teacher_prompt"),
            "solver_completion": event.get("solver_completion"),
            "extracted_code": event.get("extracted_code"),
            "reward": event.get("reward"),
            "pass_rate": event.get("pass_rate"),
            "visible_pass_rate": event.get("visible_pass_rate"),
            "execution_status": event.get("execution_status"),
            "efficiency_score": event.get("efficiency_score"),
            "optimization_hints": event.get("optimization_hints", []),
            "feedback": event.get("feedback"),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(dynamic_event)) + "\n")

        self._latest_event = dynamic_event
        if dynamic_event.get("reward") is not None:
            self._recent_rewards.append(float(dynamic_event["reward"]))
        if dynamic_event.get("pass_rate") is not None:
            self._recent_pass_rates.append(float(dynamic_event["pass_rate"]))
        if dynamic_event.get("efficiency_score") is not None:
            self._recent_efficiency_scores.append(float(dynamic_event["efficiency_score"]))

    def record_progress(self, progress: dict[str, Any]) -> None:
        self._latest_progress.update({key: value for key, value in progress.items() if value is not None})
        completed_steps = int(self._latest_progress.get("completed_steps", 0) or 0)
        interval = int(max(self.checkpoint_interval_steps, 1))
        if completed_steps > 0 and completed_steps % interval == 0 and completed_steps != self._last_checkpoint_step:
            self._write_checkpoint(completed_steps)

    def finalize(self, *, reward_curve_csv: Path | None = None, final_metrics: dict[str, Any] | None = None) -> Path:
        copied_reward_curve = None
        if reward_curve_csv is not None and reward_curve_csv.exists():
            copied_reward_curve = self.logs_dir / "reward_curve.csv"
            if reward_curve_csv.resolve() != copied_reward_curve.resolve():
                shutil.copy2(reward_curve_csv, copied_reward_curve)
            else:
                copied_reward_curve = reward_curve_csv

        summary = {
            "run_id": self.run_id,
            "finished_at": _utc_now_iso(),
            "artifact_paths": self.artifact_paths(),
            "reward_curve_csv": str(copied_reward_curve) if copied_reward_curve else None,
            "latest_progress": self._latest_progress,
            "latest_event": self._latest_event,
            "rolling_metrics": self._rolling_metrics(),
            "final_metrics": final_metrics or {},
        }
        self.run_summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
        return self.run_summary_path

    def _write_checkpoint(self, step: int) -> None:
        checkpoint_payload = {
            "run_id": self.run_id,
            "timestamp": _utc_now_iso(),
            "step": int(step),
            "phase": self._latest_progress.get("phase"),
            "total_steps": self._latest_progress.get("total_steps"),
            "remaining_steps": self._latest_progress.get("remaining_steps"),
            "progress_ratio": self._latest_progress.get("progress_ratio"),
            "current_epoch": self._latest_progress.get("current_epoch"),
            "current_difficulty": self._latest_progress.get("current_difficulty"),
            "curriculum_level": self._latest_progress.get("curriculum_level"),
            "train_episode_index": self._latest_progress.get("train_episode_index"),
            "last_problem_id": self._latest_progress.get("last_problem_id"),
            "last_problem_family": self._latest_progress.get("last_problem_family"),
            "last_execution_status": self._latest_progress.get("last_execution_status"),
            "rolling_metrics": self._rolling_metrics(),
            "artifact_paths": {
                "events_path": str(self.events_path),
                "latest_checkpoint_path": str(self.latest_checkpoint_path),
            },
        }
        checkpoint_path = self.logs_dir / f"checkpoint_step_{step:05d}.json"
        checkpoint_path.write_text(json.dumps(_json_safe(checkpoint_payload), indent=2), encoding="utf-8")
        self.latest_checkpoint_path.write_text(json.dumps(_json_safe(checkpoint_payload), indent=2), encoding="utf-8")
        self.checkpoint_paths.append(checkpoint_path)
        self._last_checkpoint_step = step

    def _rolling_metrics(self) -> dict[str, Any]:
        def _average(values: deque[float]) -> float | None:
            if not values:
                return None
            return round(sum(values) / len(values), 4)

        return {
            "avg_reward": _average(self._recent_rewards),
            "avg_pass_rate": _average(self._recent_pass_rates),
            "avg_efficiency_score": _average(self._recent_efficiency_scores),
        }
