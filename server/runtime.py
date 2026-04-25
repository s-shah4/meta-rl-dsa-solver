from __future__ import annotations

import json
import os
import shutil
import threading
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from env.adapt_env import AdaptEnvironment, MAX_STEPS_PER_EPISODE
from models import AdaptAction
from training.train_grpo import (
    SYSTEM_PROMPT,
    TrainingConfig,
    build_solver_prompt,
    build_training_config,
    extract_code,
    generate_completion,
    run_training,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class ModelState:
    loaded: bool = False
    active_model_kind: str = "unavailable"
    source_repo_id: str | None = None
    local_path: str | None = None
    revision: str | None = None
    base_model_name: str | None = None
    loaded_at: datetime | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["loaded_at"] = _iso_or_none(self.loaded_at)
        return payload


@dataclass
class TrainingJobState:
    status: str = "idle"
    run_id: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    artifact_path: str | None = None
    reward_curve_csv: str | None = None
    model_repo_id: str | None = None
    uploaded_revision: str | None = None
    logs_dir: str | None = None
    run_manifest_path: str | None = None
    events_path: str | None = None
    latest_checkpoint_path: str | None = None
    run_summary_path: str | None = None
    checkpoint_paths: list[str] = field(default_factory=list)
    logs_deleted_from_space: bool = False
    phase: str = "idle"
    completed_steps: int = 0
    total_steps: int = 0
    remaining_steps: int = 0
    current_epoch: float = 0.0
    epochs_remaining: float | None = None
    progress_ratio: float = 0.0
    precision_mode: str | None = None
    runtime_versions: dict[str, Any] = field(default_factory=dict)
    precision_policy: dict[str, Any] = field(default_factory=dict)
    precision_audit: dict[str, Any] = field(default_factory=dict)
    critical_precision_audit: dict[str, Any] = field(default_factory=dict)
    train_episode_index: int = 0
    current_difficulty: str | None = None
    curriculum_level: int | None = None
    last_problem_id: str | None = None
    last_problem_family: str | None = None
    last_pass_rate: float | None = None
    last_visible_pass_rate: float | None = None
    last_reward: float | None = None
    last_execution_status: str | None = None
    baseline_summary: dict[str, Any] = field(default_factory=dict)
    trained_summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["started_at"] = _iso_or_none(self.started_at)
        payload["finished_at"] = _iso_or_none(self.finished_at)
        return payload


class SpaceModelRegistry:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.cache_dir = self.output_root / "model_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._model: Any = None
        self._tokenizer: Any = None
        self._base_model: Any = None
        self._base_tokenizer: Any = None
        self._state = ModelState(
            source_repo_id=os.getenv("HF_MODEL_REPO_ID"),
            base_model_name=os.getenv("BASE_MODEL_NAME"),
        )

    @property
    def state(self) -> ModelState:
        with self._lock:
            return ModelState(
                loaded=self._state.loaded,
                active_model_kind=self._state.active_model_kind,
                source_repo_id=self._state.source_repo_id,
                local_path=self._state.local_path,
                revision=self._state.revision,
                base_model_name=self._state.base_model_name,
                loaded_at=self._state.loaded_at,
                error=self._state.error,
            )

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            payload = self._state.to_dict()
            payload["cache_dir"] = str(self.cache_dir)
            return payload

    def _set_state(self, **updates: Any) -> None:
        for key, value in updates.items():
            setattr(self._state, key, value)

    def _require_runtime_dependencies(self) -> tuple[Any, Any, Any]:
        try:
            import torch
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Model runtime dependencies are missing. Install `transformers`, `peft`, and `torch`."
            ) from exc
        return torch, AutoPeftModelForCausalLM, (AutoModelForCausalLM, AutoTokenizer)

    def _base_model_name(self) -> str:
        return os.getenv("BASE_MODEL_NAME") or os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-3B-Instruct"

    def _active_generation_stack(
        self,
        *,
        allow_base_fallback: bool,
    ) -> tuple[Any, Any, dict[str, Any]]:
        with self._lock:
            if self._model is not None and self._tokenizer is not None and self._state.loaded:
                return self._model, self._tokenizer, self._state.to_dict()
        if not allow_base_fallback:
            raise RuntimeError("No trained model is loaded yet.")
        try:
            base_state = self.load_base_model()
        except Exception as exc:
            raise RuntimeError(f"Base model load failed: {exc}") from exc
        with self._lock:
            if self._base_model is None or self._base_tokenizer is None:
                raise RuntimeError("No model is available for generation.")
            return self._base_model, self._base_tokenizer, base_state

    def _base_generation_stack(self) -> tuple[Any, Any, dict[str, Any]]:
        try:
            base_state = self.load_base_model()
        except Exception as exc:
            raise RuntimeError(f"Base model load failed: {exc}") from exc
        with self._lock:
            if self._base_model is None or self._base_tokenizer is None:
                raise RuntimeError("Base model could not be loaded for fallback generation.")
            return self._base_model, self._base_tokenizer, base_state

    def _generate_with_possible_base_fallback(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        allow_base_fallback: bool,
    ) -> tuple[str, dict[str, Any]]:
        model, tokenizer, model_state = self._active_generation_stack(allow_base_fallback=allow_base_fallback)
        try:
            completion = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            return completion, model_state
        except Exception as exc:
            if model_state.get("active_model_kind") == "trained" and allow_base_fallback:
                fallback_model, fallback_tokenizer, fallback_state = self._base_generation_stack()
                try:
                    completion = generate_completion(
                        model=fallback_model,
                        tokenizer=fallback_tokenizer,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                    )
                except Exception as fallback_exc:
                    raise RuntimeError(
                        "Generation failed for both the trained model and the base-model fallback. "
                        f"trained_error={exc}; base_error={fallback_exc}"
                    ) from fallback_exc
                fallback_state = dict(fallback_state)
                fallback_state["fallback_reason"] = str(exc)
                fallback_state["fallback_from"] = model_state.get("active_model_kind")
                return completion, fallback_state
            raise RuntimeError(f"Generation failed: {exc}") from exc

    def load_base_model(self) -> dict[str, Any]:
        torch, _, model_components = self._require_runtime_dependencies()
        AutoModelForCausalLM, AutoTokenizer = model_components
        base_model_name = self._base_model_name()

        with self._lock:
            if self._base_model is not None and self._base_tokenizer is not None:
                self._set_state(
                    base_model_name=base_model_name,
                    active_model_kind="base",
                    loaded=True,
                    error=None,
                )
                return self.status_payload()

            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif torch.cuda.is_available():
                dtype = torch.float16
            else:
                dtype = torch.float32
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                torch_dtype=dtype,
            )
            model.eval()
            self._base_model = model
            self._base_tokenizer = tokenizer
            self._set_state(
                loaded=True,
                active_model_kind="base",
                base_model_name=base_model_name,
                local_path=base_model_name,
                revision=None,
                loaded_at=_utc_now(),
                error=None,
            )
            return self.status_payload()

    def load_from_local(
        self,
        artifact_path: str | Path,
        *,
        source_repo_id: str | None = None,
        revision: str | None = None,
    ) -> dict[str, Any]:
        torch, AutoPeftModelForCausalLM, model_components = self._require_runtime_dependencies()
        AutoModelForCausalLM, AutoTokenizer = model_components

        artifact_dir = Path(artifact_path)
        if not artifact_dir.exists():
            raise RuntimeError(f"Trained artifact directory does not exist: {artifact_dir}")

        with self._lock:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif torch.cuda.is_available():
                dtype = torch.float16
            else:
                dtype = torch.float32
            tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir))
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

            if (artifact_dir / "adapter_config.json").exists():
                model = AutoPeftModelForCausalLM.from_pretrained(
                    str(artifact_dir),
                    device_map="auto",
                    torch_dtype=dtype,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    str(artifact_dir),
                    device_map="auto",
                    torch_dtype=dtype,
                )

            model.eval()
            self._model = model
            self._tokenizer = tokenizer
            self._set_state(
                loaded=True,
                active_model_kind="trained",
                source_repo_id=source_repo_id or self._state.source_repo_id,
                local_path=str(artifact_dir.resolve()),
                revision=revision,
                loaded_at=_utc_now(),
                error=None,
            )
            return self.status_payload()

    def load_latest_from_hub(self) -> dict[str, Any]:
        repo_id = os.getenv("HF_MODEL_REPO_ID")
        token = os.getenv("HF_TOKEN")
        base_model_name = self._base_model_name()

        if not repo_id:
            with self._lock:
                self._set_state(
                    loaded=self._base_model is not None and self._base_tokenizer is not None,
                    active_model_kind="base",
                    source_repo_id=None,
                    base_model_name=base_model_name,
                    local_path=base_model_name,
                    revision=None,
                    error=None,
                )
                return self.status_payload()

        if not token:
            with self._lock:
                self._set_state(
                    loaded=self._base_model is not None and self._base_tokenizer is not None,
                    active_model_kind="base",
                    source_repo_id=repo_id,
                    base_model_name=base_model_name,
                    local_path=base_model_name,
                    revision=None,
                    error=None,
                )
                return self.status_payload()

        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:
            raise RuntimeError("Hugging Face Hub dependency is missing. Install `huggingface_hub`.") from exc

        api = HfApi(token=token)
        try:
            info = api.model_info(repo_id=repo_id, token=token)
        except Exception as exc:
            with self._lock:
                self._set_state(
                    loaded=False,
                    active_model_kind="unavailable",
                    source_repo_id=repo_id,
                    base_model_name=base_model_name,
                    error=f"Unable to fetch model repo metadata: {exc}",
                )
                return self.status_payload()

        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            local_dir=str(self.cache_dir / repo_id.replace("/", "__")),
            local_dir_use_symlinks=False,
        )
        try:
            return self.load_from_local(local_path, source_repo_id=repo_id, revision=getattr(info, "sha", None))
        except Exception as exc:
            with self._lock:
                self._model = None
                self._tokenizer = None
                self._set_state(
                    loaded=self._base_model is not None and self._base_tokenizer is not None,
                    active_model_kind="base",
                    source_repo_id=repo_id,
                    base_model_name=base_model_name,
                    local_path=base_model_name,
                    revision=getattr(info, "sha", None),
                    error=f"Trained model repo is not loadable yet: {exc}",
                )
                return self.status_payload()

    def run_policy(
        self,
        *,
        problem_id: str | None = None,
        difficulty: str | None = None,
        max_new_tokens: int = 512,
    ) -> dict[str, Any]:
        env = AdaptEnvironment()
        observation = env.reset(problem_id=problem_id, difficulty=difficulty)
        trajectory: list[dict[str, Any]] = []
        model_state: dict[str, Any] | None = None

        for step_index in range(1, MAX_STEPS_PER_EPISODE + 1):
            prompt = build_solver_prompt(observation.model_dump())
            completion, current_model_state = self._generate_with_possible_base_fallback(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                allow_base_fallback=True,
            )
            model_state = current_model_state
            code = extract_code(completion)
            observation = env.step(AdaptAction(session_id=env.session_id, code=code))
            trajectory.append(
                {
                    "step": step_index,
                    "completion": completion,
                    "code": code,
                    "reward": float(observation.reward),
                    "done": bool(observation.done),
                    "pass_rate": float(observation.pass_rate),
                    "visible_pass_rate": float(observation.visible_pass_rate),
                    "execution_status": observation.execution_status,
                    "feedback": observation.feedback,
                }
            )
            if observation.done:
                break

        return {
            "session_id": env.session_id,
            "problem_id": observation.problem_id,
            "difficulty": observation.difficulty,
            "steps": trajectory,
            "final_observation": observation.model_dump(),
            "model": model_state or {},
        }

    def generate_code(
        self,
        *,
        problem: str,
        input_format: str,
        constraints: str,
        feedback: str | None = None,
        problem_id: str = "custom_problem",
        problem_type: str = "custom",
        difficulty: str = "custom",
        attempt_number: int = 1,
        max_steps: int = 1,
        max_new_tokens: int = 512,
    ) -> dict[str, Any]:
        prompt = build_solver_prompt(
            {
                "problem_id": problem_id,
                "problem_type": problem_type,
                "difficulty": difficulty,
                "attempt_number": attempt_number,
                "max_steps": max_steps,
                "problem": problem,
                "input_format": input_format,
                "constraints": constraints,
                "feedback": feedback or "No previous attempt yet. Solve the problem directly.",
            }
        )
        completion, model_state = self._generate_with_possible_base_fallback(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            allow_base_fallback=True,
        )
        return {
            "problem_id": problem_id,
            "problem_type": problem_type,
            "difficulty": difficulty,
            "prompt": prompt,
            "completion": completion,
            "code": extract_code(completion),
            "model": model_state,
            "system_prompt": SYSTEM_PROMPT,
        }


class SpaceTrainingManager:
    def __init__(self, output_root: str | Path | None = None) -> None:
        resolved_root = Path(output_root or os.getenv("SPACE_OUTPUT_ROOT", "/tmp/adapt-space")).resolve()
        resolved_root.mkdir(parents=True, exist_ok=True)
        self.output_root = resolved_root
        self.status_file = self.output_root / "training_status.json"
        self.runs_dir = self.output_root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._job = TrainingJobState(model_repo_id=os.getenv("HF_MODEL_REPO_ID"))
        self._worker: threading.Thread | None = None
        self.model_registry = SpaceModelRegistry(self.output_root)
        self._restore_status()

    def _restore_status(self) -> None:
        if not self.status_file.exists():
            return
        try:
            payload = json.loads(self.status_file.read_text(encoding="utf-8"))
            self._job = TrainingJobState(
                status=payload.get("status", "idle"),
                run_id=payload.get("run_id"),
                config=payload.get("config", {}),
                started_at=datetime.fromisoformat(payload["started_at"]) if payload.get("started_at") else None,
                finished_at=datetime.fromisoformat(payload["finished_at"]) if payload.get("finished_at") else None,
                artifact_path=payload.get("artifact_path"),
                reward_curve_csv=payload.get("reward_curve_csv"),
                model_repo_id=payload.get("model_repo_id"),
                uploaded_revision=payload.get("uploaded_revision"),
                logs_dir=payload.get("logs_dir"),
                run_manifest_path=payload.get("run_manifest_path"),
                events_path=payload.get("events_path"),
                latest_checkpoint_path=payload.get("latest_checkpoint_path"),
                run_summary_path=payload.get("run_summary_path"),
                checkpoint_paths=payload.get("checkpoint_paths", []),
                logs_deleted_from_space=bool(payload.get("logs_deleted_from_space", False)),
                phase=payload.get("phase", "idle"),
                completed_steps=int(payload.get("completed_steps", 0) or 0),
                total_steps=int(payload.get("total_steps", 0) or 0),
                remaining_steps=int(payload.get("remaining_steps", 0) or 0),
                current_epoch=float(payload.get("current_epoch", 0.0) or 0.0),
                epochs_remaining=payload.get("epochs_remaining"),
                progress_ratio=float(payload.get("progress_ratio", 0.0) or 0.0),
                precision_mode=payload.get("precision_mode"),
                runtime_versions=payload.get("runtime_versions", {}),
                precision_policy=payload.get("precision_policy", {}),
                precision_audit=payload.get("precision_audit", {}),
                critical_precision_audit=payload.get("critical_precision_audit", {}),
                train_episode_index=int(payload.get("train_episode_index", 0) or 0),
                current_difficulty=payload.get("current_difficulty"),
                curriculum_level=payload.get("curriculum_level"),
                last_problem_id=payload.get("last_problem_id"),
                last_problem_family=payload.get("last_problem_family"),
                last_pass_rate=payload.get("last_pass_rate"),
                last_visible_pass_rate=payload.get("last_visible_pass_rate"),
                last_reward=payload.get("last_reward"),
                last_execution_status=payload.get("last_execution_status"),
                baseline_summary=payload.get("baseline_summary", {}),
                trained_summary=payload.get("trained_summary", {}),
                error=payload.get("error"),
                traceback=payload.get("traceback"),
            )
        except Exception:
            self._job = TrainingJobState(model_repo_id=os.getenv("HF_MODEL_REPO_ID"))

    def _persist_status(self) -> None:
        self.status_file.write_text(
            json.dumps(_json_safe(self._job.to_dict()), indent=2),
            encoding="utf-8",
        )

    def _update_progress(self, updates: dict[str, Any]) -> None:
        with self._lock:
            for key, value in updates.items():
                if hasattr(self._job, key) and value is not None:
                    setattr(self._job, key, value)
            self._job.total_steps = int(self._job.config.get("max_steps", self._job.total_steps or 0) or 0)
            self._job.completed_steps = min(int(self._job.completed_steps), int(self._job.total_steps or self._job.completed_steps))
            self._job.remaining_steps = max(int(self._job.total_steps) - int(self._job.completed_steps), 0)
            self._job.progress_ratio = (
                round(float(self._job.completed_steps) / float(self._job.total_steps), 4)
                if self._job.total_steps
                else 0.0
            )
            self._job.epochs_remaining = (
                round(max(float(self._job.total_steps) - float(self._job.current_epoch), 0.0), 4)
                if self._job.total_steps
                else None
            )
            self._persist_status()

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            payload = self._job.to_dict()
            payload["output_root"] = str(self.output_root)
            payload["status_file"] = str(self.status_file)
            payload["active"] = payload["status"] == "running"
            return payload

    def model_status_payload(self) -> dict[str, Any]:
        return self.model_registry.status_payload()

    def start_training(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        request_payload = payload or {}
        preset = request_payload.get("preset", "smoke")
        overrides = {key: value for key, value in request_payload.items() if key != "preset"}

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("A training run is already in progress.")

            run_id = str(uuid4())
            config = build_training_config(preset=preset, overrides=overrides)
            requested_output_dir = Path(config.output_dir)
            if requested_output_dir.is_absolute():
                output_dir = requested_output_dir / run_id
            else:
                output_dir = self.output_root / requested_output_dir / run_id
            config.output_dir = str(output_dir)
            logs_dir = output_dir / "logs"

            self._job = TrainingJobState(
                status="running",
                run_id=run_id,
                config=config.to_dict(),
                started_at=_utc_now(),
                finished_at=None,
                artifact_path=None,
                reward_curve_csv=None,
                model_repo_id=os.getenv("HF_MODEL_REPO_ID"),
                uploaded_revision=None,
                logs_dir=str(logs_dir),
                run_manifest_path=str(logs_dir / "run_manifest.json"),
                events_path=str(logs_dir / "events.jsonl"),
                latest_checkpoint_path=str(logs_dir / "latest_checkpoint.json"),
                run_summary_path=str(logs_dir / "run_summary.json"),
                checkpoint_paths=[],
                phase="queued",
                completed_steps=0,
                total_steps=int(config.max_steps),
                remaining_steps=int(config.max_steps),
                current_epoch=0.0,
                epochs_remaining=float(config.max_steps),
                progress_ratio=0.0,
                precision_mode=None,
                runtime_versions={},
                precision_policy={},
                precision_audit={},
                critical_precision_audit={},
                error=None,
                traceback=None,
            )
            self._persist_status()
            self._worker = threading.Thread(
                target=self._run_training_job,
                args=(run_id, config),
                daemon=True,
                name=f"space-train-{run_id}",
            )
            self._worker.start()
            return self._job.to_dict()

    def _upload_artifacts(self, artifact_path: str, run_id: str) -> str:
        token = os.getenv("HF_TOKEN")
        repo_id = os.getenv("HF_MODEL_REPO_ID")
        if not token:
            raise RuntimeError("HF_TOKEN is required to upload trained artifacts.")
        if not repo_id:
            raise RuntimeError("HF_MODEL_REPO_ID is required to upload trained artifacts.")

        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("Hugging Face Hub dependency is missing. Install `huggingface_hub`.") from exc

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        commit_info = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=artifact_path,
            commit_message=f"Upload trained artifact for run {run_id}",
        )
        return getattr(commit_info, "oid", None) or getattr(commit_info, "commit_hash", None) or "unknown"

    def _cleanup_local_logs(self, log_dir: str | None) -> bool:
        if not log_dir:
            return False
        folder_path = Path(log_dir)
        if not folder_path.exists():
            return False
        shutil.rmtree(folder_path, ignore_errors=True)
        return not folder_path.exists()

    def _run_training_job(self, run_id: str, config: TrainingConfig) -> None:
        summary: dict[str, Any] | None = None
        try:
            summary = run_training(config, run_id=run_id, progress_callback=self._update_progress)
            artifact_path = summary["output_dir"]
            uploaded_revision = self._upload_artifacts(artifact_path, run_id)
            logs_deleted = self._cleanup_local_logs(summary.get("logs_dir"))
            self.model_registry.load_latest_from_hub()

            with self._lock:
                self._job.status = "succeeded"
                self._job.finished_at = _utc_now()
                self._job.artifact_path = artifact_path
                self._job.reward_curve_csv = summary.get("reward_curve_csv")
                self._job.model_repo_id = os.getenv("HF_MODEL_REPO_ID")
                self._job.uploaded_revision = uploaded_revision
                self._job.logs_dir = None if logs_deleted else summary.get("logs_dir")
                self._job.run_manifest_path = None if logs_deleted else summary.get("run_manifest_path")
                self._job.events_path = None if logs_deleted else summary.get("events_path")
                self._job.latest_checkpoint_path = None if logs_deleted else summary.get("latest_checkpoint_path")
                self._job.run_summary_path = None if logs_deleted else summary.get("run_summary_path")
                self._job.checkpoint_paths = [] if logs_deleted else summary.get("checkpoint_paths", [])
                self._job.logs_deleted_from_space = logs_deleted
                self._job.phase = "completed"
                self._job.completed_steps = int(summary.get("completed_steps", config.max_steps))
                self._job.total_steps = int(config.max_steps)
                self._job.remaining_steps = 0
                self._job.progress_ratio = 1.0 if self._job.total_steps else 0.0
                self._job.precision_mode = summary.get("precision_mode")
                self._job.runtime_versions = summary.get("runtime_versions", {})
                self._job.precision_policy = summary.get("precision_policy", {})
                self._job.precision_audit = summary.get("precision_audit", {})
                self._job.critical_precision_audit = summary.get("critical_precision_audit", {})
                self._job.current_epoch = float(summary.get("completed_steps", config.max_steps))
                self._job.epochs_remaining = 0.0
                self._job.baseline_summary = summary.get("baseline_summary", {})
                self._job.trained_summary = summary.get("trained_summary", {})
                self._job.error = None
                self._job.traceback = None
                self._persist_status()
        except Exception as exc:
            logs_deleted = self._cleanup_local_logs(summary.get("logs_dir") if summary else self._job.logs_dir)
            with self._lock:
                self._job.status = "failed"
                self._job.finished_at = _utc_now()
                if logs_deleted:
                    self._job.logs_dir = None
                    self._job.run_manifest_path = None
                    self._job.events_path = None
                    self._job.latest_checkpoint_path = None
                    self._job.run_summary_path = None
                    self._job.checkpoint_paths = []
                self._job.logs_deleted_from_space = logs_deleted
                self._job.error = str(exc)
                self._job.traceback = traceback.format_exc()
                self._persist_status()

    def load_latest_model(self) -> dict[str, Any]:
        return self.model_registry.load_latest_from_hub()

    def run_trained_policy(
        self,
        *,
        problem_id: str | None = None,
        difficulty: str | None = None,
        max_new_tokens: int = 512,
    ) -> dict[str, Any]:
        return self.model_registry.run_policy(
            problem_id=problem_id,
            difficulty=difficulty,
            max_new_tokens=max_new_tokens,
        )

    def generate_code(
        self,
        *,
        problem: str,
        input_format: str,
        constraints: str,
        feedback: str | None = None,
        problem_id: str = "custom_problem",
        problem_type: str = "custom",
        difficulty: str = "custom",
        attempt_number: int = 1,
        max_steps: int = 1,
        max_new_tokens: int = 512,
    ) -> dict[str, Any]:
        return self.model_registry.generate_code(
            problem=problem,
            input_format=input_format,
            constraints=constraints,
            feedback=feedback,
            problem_id=problem_id,
            problem_type=problem_type,
            difficulty=difficulty,
            attempt_number=attempt_number,
            max_steps=max_steps,
            max_new_tokens=max_new_tokens,
        )
