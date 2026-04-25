from __future__ import annotations

import json
import os
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
        self._state = ModelState(
            source_repo_id=os.getenv("HF_MODEL_REPO_ID"),
            base_model_name=os.getenv("BASE_MODEL_NAME"),
        )

    @property
    def state(self) -> ModelState:
        with self._lock:
            return ModelState(
                loaded=self._state.loaded,
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
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
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
        base_model_name = os.getenv("BASE_MODEL_NAME")

        if not repo_id:
            with self._lock:
                self._set_state(
                    loaded=False,
                    source_repo_id=None,
                    base_model_name=base_model_name,
                    error="HF_MODEL_REPO_ID is not configured.",
                )
                return self.status_payload()

        if not token:
            with self._lock:
                self._set_state(
                    loaded=False,
                    source_repo_id=repo_id,
                    base_model_name=base_model_name,
                    error="HF_TOKEN is not configured.",
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
        return self.load_from_local(local_path, source_repo_id=repo_id, revision=getattr(info, "sha", None))

    def run_policy(
        self,
        *,
        problem_id: str | None = None,
        difficulty: str | None = None,
        max_new_tokens: int = 512,
    ) -> dict[str, Any]:
        with self._lock:
            if self._model is None or self._tokenizer is None or not self._state.loaded:
                raise RuntimeError("No trained model is loaded yet.")
            model = self._model
            tokenizer = self._tokenizer
            model_state = self._state.to_dict()

        env = AdaptEnvironment()
        observation = env.reset(problem_id=problem_id, difficulty=difficulty)
        trajectory: list[dict[str, Any]] = []

        for step_index in range(1, MAX_STEPS_PER_EPISODE + 1):
            prompt = build_solver_prompt(observation.model_dump())
            completion = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
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
            "model": model_state,
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
        with self._lock:
            if self._model is None or self._tokenizer is None or not self._state.loaded:
                raise RuntimeError("No trained model is loaded yet.")
            model = self._model
            tokenizer = self._tokenizer
            model_state = self._state.to_dict()

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
        completion = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
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

    def _run_training_job(self, run_id: str, config: TrainingConfig) -> None:
        try:
            summary = run_training(config)
            artifact_path = summary["output_dir"]
            uploaded_revision = self._upload_artifacts(artifact_path, run_id)
            self.model_registry.load_latest_from_hub()

            with self._lock:
                self._job.status = "succeeded"
                self._job.finished_at = _utc_now()
                self._job.artifact_path = artifact_path
                self._job.reward_curve_csv = summary.get("reward_curve_csv")
                self._job.model_repo_id = os.getenv("HF_MODEL_REPO_ID")
                self._job.uploaded_revision = uploaded_revision
                self._job.error = None
                self._job.traceback = None
                self._persist_status()
        except Exception as exc:
            with self._lock:
                self._job.status = "failed"
                self._job.finished_at = _utc_now()
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
