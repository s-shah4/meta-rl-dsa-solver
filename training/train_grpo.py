from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from env.adapt_env import AdaptEnvironment, MAX_STEPS_PER_EPISODE
from env.generator import DIFFICULTY_LABELS, GeneratorAgent
from models import AdaptAction
from training.trace_logging import TraceArtifactLogger

SYSTEM_PROMPT = """You are the Solver Agent for ADAPT.
Write only runnable Python code.
The program must read from stdin and print to stdout.
If feedback is present, repair your previous solution instead of starting from scratch.
Do not include markdown fences or explanations."""


@dataclass
class TrainingConfig:
    model_name: str = "unsloth/Llama-3.2-3B-Instruct"
    output_dir: str = "outputs_v3"
    dataset_size: int = 200
    max_steps: int = 250
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 8
    max_seq_length: int = 2048
    max_prompt_length: int = 1024
    max_completion_length: int = 512
    learning_rate: float = 5e-6
    lora_rank: int = 16
    lora_alpha: int = 16
    disable_4bit: bool = False
    bf16: bool = False
    baseline_eval: bool = False
    evaluation_episodes: int = 20
    eval_max_new_tokens: int = 512
    disable_wandb: bool = False
    wandb_project: str = "adapt-dsa-tutor"
    wandb_run_name: str | None = None
    generator_mode: str = "reward_aware"
    non_deterministic_generator: bool = False
    trace_logging_enabled: bool = True
    checkpoint_log_interval_steps: int = 10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TRAINING_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "dataset_size": 12,
        "max_steps": 6,
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "num_generations": 2,
        "evaluation_episodes": 3,
        "baseline_eval": False,
        "disable_wandb": True,
        "disable_4bit": True,
        "output_dir": "outputs_smoke",
        "checkpoint_log_interval_steps": 2,
    },
    "default": {},
}


def extract_code(completion: str) -> str:
    text = completion.strip()
    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


def format_examples(problem: dict[str, Any]) -> str:
    visible_cases = [test_case for test_case in problem.get("test_cases", []) if test_case.get("is_visible", False)]
    if not visible_cases:
        return problem["problem"]

    chunks = []
    for test_case in visible_cases:
        chunks.append(f"Input:\n{test_case['input']}Expected Output:\n{test_case['output']}\n")
    return f"{problem['problem']}\n\nExamples:\n" + "\n".join(chunks).rstrip()


def build_solver_prompt(payload: dict[str, Any]) -> str:
    feedback = payload.get("feedback") or "No previous attempt yet."
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Problem ID: {payload['problem_id']}\n"
        f"Problem Family: {payload['problem_type']}\n"
        f"Difficulty: {payload['difficulty']}\n"
        f"Attempt: {payload.get('attempt_number', 0)}/{payload.get('max_steps', MAX_STEPS_PER_EPISODE)}\n\n"
        f"Problem:\n{payload['problem']}\n\n"
        f"Input Format:\n{payload['input_format']}\n\n"
        f"Constraints:\n{payload['constraints']}\n\n"
        f"Feedback:\n{feedback}\n"
    )


def build_prompt_from_problem(problem: dict[str, Any]) -> str:
    payload = {
        "problem_id": problem["problem_id"],
        "problem_type": problem["problem_type"],
        "difficulty": problem["difficulty_label"],
        "attempt_number": 0,
        "max_steps": MAX_STEPS_PER_EPISODE,
        "problem": format_examples(problem),
        "input_format": problem["input_format"],
        "constraints": problem["constraints"],
        "feedback": "No previous attempt yet. Solve the problem directly from the examples and constraints.",
    }
    return build_solver_prompt(payload)


def build_training_config(
    preset: str = "default",
    overrides: dict[str, Any] | None = None,
) -> TrainingConfig:
    if preset not in TRAINING_PRESETS:
        raise ValueError(f"Unknown training preset: {preset}")

    payload = TrainingConfig().to_dict()
    payload.update(TRAINING_PRESETS[preset])
    if overrides:
        for key, value in overrides.items():
            if value is not None and key in payload:
                payload[key] = value
    return TrainingConfig(**payload)


def namespace_to_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        dataset_size=args.dataset_size,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_seq_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        disable_4bit=args.disable_4bit,
        bf16=args.bf16,
        baseline_eval=args.baseline_eval,
        evaluation_episodes=args.evaluation_episodes,
        eval_max_new_tokens=args.eval_max_new_tokens,
        disable_wandb=args.disable_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        generator_mode=args.generator_mode,
        non_deterministic_generator=args.non_deterministic_generator,
        trace_logging_enabled=args.trace_logging_enabled,
        checkpoint_log_interval_steps=args.checkpoint_log_interval_steps,
    )


@dataclass
class CurriculumManager:
    difficulties: list[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    current_idx: int = 0
    success_history: list[float] = field(default_factory=list)
    window_size: int = 10
    promote_threshold: float = 0.70
    demote_threshold: float = 0.30

    def current_difficulty(self) -> str:
        return self.difficulties[self.current_idx]

    def current_level(self) -> int:
        return self.current_idx + 1

    def update(self, episode_pass_rate: float) -> None:
        self.success_history.append(float(episode_pass_rate))
        if len(self.success_history) > self.window_size:
            self.success_history.pop(0)

        if len(self.success_history) < self.window_size:
            return

        moving_average = sum(self.success_history) / len(self.success_history)
        if moving_average >= self.promote_threshold and self.current_idx < len(self.difficulties) - 1:
            self.current_idx += 1
            self.success_history.clear()
            print(
                f"[curriculum] promoted to {self.current_difficulty()} "
                f"(moving_pass_rate={moving_average:.2f})"
            )
        elif moving_average <= self.demote_threshold and self.current_idx > 0:
            self.current_idx -= 1
            self.success_history.clear()
            print(
                f"[curriculum] demoted to {self.current_difficulty()} "
                f"(moving_pass_rate={moving_average:.2f})"
            )


@dataclass
class GeneratorController:
    mode: str = "heuristic"
    deterministic: bool = True
    temperature: float = 0.5
    generator: GeneratorAgent = field(init=False)
    history: dict[str, Any] = field(
        default_factory=lambda: {
            "recent_pass_rates": [],
            "problem_types": [],
            "generator_rewards": [],
            "problem_signatures": [],
            "episode_index": 0,
        }
    )
    prompt_registry: dict[str, dict[str, Any]] = field(default_factory=dict)
    family_productivity: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.generator = GeneratorAgent(deterministic=self.deterministic)
        if not self.family_productivity:
            self.family_productivity = {
                template.problem_type: 0.0 for template in self.generator.templates
            }

    @property
    def family_names(self) -> list[str]:
        return sorted(self.family_productivity)

    def sample_problem(self, difficulty: str) -> dict[str, Any]:
        family_weights = self.family_weights_for_difficulty(difficulty)
        problem = self.generator.generate_problem(
            difficulty_level=difficulty,
            history=self.history,
            family_weights=family_weights,
        )
        return problem

    def create_rollout_problem(self, difficulty: str) -> tuple[str, dict[str, Any]]:
        problem = self.sample_problem(difficulty)
        prompt = build_prompt_from_problem(problem)
        self.prompt_registry[prompt] = problem
        return prompt, problem

    def resolve_prompt(self, prompt: str) -> dict[str, Any]:
        if prompt not in self.prompt_registry:
            raise KeyError("Prompt was not registered with the generator controller.")
        return self.prompt_registry[prompt]

    def family_weights_for_difficulty(self, difficulty: str) -> dict[str, float] | None:
        if self.mode != "reward_aware":
            return None

        eligible = [
            template.problem_type
            for template in self.generator.templates
            if DIFFICULTY_LABELS[template.difficulty_tier] == difficulty
        ]
        if not eligible:
            return None

        logits = [self.family_productivity.get(family, 0.0) / self.temperature for family in eligible]
        max_logit = max(logits)
        exp_values = [math.exp(logit - max_logit) for logit in logits]
        return {family: value for family, value in zip(eligible, exp_values)}

    def update(
        self,
        problem: dict[str, Any],
        pass_rate: float,
        generator_reward_signal: float,
        *,
        update_productivity: bool = True,
    ) -> None:
        self.history["recent_pass_rates"].append(round(float(pass_rate), 4))
        self.history["problem_types"].append(problem.get("problem_type", ""))
        self.history["problem_signatures"].append(problem.get("problem_id", ""))
        self.history["generator_rewards"].append(round(float(generator_reward_signal), 4))
        self.history["episode_index"] = int(self.history.get("episode_index", 0)) + 1

        if self.mode == "reward_aware" and update_productivity:
            family = problem.get("problem_type", "")
            current = float(self.family_productivity.get(family, 0.0))
            updated = 0.9 * current + 0.1 * float(generator_reward_signal)
            self.family_productivity[family] = round(updated, 6)

        for key in ("recent_pass_rates", "problem_types", "problem_signatures", "generator_rewards"):
            values = self.history[key]
            if len(values) > 100:
                del values[:-100]

    def stats_snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "episodes": self.history["episode_index"],
            "recent_pass_rates": list(self.history["recent_pass_rates"][-5:]),
            "recent_problem_types": list(self.history["problem_types"][-5:]),
            "recent_generator_rewards": list(self.history["generator_rewards"][-5:]),
            "family_productivity": self.productivity_snapshot(),
        }

    def productivity_snapshot(self) -> dict[str, float]:
        return {
            family: round(float(value), 6)
            for family, value in sorted(self.family_productivity.items())
        }


class GeneratorRolloutDataset:
    def __init__(self, size: int, controller: GeneratorController, curriculum: CurriculumManager) -> None:
        self.size = size
        self.controller = controller
        self.curriculum = curriculum

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, str]:
        del index
        prompt, _ = self.controller.create_rollout_problem(self.curriculum.current_difficulty())
        return {"prompt": prompt}


@dataclass
class TrainingLogger:
    output_dir: Path
    family_names: list[str]
    use_wandb: bool = True
    wandb_project: str = "adapt-dsa-tutor"
    wandb_run_name: str | None = None
    run_id: str | None = None
    training_config: dict[str, Any] = field(default_factory=dict)
    model_identifiers: dict[str, Any] = field(default_factory=dict)
    trace_logging_enabled: bool = True
    checkpoint_log_interval_steps: int = 10
    rows: list[dict[str, Any]] = field(default_factory=list)
    global_step: int = 0
    _wandb_run: Any = field(default=None, init=False, repr=False)
    _trace_logger: TraceArtifactLogger | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.trace_logging_enabled and self.run_id:
            self._trace_logger = TraceArtifactLogger(
                run_id=self.run_id,
                output_dir=self.output_dir,
                training_config=dict(self.training_config),
                model_identifiers=dict(self.model_identifiers),
                system_prompt=SYSTEM_PROMPT,
                checkpoint_interval_steps=int(max(self.checkpoint_log_interval_steps, 1)),
            )
        if not self.use_wandb:
            return
        try:
            import wandb

            self._wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config={"family_names": self.family_names},
                reinit=True,
            )
        except Exception:
            self._wandb_run = None

    def log_event(
        self,
        *,
        phase: str,
        episode_reward: float,
        pass_rate: float,
        visible_pass_rate: float,
        difficulty_tier: str,
        problem_family: str,
        curriculum_level: int,
        execution_status: str,
        attempt_number: int,
        family_productivity: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "step": self.global_step,
            "phase": phase,
            "episode_reward": round(float(episode_reward), 4),
            "pass_rate": round(float(pass_rate), 4),
            "visible_pass_rate": round(float(visible_pass_rate), 4),
            "difficulty_tier": difficulty_tier,
            "problem_family": problem_family,
            "curriculum_level": curriculum_level,
            "execution_status": execution_status,
            "attempt_number": int(attempt_number),
        }
        for family in self.family_names:
            row[f"family_productivity__{family}"] = round(float(family_productivity.get(family, 0.0)), 6)
        if extra:
            row.update(extra)
        self.rows.append(row)
        if self._trace_logger is not None:
            self._trace_logger.log_event(
                {
                    "phase": phase,
                    "step": self.global_step,
                    "train_episode_index": extra.get("train_episode_index") if extra else None,
                    "problem_id": row.get("problem_id"),
                    "problem_family": row.get("problem_family"),
                    "difficulty": row.get("difficulty_tier"),
                    "teacher_prompt": row.get("teacher_prompt"),
                    "solver_completion": row.get("solver_completion"),
                    "extracted_code": row.get("extracted_code"),
                    "reward": row.get("episode_reward"),
                    "pass_rate": row.get("pass_rate"),
                    "visible_pass_rate": row.get("visible_pass_rate"),
                    "execution_status": row.get("execution_status"),
                    "efficiency_score": row.get("efficiency_score"),
                    "optimization_hints": row.get("optimization_hints", []),
                    "feedback": row.get("feedback"),
                }
            )
        if self._wandb_run is not None:
            self._wandb_run.log(row, step=self.global_step)
        self.global_step += 1

    def record_progress(self, updates: dict[str, Any]) -> dict[str, Any]:
        if self._trace_logger is None:
            return {}
        self._trace_logger.record_progress(updates)
        return self._trace_logger.artifact_paths()

    def write_csv(self) -> Path:
        output_path = self.output_dir / "reward_curve.csv"
        fieldnames: list[str] = []
        for row in self.rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        return output_path

    def close(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()

    def finalize_trace_artifacts(
        self,
        *,
        reward_curve_csv: Path | None = None,
        final_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._trace_logger is None:
            return {}
        self._trace_logger.finalize(reward_curve_csv=reward_curve_csv, final_metrics=final_metrics)
        return self._trace_logger.artifact_paths()


def build_dataset(size: int, controller: GeneratorController, curriculum: CurriculumManager) -> GeneratorRolloutDataset:
    return GeneratorRolloutDataset(size=size, controller=controller, curriculum=curriculum)


def extract_optimization_hints(feedback: str) -> list[str]:
    lines = [line.strip() for line in feedback.splitlines()]
    hints: list[str] = []
    capture = False
    for line in lines:
        if line == "Optimization hints:":
            capture = True
            continue
        if capture and line.startswith("- "):
            hints.append(line[2:])
        elif capture and line:
            break
    return hints


def build_reward_func(
    curriculum: CurriculumManager,
    controller: GeneratorController,
    logger: TrainingLogger,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
):
    def reward_func(prompts, completions, **kwargs) -> list[float]:
        del kwargs
        rewards: list[float] = []

        for prompt, completion in zip(prompts, completions):
            problem = controller.resolve_prompt(prompt)
            env = AdaptEnvironment(generator=controller.generator, generator_mode=controller.mode)
            env.reset(
                difficulty=problem["difficulty_label"],
                generated_problem=problem,
                generator_mode=controller.mode,
                session_id=env.session_id,
            )
            observation = env.step(
                AdaptAction(
                    session_id=env.session_id,
                    code=extract_code(completion),
                )
            )
            rewards.append(float(observation.reward))
            controller.update(
                problem=problem,
                pass_rate=observation.pass_rate,
                generator_reward_signal=observation.generator_reward_signal,
            )
            curriculum.update(observation.pass_rate)
            logger.log_event(
                phase="train",
                episode_reward=float(observation.reward),
                pass_rate=float(observation.pass_rate),
                visible_pass_rate=float(observation.visible_pass_rate),
                difficulty_tier=problem["difficulty_label"],
                problem_family=problem["problem_type"],
                curriculum_level=curriculum.current_level(),
                execution_status=observation.execution_status,
                attempt_number=int(observation.attempt_number),
                family_productivity=controller.productivity_snapshot(),
                extra={
                    "generator_reward": round(float(observation.generator_reward_signal), 4),
                    "problem_id": problem["problem_id"],
                    "teacher_prompt": prompt,
                    "solver_completion": completion,
                    "extracted_code": extract_code(completion),
                    "feedback": observation.feedback,
                    "efficiency_score": observation.reward_components.get("efficiency_score"),
                    "optimization_hints": extract_optimization_hints(observation.feedback),
                    "train_episode_index": int(controller.history["episode_index"]),
                },
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "train",
                        "train_episode_index": int(controller.history["episode_index"]),
                        "curriculum_level": int(curriculum.current_level()),
                        "current_difficulty": curriculum.current_difficulty(),
                        "last_problem_id": problem["problem_id"],
                        "last_problem_family": problem["problem_type"],
                        "last_pass_rate": round(float(observation.pass_rate), 4),
                        "last_visible_pass_rate": round(float(observation.visible_pass_rate), 4),
                        "last_reward": round(float(observation.reward), 4),
                        "last_execution_status": observation.execution_status,
                    }
                )
            if controller.mode == "reward_aware" and controller.history["episode_index"] % 50 == 0:
                print("[family_productivity]", json.dumps(controller.productivity_snapshot()))

        return rewards

    return reward_func


def render_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{SYSTEM_PROMPT}\n\n{prompt}"


def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("`torch` is required to run generation for evaluation or trained-policy execution.") from exc

    rendered = render_prompt(tokenizer, prompt)
    inputs = tokenizer(rendered, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def run_policy_evaluation(
    *,
    model: Any,
    tokenizer: Any,
    generator_mode: str,
    deterministic_generator: bool,
    episodes: int,
    logger: TrainingLogger,
    phase: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    controller = GeneratorController(
        mode=generator_mode,
        deterministic=deterministic_generator,
    )
    schedule = ["easy"] * (episodes // 3 + (1 if episodes % 3 > 0 else 0))
    schedule += ["medium"] * (episodes // 3 + (1 if episodes % 3 > 1 else 0))
    schedule += ["hard"] * (episodes // 3)
    schedule = schedule[:episodes]

    tier_records: dict[str, list[float]] = {"easy": [], "medium": [], "hard": []}

    for difficulty in schedule:
        problem = controller.sample_problem(difficulty)
        env = AdaptEnvironment(generator=controller.generator, generator_mode=generator_mode)
        observation = env.reset(
            difficulty=difficulty,
            generated_problem=problem,
            session_id=env.session_id,
            generator_mode=generator_mode,
        )
        last_prompt = ""
        last_completion = ""
        last_code = ""

        for _ in range(MAX_STEPS_PER_EPISODE):
            prompt = build_solver_prompt(observation.model_dump())
            completion = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            last_prompt = prompt
            last_completion = completion
            last_code = extract_code(completion)
            observation = env.step(
                AdaptAction(
                    session_id=env.session_id,
                    code=last_code,
                )
            )
            if observation.done:
                break

        controller.update(
            problem=problem,
            pass_rate=observation.pass_rate,
            generator_reward_signal=observation.generator_reward_signal,
            update_productivity=False,
        )
        tier_records[difficulty].append(float(observation.pass_rate))
        logger.log_event(
            phase=phase,
            episode_reward=float(observation.reward),
            pass_rate=float(observation.pass_rate),
            visible_pass_rate=float(observation.visible_pass_rate),
            difficulty_tier=difficulty,
            problem_family=problem["problem_type"],
            curriculum_level={"easy": 1, "medium": 2, "hard": 3}[difficulty],
            execution_status=observation.execution_status,
            attempt_number=int(observation.attempt_number),
            family_productivity=controller.productivity_snapshot(),
            extra={
                "generator_reward": round(float(observation.generator_reward_signal), 4),
                "problem_id": problem["problem_id"],
                "teacher_prompt": last_prompt,
                "solver_completion": last_completion,
                "extracted_code": last_code,
                "feedback": observation.feedback,
                "efficiency_score": observation.reward_components.get("efficiency_score"),
                "optimization_hints": extract_optimization_hints(observation.feedback),
                "train_episode_index": int(controller.history["episode_index"]),
            },
        )

    summary = {
        tier: (sum(values) / len(values) if values else 0.0)
        for tier, values in tier_records.items()
    }
    summary["overall"] = (
        sum(value for values in tier_records.values() for value in values) / episodes if episodes else 0.0
    )
    return summary


def print_evaluation_summary(baseline: dict[str, Any], trained: dict[str, Any]) -> None:
    print("\nBaseline vs trained pass rate summary")
    print(f"{'Difficulty':<12} {'Baseline':>10} {'Trained':>10}")
    print("-" * 34)
    for tier in ("easy", "medium", "hard", "overall"):
        print(f"{tier:<12} {baseline.get(tier, 0.0):>10.3f} {trained.get(tier, 0.0):>10.3f}")


def run_training(
    config: TrainingConfig | argparse.Namespace,
    *,
    run_id: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if isinstance(config, argparse.Namespace):
        config = namespace_to_config(config)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Training requires `torch` to be installed.") from exc

    try:
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel, PatchFastRL
        from transformers import TrainerCallback
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install `trl` and `unsloth` before running GRPO training."
        ) from exc

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    PatchFastRL("GRPO", FastLanguageModel)

    use_cuda = torch.cuda.is_available()
    use_bf16 = bool(config.bf16)
    model_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32)
    load_in_4bit = not config.disable_4bit

    # Unsloth's fast GRPO generation path has hit repeated dtype mismatches in the
    # Space when combining 4-bit loading with FP16 training. Prefer stable FP16
    # weights for this path so `/train` can complete reliably in the Space.
    if use_cuda and not use_bf16 and load_in_4bit:
        print("[training] disabling 4-bit loading for CUDA FP16 GRPO to avoid dtype mismatch")
        load_in_4bit = False

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=model_dtype,
        load_in_4bit=load_in_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, "config"):
        model.config.torch_dtype = model_dtype

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=config.lora_alpha,
        lora_dropout=0.0,
    )
    if not load_in_4bit:
        model = model.to(model_dtype)
        if hasattr(model, "config"):
            model.config.torch_dtype = model_dtype

    curriculum = CurriculumManager()
    controller = GeneratorController(
        mode="reward_aware" if config.generator_mode == "reward_aware" else "heuristic",
        deterministic=not config.non_deterministic_generator,
    )
    logger = TrainingLogger(
        output_dir=output_dir,
        family_names=controller.family_names,
        use_wandb=not config.disable_wandb,
        wandb_project=config.wandb_project,
        wandb_run_name=config.wandb_run_name,
        run_id=run_id,
        training_config=config.to_dict(),
        model_identifiers={
            "model_name": config.model_name,
            "generator_mode": config.generator_mode,
        },
        trace_logging_enabled=config.trace_logging_enabled,
        checkpoint_log_interval_steps=config.checkpoint_log_interval_steps,
    )

    def emit_progress(update: dict[str, Any]) -> None:
        artifact_paths = logger.record_progress(update)
        if progress_callback is not None:
            payload = dict(update)
            payload.update(artifact_paths)
            progress_callback(payload)

    baseline_summary = {"easy": 0.0, "medium": 0.0, "hard": 0.0, "overall": 0.0}
    trained_summary = {"easy": 0.0, "medium": 0.0, "hard": 0.0, "overall": 0.0}

    if config.baseline_eval:
        FastLanguageModel.for_inference(model)
        emit_progress(
            {
                "phase": "baseline_eval",
                "status": "running",
                "completed_steps": 0,
                "total_steps": int(config.max_steps),
            }
        )
        baseline_summary = run_policy_evaluation(
            model=model,
            tokenizer=tokenizer,
            generator_mode=controller.mode,
            deterministic_generator=not config.non_deterministic_generator,
            episodes=config.evaluation_episodes,
            logger=logger,
            phase="baseline_eval",
            max_new_tokens=config.eval_max_new_tokens,
        )
        print(f"[baseline_eval] {json.dumps(baseline_summary)}")
        emit_progress(
            {
                "phase": "baseline_eval",
                "status": "completed",
                "baseline_summary": baseline_summary,
            }
        )

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        max_steps=config.max_steps,
        logging_steps=1,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        report_to=[],
    )

    class ProgressCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
            del args, control, kwargs
            emit_progress(
                {
                    "phase": "train",
                    "status": "running",
                    "completed_steps": int(getattr(state, "global_step", 0)),
                    "total_steps": int(config.max_steps),
                    "current_epoch": float(getattr(state, "epoch", 0.0) or 0.0),
                }
            )

        def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
            del args, control, kwargs
            emit_progress(
                {
                    "phase": "train",
                    "status": "running",
                    "completed_steps": int(getattr(state, "global_step", 0)),
                    "total_steps": int(config.max_steps),
                    "current_epoch": float(getattr(state, "epoch", 0.0) or 0.0),
                }
            )

        def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
            del args, control, kwargs
            emit_progress(
                {
                    "phase": "train",
                    "status": "completed",
                    "completed_steps": int(getattr(state, "global_step", 0)),
                    "total_steps": int(config.max_steps),
                    "current_epoch": float(getattr(state, "epoch", 0.0) or 0.0),
                }
            )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[build_reward_func(curriculum, controller, logger, progress_callback)],
        args=training_args,
        train_dataset=build_dataset(config.dataset_size, controller, curriculum),
        callbacks=[ProgressCallback()],
    )
    trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if config.baseline_eval:
        FastLanguageModel.for_inference(model)
        emit_progress(
            {
                "phase": "trained_eval",
                "status": "running",
                "completed_steps": int(config.max_steps),
                "total_steps": int(config.max_steps),
            }
        )
        trained_summary = run_policy_evaluation(
            model=model,
            tokenizer=tokenizer,
            generator_mode=controller.mode,
            deterministic_generator=not config.non_deterministic_generator,
            episodes=config.evaluation_episodes,
            logger=logger,
            phase="trained_eval",
            max_new_tokens=config.eval_max_new_tokens,
        )
        print(f"[trained_eval] {json.dumps(trained_summary)}")
        print_evaluation_summary(baseline_summary, trained_summary)
        emit_progress(
            {
                "phase": "trained_eval",
                "status": "completed",
                "trained_summary": trained_summary,
            }
        )

    csv_path = logger.write_csv()
    trace_artifact_paths = logger.finalize_trace_artifacts(
        reward_curve_csv=csv_path,
        final_metrics={
            "baseline_summary": baseline_summary,
            "trained_summary": trained_summary,
            "completed_steps": int(config.max_steps),
        },
    )
    logger.close()
    print(f"[artifacts] reward curve CSV written to {csv_path}")

    return {
        "config": config.to_dict(),
        "output_dir": str(output_dir.resolve()),
        "reward_curve_csv": str(csv_path.resolve()),
        **trace_artifact_paths,
        "baseline_summary": baseline_summary,
        "trained_summary": trained_summary,
        "completed_steps": int(config.max_steps),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRPO training entrypoint for the ADAPT DSA environment.")
    parser.add_argument("--model-name", default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument("--output-dir", default="outputs_v3")
    parser.add_argument("--dataset-size", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--disable-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--baseline-eval", action="store_true")
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--eval-max-new-tokens", type=int, default=512)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="adapt-dsa-tutor")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--trace-logging-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-log-interval-steps", type=int, default=10)
    parser.add_argument(
        "--generator-mode",
        choices=["heuristic", "reward_aware"],
        default="reward_aware",
        help="Use heuristic generation or reward-aware family weighting.",
    )
    parser.add_argument(
        "--non-deterministic-generator",
        action="store_true",
        help="Disable deterministic fallback seeding for generator rollouts.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
