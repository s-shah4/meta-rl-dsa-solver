from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from env.adapt_env import AdaptEnvironment, MAX_STEPS_PER_EPISODE
from env.generator import DIFFICULTY_LABELS, GeneratorAgent
from models import AdaptAction

SYSTEM_PROMPT = """You are the Solver Agent for ADAPT.
Write only runnable Python code.
The program must read from stdin and print to stdout.
If feedback is present, repair your previous solution instead of starting from scratch.
Do not include markdown fences or explanations."""


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
        return self.prompt_registry.pop(prompt)

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
    rows: list[dict[str, Any]] = field(default_factory=list)
    global_step: int = 0
    _wandb_run: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
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
        if self._wandb_run is not None:
            self._wandb_run.log(row, step=self.global_step)
        self.global_step += 1

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


def build_dataset(size: int, controller: GeneratorController, curriculum: CurriculumManager) -> GeneratorRolloutDataset:
    return GeneratorRolloutDataset(size=size, controller=controller, curriculum=curriculum)


def build_reward_func(
    curriculum: CurriculumManager,
    controller: GeneratorController,
    logger: TrainingLogger,
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
                },
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

        for _ in range(MAX_STEPS_PER_EPISODE):
            prompt = build_solver_prompt(observation.model_dump())
            completion = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            observation = env.step(
                AdaptAction(
                    session_id=env.session_id,
                    code=extract_code(completion),
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


def run_training(args: argparse.Namespace) -> None:
    try:
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel, PatchFastRL
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install `trl` and `unsloth` before running GRPO training."
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    PatchFastRL("GRPO", FastLanguageModel)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.disable_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
    )

    curriculum = CurriculumManager()
    controller = GeneratorController(
        mode="reward_aware" if args.generator_mode == "reward_aware" else "heuristic",
        deterministic=not args.non_deterministic_generator,
    )
    logger = TrainingLogger(
        output_dir=output_dir,
        family_names=controller.family_names,
        use_wandb=not args.disable_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    baseline_summary = {"easy": 0.0, "medium": 0.0, "hard": 0.0, "overall": 0.0}
    trained_summary = {"easy": 0.0, "medium": 0.0, "hard": 0.0, "overall": 0.0}

    if args.baseline_eval:
        FastLanguageModel.for_inference(model)
        baseline_summary = run_policy_evaluation(
            model=model,
            tokenizer=tokenizer,
            generator_mode=controller.mode,
            deterministic_generator=not args.non_deterministic_generator,
            episodes=args.evaluation_episodes,
            logger=logger,
            phase="baseline_eval",
            max_new_tokens=args.eval_max_new_tokens,
        )
        print(f"[baseline_eval] {json.dumps(baseline_summary)}")

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        logging_steps=1,
        bf16=args.bf16,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[build_reward_func(curriculum, controller, logger)],
        args=training_args,
        train_dataset=build_dataset(args.dataset_size, controller, curriculum),
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.baseline_eval:
        FastLanguageModel.for_inference(model)
        trained_summary = run_policy_evaluation(
            model=model,
            tokenizer=tokenizer,
            generator_mode=controller.mode,
            deterministic_generator=not args.non_deterministic_generator,
            episodes=args.evaluation_episodes,
            logger=logger,
            phase="trained_eval",
            max_new_tokens=args.eval_max_new_tokens,
        )
        print(f"[trained_eval] {json.dumps(trained_summary)}")
        print_evaluation_summary(baseline_summary, trained_summary)

    csv_path = logger.write_csv()
    logger.close()
    print(f"[artifacts] reward curve CSV written to {csv_path}")


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
