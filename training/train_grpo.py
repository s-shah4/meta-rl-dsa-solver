from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

from env.adapt_env import AdaptEnvironment
from env.generator import GeneratorAgent
from models import AdaptAction


def extract_code(completion: str) -> str:
    text = completion.strip()
    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


def build_solver_prompt(problem: dict[str, Any]) -> str:
    public_problem = {
        "problem_id": problem["problem_id"],
        "difficulty": problem["difficulty_label"],
        "problem": problem["problem"],
        "input_format": problem["input_format"],
        "constraints": problem["constraints"],
    }
    return (
        "You are the Solver Agent for ADAPT.\n"
        "Read the generated DSA task and reply with only runnable Python code.\n"
        "The program must read from stdin and print to stdout.\n"
        "No markdown, no explanation.\n\n"
        f"{json.dumps(public_problem, indent=2)}"
    )


@dataclass
class CurriculumManager:
    difficulties: list[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    current_idx: int = 0
    success_history: list[float] = field(default_factory=list)
    window_size: int = 10

    def current_difficulty(self) -> str:
        return self.difficulties[self.current_idx]

    def update(self, batch_success_rate: float) -> None:
        self.success_history.append(float(batch_success_rate))
        if len(self.success_history) > self.window_size:
            self.success_history.pop(0)

        moving_average = sum(self.success_history) / len(self.success_history)
        if moving_average > 0.70 and self.current_idx < len(self.difficulties) - 1:
            self.current_idx += 1
            self.success_history.clear()
            print(
                f"[curriculum] promoted to {self.current_difficulty()} "
                f"(moving_success={moving_average:.2f})"
            )
        elif moving_average < 0.25 and self.current_idx > 0:
            self.current_idx -= 1
            self.success_history.clear()
            print(
                f"[curriculum] reduced to {self.current_difficulty()} "
                f"(moving_success={moving_average:.2f})"
            )


@dataclass
class GeneratorController:
    mode: str = "heuristic"
    deterministic: bool = True
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

    def __post_init__(self) -> None:
        self.generator = GeneratorAgent(deterministic=self.deterministic)

    def create_rollout_problem(self, difficulty: str) -> tuple[str, dict[str, Any]]:
        problem = self.generator.generate(difficulty, self.history)
        prompt = build_solver_prompt(problem)
        self.prompt_registry[prompt] = problem
        return prompt, problem

    def resolve_prompt(self, prompt: str) -> dict[str, Any]:
        if prompt not in self.prompt_registry:
            raise KeyError("Prompt was not registered with the generator controller.")
        return self.prompt_registry[prompt]

    def update(self, problem: dict[str, Any], pass_rate: float, generator_reward_signal: float) -> None:
        self.history["recent_pass_rates"].append(round(float(pass_rate), 4))
        self.history["problem_types"].append(problem.get("problem_type", ""))
        self.history["problem_signatures"].append(problem.get("problem_id", ""))
        if self.mode == "reward_aware":
            self.history["generator_rewards"].append(round(float(generator_reward_signal), 4))
        else:
            self.history["generator_rewards"].append(0.0)
        self.history["episode_index"] = int(self.history.get("episode_index", 0)) + 1

        for key in ("recent_pass_rates", "problem_types", "problem_signatures", "generator_rewards"):
            values = self.history[key]
            if len(values) > 50:
                del values[:-50]

    def stats_snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "episodes": self.history["episode_index"],
            "recent_pass_rates": list(self.history["recent_pass_rates"][-5:]),
            "recent_problem_types": list(self.history["problem_types"][-5:]),
            "recent_generator_rewards": list(self.history["generator_rewards"][-5:]),
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


def build_reward_func(curriculum: CurriculumManager, controller: GeneratorController):
    def reward_func(prompts, completions, **kwargs) -> list[float]:
        del kwargs
        env = AdaptEnvironment(generator=controller.generator, generator_mode=controller.mode)
        rewards: list[float] = []
        pass_rates: list[float] = []

        for prompt, completion in zip(prompts, completions):
            problem = controller.resolve_prompt(prompt)
            env.reset(
                difficulty=problem["difficulty_label"],
                generated_problem=problem,
                generator_mode=controller.mode,
            )
            observation = env.step(AdaptAction(code=extract_code(completion)))
            rewards.append(float(observation.reward))
            pass_rates.append(float(observation.pass_rate))
            controller.update(problem, observation.pass_rate, observation.generator_reward_signal)
            print(
                "[rollout]",
                json.dumps(
                    {
                        "problem_id": problem["problem_id"],
                        "problem_type": problem["problem_type"],
                        "difficulty": problem["difficulty_label"],
                        "solver_reward": observation.reward,
                        "pass_rate": observation.pass_rate,
                        "generator_reward": observation.generator_reward_signal,
                        "status": observation.execution_status,
                    }
                ),
            )

        if pass_rates:
            curriculum.update(sum(pass_rates) / len(pass_rates))
            print("[generator]", json.dumps(controller.stats_snapshot()))

        return rewards

    return reward_func


def build_dataset(size: int, controller: GeneratorController, curriculum: CurriculumManager) -> GeneratorRolloutDataset:
    return GeneratorRolloutDataset(size=size, controller=controller, curriculum=curriculum)


def run_training(args: argparse.Namespace) -> None:
    try:
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel, PatchFastRL
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install `trl` and `unsloth` before running GRPO training."
        ) from exc

    PatchFastRL("GRPO", FastLanguageModel)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.disable_4bit,
    )

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
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[build_reward_func(curriculum, controller)],
        args=training_args,
        train_dataset=build_dataset(args.dataset_size, controller, curriculum),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRPO training entrypoint for the ADAPT DSA environment.")
    parser.add_argument("--model-name", default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument("--output-dir", default="outputs_v2")
    parser.add_argument("--dataset-size", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--disable-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--generator-mode",
        choices=["heuristic", "reward_aware"],
        default="heuristic",
        help="Use heuristic generation (V1/V2) or reward-aware bookkeeping for V3-ready training.",
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
