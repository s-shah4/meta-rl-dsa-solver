from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

from env.adapt_env import AdaptEnvironment
from models import AdaptAction


def extract_code(completion: str) -> str:
    text = completion.strip()
    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


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


def build_reward_func(curriculum: CurriculumManager):
    def reward_func(prompts, completions, **kwargs) -> list[float]:
        del prompts, kwargs
        env = AdaptEnvironment()
        rewards: list[float] = []
        successes: list[float] = []
        difficulty = curriculum.current_difficulty()

        for completion in completions:
            env.reset(difficulty=difficulty)
            observation = env.step(AdaptAction(code=extract_code(completion)))
            rewards.append(float(observation.reward))
            successes.append(1.0 if observation.pass_rate == 1.0 else 0.0)

        if successes:
            curriculum.update(sum(successes) / len(successes))

        return rewards

    return reward_func


def build_dataset(size: int) -> list[dict[str, str]]:
    prompt = (
        "Read the problem statement carefully. "
        "Write a Python solution that reads from stdin and prints to stdout."
    )
    return [{"prompt": prompt}] * size


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
        reward_funcs=[build_reward_func(curriculum)],
        args=training_args,
        train_dataset=build_dataset(args.dataset_size),
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
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--disable-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
