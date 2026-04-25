from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_grpo import build_training_config, resolve_precision_policy
from verifier.metrics import compute_episode_reward


class FakeCuda:
    def __init__(self, available: bool, bf16_supported: bool) -> None:
        self._available = available
        self._bf16_supported = bf16_supported

    def is_available(self) -> bool:
        return self._available

    def is_bf16_supported(self) -> bool:
        return self._bf16_supported


class FakeTorch:
    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"

    def __init__(self, available: bool, bf16_supported: bool) -> None:
        self.cuda = FakeCuda(available=available, bf16_supported=bf16_supported)


def main() -> None:
    l4_config = build_training_config("l4")
    smoke_config = build_training_config("smoke")

    assert l4_config.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert l4_config.load_in_4bit is True
    assert l4_config.gradient_checkpointing is True
    assert l4_config.num_generations == 4

    assert smoke_config.load_in_4bit is False
    assert smoke_config.gradient_checkpointing is False

    bf16_policy = resolve_precision_policy(l4_config, FakeTorch(available=True, bf16_supported=True))
    assert bf16_policy["precision_mode"] == "bf16"
    assert bf16_policy["load_in_4bit"] is True

    fp16_policy = resolve_precision_policy(l4_config, FakeTorch(available=True, bf16_supported=False))
    assert fp16_policy["precision_mode"] == "fp16"
    assert fp16_policy["load_in_4bit"] is True

    cpu_policy = resolve_precision_policy(smoke_config, FakeTorch(available=False, bf16_supported=False))
    assert cpu_policy["precision_mode"] == "fp32"
    assert cpu_policy["load_in_4bit"] is False

    reward, components = compute_episode_reward(
        pass_rate=1.0,
        step_number=1,
        execution_status="completed",
        previous_pass_rate=0.0,
        done=False,
        efficiency_score=0.94,
        optimization_target_met=False,
    )
    assert reward == 0.94
    assert components["progress_delta"] == 1.0
    print("Training config smoke tests passed")


if __name__ == "__main__":
    main()
