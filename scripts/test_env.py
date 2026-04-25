from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.adapt_env import AdaptEnvironment, MAX_STEPS_PER_EPISODE
from env.generator import GeneratorAgent
from models import AdaptAction


def assert_hidden_tests_are_not_exposed(payload: dict) -> None:
    text = str(payload)
    assert "test_cases" not in text
    assert "visible_tests" not in text
    assert '"is_visible": True' not in text


def main() -> None:
    env = AdaptEnvironment(generator=GeneratorAgent())
    observation = env.reset(problem_id="sum_even_numbers", difficulty="easy")
    assert observation.problem
    assert "Examples:" in observation.problem
    assert observation.input_format
    assert observation.constraints
    assert observation.problem_type == "sum_even_numbers"
    assert observation.execution_status == "ready"
    assert observation.max_steps == MAX_STEPS_PER_EPISODE
    assert_hidden_tests_are_not_exposed(observation.model_dump())

    correct = env.step(
        AdaptAction(
            code=(
                "n=int(input())\n"
                "nums=list(map(int,input().split()))\n"
                "print(sum(x for x in nums if x % 2 == 0))"
            )
        )
    )
    print(correct)
    assert correct.reward == 1.0, correct.model_dump()
    assert correct.pass_rate == 1.0
    assert correct.execution_status == "completed"
    assert correct.done is True
    assert correct.reward_components["efficiency_score"] >= 0.95

    observation = env.reset(problem_id="running_total", difficulty="easy")
    repair_1 = env.step(
        AdaptAction(
            code=(
                "n=int(input())\n"
                "nums=list(map(int,input().split()))\n"
                "print(sum(nums))"
            )
        )
    )
    print(repair_1)
    assert repair_1.done is False
    assert repair_1.execution_status in {"wrong_answer", "runtime_error", "invalid_output_format"}
    assert "Previous attempt status: ready" in repair_1.feedback

    repair_2 = env.step(
        AdaptAction(
            code=(
                "n=int(input())\n"
                "nums=list(map(int,input().split()))\n"
                "running=0\n"
                "out=[]\n"
                "for x in nums:\n"
                "    running += x\n"
                "    out.append(str(running))\n"
                "print(' '.join(out))"
            )
        )
    )
    print(repair_2)
    assert repair_2.done is True
    assert repair_2.pass_rate == 1.0
    assert repair_2.reward == 0.85
    assert "Previous attempt status:" in repair_2.feedback

    observation = env.reset(problem_id="sum_even_numbers", difficulty="easy")
    less_optimized = env.step(
        AdaptAction(
            code=(
                "n=int(input())\n"
                "nums=list(map(int,input().split()))\n"
                "evens=[x for x in nums if x % 2 == 0]\n"
                "print(sum(evens))"
            )
        )
    )
    print(less_optimized)
    assert less_optimized.pass_rate == 1.0
    assert less_optimized.done is False
    assert less_optimized.reward < 1.0
    assert "can still be optimized further" in less_optimized.feedback

    observation = env.reset(problem_id="sum_even_numbers", difficulty="easy")
    syntax = env.step(AdaptAction(code="def broken(:\n    pass"))
    print(syntax)
    assert syntax.reward == 0.0
    assert syntax.done is False
    assert syntax.execution_status == "syntax_error"

    runtime = env.step(
        AdaptAction(
            code=(
                "n=int(input())\n"
                "nums=list(map(int,input().split()))\n"
                "print(nums[n])"
            )
        )
    )
    print(runtime)
    assert runtime.execution_status == "runtime_error"

    timeout = env.step(AdaptAction(code="while True:\n    pass"))
    print(timeout)
    assert timeout.timeout_count > 0
    assert timeout.execution_status == "timeout"
    assert timeout.done is True

    observation = env.reset(problem_id="sum_even_numbers", difficulty="easy")
    unsafe = env.step(AdaptAction(code="import os\nprint(os.listdir('.'))"))
    print(unsafe)
    assert unsafe.reward == 0.0
    assert unsafe.execution_status == "safety_violation"
    assert unsafe.done is False

    assert env.state.history["attempts"]
    assert_hidden_tests_are_not_exposed(timeout.model_dump())
    print("ADAPT OpenEnv smoke tests passed")


if __name__ == "__main__":
    main()
