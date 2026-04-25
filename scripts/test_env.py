from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.adapt_env import AdaptEnvironment
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
    assert observation.input_format
    assert observation.constraints
    assert observation.problem_type == "sum_even_numbers"
    assert observation.execution_status == "ready"
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
    assert correct.reward > 0.8, correct.model_dump()
    assert correct.pass_rate == 1.0
    assert correct.execution_status == "completed"

    wrong = env.step(
        AdaptAction(
            code=(
                "n=int(input())\n"
                "nums=list(map(int,input().split()))\n"
                "print(sum(nums))"
            )
        )
    )
    print(wrong)
    assert 0.0 <= float(wrong.reward) < 1.0
    assert wrong.execution_status in {"wrong_answer", "completed"}
    assert wrong.pass_rate < 1.0

    invalid_output = env.step(
        AdaptAction(
            code=(
                "n=int(input())\n"
                "input()\n"
                "print()"
            )
        )
    )
    print(invalid_output)
    assert invalid_output.invalid_output_count > 0
    assert invalid_output.execution_status == "invalid_output_format"

    syntax = env.step(AdaptAction(code="def broken(:\n    pass"))
    print(syntax)
    assert syntax.reward == 0.0
    assert syntax.execution_status == "syntax_error"

    timeout = env.step(AdaptAction(code="while True:\n    pass"))
    print(timeout)
    assert timeout.timeout_count > 0
    assert timeout.execution_status == "timeout"

    unsafe = env.step(AdaptAction(code="import os\nprint(os.listdir('.'))"))
    print(unsafe)
    assert unsafe.reward == 0.0
    assert unsafe.execution_status == "safety_violation"

    assert env.state.step_count == 6
    assert env.state.history["recent_pass_rates"]
    assert_hidden_tests_are_not_exposed(timeout.model_dump())
    print("ADAPT OpenEnv smoke tests passed")


if __name__ == "__main__":
    main()
