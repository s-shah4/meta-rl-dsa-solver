from __future__ import annotations

from env.adapt_env import AdaptEnvironment
from models import AdaptAction, AdaptObservation


def assert_hidden_tests_are_not_exposed(payload: dict) -> None:
    text = str(payload)
    assert "hidden_tests" not in text
    assert "-1000000000" not in text


def main() -> None:
    env = AdaptEnvironment()
    observation = env.reset()
    assert isinstance(observation, AdaptObservation)
    assert observation.visible_tests
    assert observation.problem_id == "easy_double"
    assert_hidden_tests_are_not_exposed(observation.model_dump())

    correct = env.step(AdaptAction(code="n=int(input())\nprint(n*2)"))
    print(correct)
    assert correct.reward == 1.0, correct.model_dump()
    assert correct.pass_rate == 1.0

    wrong = env.step(AdaptAction(code="n=int(input())\nprint(n+2)"))
    print(wrong)
    assert 0.0 <= float(wrong.reward) < 1.0
    assert wrong.pass_rate < 1.0
    assert "Failed" in wrong.feedback

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

    assert env.state.step_count == 5
    assert_hidden_tests_are_not_exposed(timeout.model_dump())
    print("ADAPT OpenEnv smoke tests passed")


if __name__ == "__main__":
    main()



