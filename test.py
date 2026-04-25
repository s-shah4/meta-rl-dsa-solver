from __future__ import annotations

from scripts.test_dataset_mode import main as run_dataset_mode_smoke
from scripts.test_env import main as run_env_smoke
from scripts.test_space_api import main as run_space_api_smoke
from scripts.test_training_config import main as run_training_config_smoke
from scripts.test_trace_logging import main as run_trace_logging_smoke
from scripts.test_verifier import test_cases
from verifier.verifier import verify


def main() -> None:
    run_dataset_mode_smoke()
    run_env_smoke()
    run_space_api_smoke()
    run_training_config_smoke()
    run_trace_logging_smoke()

    reward, info = verify(
        "n=int(input())\nnums=list(map(int,input().split()))\nprint(sum(x for x in nums if x % 2 == 0))",
        test_cases,
    )
    assert reward > 0.8
    assert info["pass_rate"] == 1.0
    print("Repository smoke tests passed")


if __name__ == "__main__":
    main()
