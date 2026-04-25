from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.trace_logging import TraceArtifactLogger


def main() -> None:
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        logger = TraceArtifactLogger(
            run_id="run-123",
            output_dir=output_dir,
            training_config={"max_steps": 6, "model_name": "demo-model"},
            model_identifiers={"model_name": "demo-model", "generator_mode": "reward_aware"},
            system_prompt="You are the Solver Agent.",
            checkpoint_interval_steps=2,
        )

        manifest_path = output_dir / "logs" / "run_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["run_id"] == "run-123"
        assert manifest["training_config"]["max_steps"] == 6

        logger.log_event(
            {
                "phase": "train",
                "step": 0,
                "train_episode_index": 1,
                "problem_id": "sum_even_numbers_1",
                "problem_family": "sum_even_numbers",
                "difficulty": "easy",
                "teacher_prompt": "Problem: sum the even numbers",
                "solver_completion": "print(sum(x for x in nums if x % 2 == 0))",
                "extracted_code": "print(sum(x for x in nums if x % 2 == 0))",
                "reward": 0.94,
                "pass_rate": 1.0,
                "visible_pass_rate": 1.0,
                "execution_status": "completed",
                "efficiency_score": 0.94,
                "optimization_hints": ["Avoid materializing temporary containers."],
                "feedback": "All hidden tests passed, but the solution can still be optimized further.",
            }
        )
        logger.record_progress(
            {
                "phase": "train",
                "completed_steps": 2,
                "total_steps": 6,
                "remaining_steps": 4,
                "progress_ratio": 0.3333,
                "current_epoch": 2.0,
                "current_difficulty": "easy",
                "curriculum_level": 1,
                "train_episode_index": 1,
                "last_problem_id": "sum_even_numbers_1",
                "last_problem_family": "sum_even_numbers",
                "last_execution_status": "completed",
            }
        )
        artifact_paths = logger.artifact_paths()

        events_path = Path(artifact_paths["events_path"])
        latest_checkpoint_path = Path(artifact_paths["latest_checkpoint_path"])
        assert events_path.exists()
        assert latest_checkpoint_path.exists()

        event_line = events_path.read_text(encoding="utf-8").strip().splitlines()[0]
        event = json.loads(event_line)
        assert event["problem_id"] == "sum_even_numbers_1"
        assert event["teacher_prompt"] == "Problem: sum the even numbers"
        assert "training_config" not in event

        checkpoint = json.loads(latest_checkpoint_path.read_text(encoding="utf-8"))
        assert checkpoint["step"] == 2
        assert checkpoint["rolling_metrics"]["avg_reward"] == 0.94
        assert "training_config" not in checkpoint

        reward_curve = output_dir / "reward_curve.csv"
        reward_curve.write_text("step,episode_reward\n0,0.94\n", encoding="utf-8")
        summary_paths = logger.finalize(
            reward_curve_csv=reward_curve,
            final_metrics={"completed_steps": 6},
        )
        summary_path = Path(summary_paths)
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["final_metrics"]["completed_steps"] == 6

    print("Trace logging smoke tests passed")


if __name__ == "__main__":
    main()
