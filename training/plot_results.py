from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = value
                    continue
                value = value.strip()
                if value == "":
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = float(value) if "." in value else int(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
        return rows


def rolling_mean(values: list[float], window: int) -> list[float]:
    output: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        output.append(sum(chunk) / len(chunk))
    return output


def plot_reward_curve(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    train_rows = [row for row in rows if row.get("phase") == "train"]
    steps = [int(row["step"]) for row in train_rows]
    rewards = [float(row["episode_reward"]) for row in train_rows]
    reward_smooth = rolling_mean(rewards, window=20)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, rewards, alpha=0.25, label="Episode reward")
    plt.plot(steps, reward_smooth, linewidth=2, label="20-step moving average")
    plt.xlabel("Training step")
    plt.ylabel("Reward")
    plt.title("ADAPT Training Reward Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reward_curve.png", dpi=200)
    plt.close()


def plot_pass_rate_by_difficulty(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    train_rows = [row for row in rows if row.get("phase") == "train"]
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in train_rows:
        grouped[str(row["difficulty_tier"])].append((int(row["step"]), float(row["pass_rate"])))

    plt.figure(figsize=(10, 5))
    for difficulty in ("easy", "medium", "hard"):
        points = grouped.get(difficulty, [])
        if not points:
            continue
        steps = [step for step, _ in points]
        values = [value for _, value in points]
        smooth = rolling_mean(values, window=10)
        plt.plot(steps, smooth, linewidth=2, label=difficulty.title())

    plt.xlabel("Training step")
    plt.ylabel("Pass rate")
    plt.title("Pass Rate by Difficulty Tier")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pass_rate_by_difficulty.png", dpi=200)
    plt.close()


def plot_family_productivity(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    train_rows = [row for row in rows if row.get("phase") == "train"]
    productivity_columns = [key for key in train_rows[0].keys() if str(key).startswith("family_productivity__")]
    if not productivity_columns:
        return

    ranked_columns = sorted(
        productivity_columns,
        key=lambda column: float(train_rows[-1].get(column, 0.0)),
        reverse=True,
    )[:8]

    plt.figure(figsize=(11, 6))
    steps = [int(row["step"]) for row in train_rows]
    for column in ranked_columns:
        family = column.split("__", 1)[1]
        values = [float(row.get(column, 0.0)) for row in train_rows]
        plt.plot(steps, values, linewidth=2, label=family)

    plt.xlabel("Training step")
    plt.ylabel("Family productivity EMA")
    plt.title("Reward-Aware Family Productivity Over Training")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "family_productivity.png", dpi=200)
    plt.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot ADAPT reward and curriculum artifacts from reward_curve.csv.")
    parser.add_argument("csv_path", help="Path to reward_curve.csv")
    parser.add_argument("--output-dir", default=None, help="Directory for PNG outputs. Defaults to the CSV directory.")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    plot_reward_curve(rows, output_dir)
    plot_pass_rate_by_difficulty(rows, output_dir)
    plot_family_productivity(rows, output_dir)
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
