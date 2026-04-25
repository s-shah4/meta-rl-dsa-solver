from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.adapt_env import AdaptEnvironment
from models import AdaptAction

env = AdaptEnvironment()

# Test Case 1: Easy Double
print("--- Testing Easy Double ---")
obs = env.reset(problem_id="easy_double")
print(f"Problem: {obs.problem}")

# Simulate LLM providing code that reads from stdin (as required by your env)
sample_code = """
import sys
for line in sys.stdin:
    n = int(line.strip())
    print(n * 2)
"""

action = AdaptAction(code=sample_code)
obs = env.step(action)

print(f"Reward: {obs.reward}")
print(f"Pass Rate: {obs.pass_rate}")
print(f"Feedback: {obs.feedback}")
print(f"Components: {obs.reward_components}")
