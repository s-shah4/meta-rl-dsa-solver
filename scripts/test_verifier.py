from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verifier.verifier import verify


test_cases = [
    {"input": "5\n2 3 4 5 6\n", "output": "12", "is_visible": False},
    {"input": "4\n1 3 5 7\n", "output": "0", "is_visible": False},
    {"input": "6\n-2 -3 -4 -5 -6 -7\n", "output": "-12", "is_visible": False},
    {"input": "3\n0 10 11\n", "output": "10", "is_visible": False},
    {"input": "5\n8 8 8 8 8\n", "output": "40", "is_visible": False},
]

correct_code = """
n = int(input())
nums = list(map(int, input().split()))
print(sum(x for x in nums if x % 2 == 0))
"""

wrong_code = """
n = int(input())
nums = list(map(int, input().split()))
print(sum(nums))
"""

less_optimized_code = """
n = int(input())
nums = list(map(int, input().split()))
evens = [x for x in nums if x % 2 == 0]
print(sum(evens))
"""

invalid_output_code = """
n = int(input())
input()
print()
"""

timeout_code = """
while True:
    pass
"""

runtime_error_code = """
n = int(input())
nums = list(map(int, input().split()))
print(nums[n])
"""

safety_violation_code = """
import os
print(os.listdir("."))
"""

for name, code in [
    ("correct", correct_code),
    ("wrong", wrong_code),
    ("less_optimized", less_optimized_code),
    ("invalid_output", invalid_output_code),
    ("timeout", timeout_code),
    ("runtime_error", runtime_error_code),
    ("safety_violation", safety_violation_code),
]:
    reward, info = verify(code, test_cases)

    print("\nCASE:", name)
    print("Reward:", reward)
    print("Pass rate:", info["pass_rate"])
    print("Passed:", info["passed"], "/", info["total"])
    print("Timeouts:", info["timeout_count"])
    print("Runtime errors:", info["runtime_error_count"])
    print("Invalid output:", info["invalid_output_count"])
    print("Wrong answers:", info["wrong_answer_count"])
    print("Status:", info["execution_status"])
    print("Efficiency:", info.get("efficiency_score"))

reward_optimal, info_optimal = verify(correct_code, test_cases)
reward_less_optimal, info_less_optimal = verify(less_optimized_code, test_cases)
reward_safety, info_safety = verify(safety_violation_code, test_cases)
assert info_optimal["efficiency_score"] > info_less_optimal["efficiency_score"]
assert info_less_optimal["complexity_signals"]["list_comprehensions"] > 0
assert info_optimal["verifier_components"]["hidden_correctness"] == 1.0
assert info_optimal["verifier_components"]["anti_cheat_compliance"] == 1.0
assert reward_safety == 0.0
assert info_safety["execution_status"] == "safety_violation"
