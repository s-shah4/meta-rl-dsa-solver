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

for name, code in [
    ("correct", correct_code),
    ("wrong", wrong_code),
    ("invalid_output", invalid_output_code),
    ("timeout", timeout_code),
    ("runtime_error", runtime_error_code),
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
