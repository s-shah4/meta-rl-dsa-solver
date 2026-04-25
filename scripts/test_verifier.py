from verifier.verifier import verify


test_cases = [
    ("5\n", "10"),
    ("0\n", "0"),
    ("-3\n", "-6"),
    ("100\n", "200"),
    ("999\n", "1998"),
]


correct_code = """
n = int(input())
print(n * 2)
"""

wrong_code = """
n = int(input())
print(n)
"""

for name, code in [
    ("correct", correct_code),
    ("wrong", wrong_code),
]:
    reward, info = verify(code, test_cases)

    print("\nCASE:", name)
    print("Reward:", reward)
    print("Pass rate:", info["pass_rate"])
    print("Passed:", info["passed"], "/", info["total"])