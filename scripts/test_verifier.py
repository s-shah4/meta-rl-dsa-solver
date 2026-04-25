from verifier.verifier import verify
test_cases = [
    ("5\n", "10"),
    ("0\n", "0"),
    ("-3\n", "-6"),
]

code = """
n = int(input())
print(n * 2)
"""

reward, info = verify(code, test_cases)

print("Reward:", reward)
print(info)