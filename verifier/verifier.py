from verifier.sandbox import run_code
from verifier.metrics import compute_pass_rate


def verify(code: str, test_cases):
    results = []

    for stdin, expected in test_cases:
        ok, output = run_code(code, stdin)

        passed = ok and output.strip() == expected.strip()

        results.append({
            "input": stdin.strip(),
            "expected": expected.strip(),
            "output": output.strip(),
            "passed": passed,
            "error": None if ok else output,
        })

    reward, metrics = compute_pass_rate(results)

    return reward, {
        **metrics,
        "results": results,
    }