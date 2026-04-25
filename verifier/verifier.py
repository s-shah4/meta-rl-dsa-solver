from verifier.sandbox import run_code
from verifier.metrics import compute_pass_rate


def verify(code: str, test_cases):
    results = []

    for test_case in test_cases:
        if isinstance(test_case, dict):
            stdin = str(test_case.get("input", ""))
            expected = str(test_case.get("output", ""))
        else:
            stdin, expected = test_case

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
