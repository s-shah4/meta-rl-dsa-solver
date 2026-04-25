def compute_pass_rate(results):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    pass_rate = passed / total if total else 0.0

    reward = 1.0 if pass_rate == 1.0 else 0.0

    return reward, {
        "passed": passed,
        "total": total,
        "pass_rate": pass_rate,
    }