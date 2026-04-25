def compute_pass_rate(results):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    timeout_count = sum(1 for r in results if r.get("error") == "TIMEOUT")
    error_count = sum(1 for r in results if r.get("error"))

    pass_rate = passed / total if total else 0.0

    # V2 shaped reward
    reward = pass_rate

    # small penalty for unsafe/broken behavior
    if timeout_count > 0:
        reward -= 0.2
    if error_count > 0:
        reward -= 0.1

    reward = max(0.0, min(1.0, reward))

    return reward, {
        "passed": passed,
        "total": total,
        "pass_rate": pass_rate,
        "timeout_count": timeout_count,
        "error_count": error_count,
    }