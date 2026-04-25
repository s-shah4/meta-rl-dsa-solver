# meta-rl-dsa-solver

ADAPT (Adversarial DSA Tutor) is a minimal reinforcement learning environment for DSA code-generation tasks.

The current implementation is V1: direct Python usage, no FastAPI, multiple test cases, hidden tests, subprocess execution, and verifier-based rewards.

## Usage

```python
from env.adapt_env import AdaptEnv

env = AdaptEnv()

obs = env.reset()
result = env.step("n=int(input())\nprint(n*2)")

reward = result["reward"]
```

Flow:

```text
model -> generates code -> env.step(code) -> executor runs code -> verifier evaluates -> env returns result
```

## Files

- `env/adapt_env.py`: reset/step orchestration only
- `env/executor.py`: subprocess execution with a 2 second timeout
- `env/test_cases.py`: problem definition plus visible and hidden test cases

## Observation

`reset()` returns:

```python
{
    "problem": str,
    "input_format": str,
    "constraints": str,
    "examples": list,
    "visible_tests": list,
}
```

Hidden tests are kept inside the environment and are not shown in the observation.

## Step Result

`step(code)` returns:

```python
{
    "reward": float,
    "done": bool,
    "feedback": str,
    "pass_rate": float,
}
```

## Verifier Requirement

`env.step(code)` calls:

```python
from verifier.verifier import verify

reward, metadata = verify(code, test_cases)
```

The verifier should return:

```python
(
    1.0,
    {
        "pass_rate": 1.0,
        "feedback": "All tests passed. Pass rate: 1.00",
    },
)
```

If `metadata` does not include `pass_rate` or `feedback`, the environment computes fallback values from executor results.

## Smoke Checks

From this directory:

```powershell
cd C:\Users\kaust\PycharmProjects\meta-rl-dsa-solver
```

Check reset and visible/hidden split:

```powershell
python -B -c "from env.adapt_env import AdaptEnv; env=AdaptEnv(); print(env.reset()); print(len(env.visible_tests), len(env.hidden_tests))"
```

Expected split:

```text
3 5
```

Check executor directly:

```powershell
python -B -c "from env.executor import run_code; print(run_code('n=int(input())\nprint(n*2)', '5\n'))"
```

Expected output:

```python
{'stdout': '10\n', 'stderr': '', 'exit_code': 0}
```

Once `verifier/verifier.py` exists, check the full environment:

```powershell
python -B -c "from env.adapt_env import AdaptEnv; env=AdaptEnv(); env.reset(); print(env.step('n=int(input())\nprint(n*2)'))"
```

Check a wrong answer:

```powershell
python -B -c "from env.adapt_env import AdaptEnv; env=AdaptEnv(); env.reset(); print(env.step('n=int(input())\nprint(n+2)'))"
```
