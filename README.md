# meta-rl-dsa-solver

ADAPT (Adversarial DSA Tutor) is a minimal reinforcement learning environment for coding tasks. The current V0 environment is a pure Python class with no API dependency, so it can be used directly from a training loop with `env.reset()` and `env.step(...)`.

## Current V0

- Fixed DSA problem: given an integer `n`, return `n * 2`
- Single test input: `5`
- Expected output: `10`
- Binary reward: `1.0` for correct output, `0.0` otherwise
- Subprocess execution with a 2 second timeout

## Run a Smoke Test

From this directory:

```powershell
cd C:\Users\kaust\PycharmProjects\meta-rl-dsa-solver
python3 -c "from environment import AdaptEnv; env=AdaptEnv(); print(env.reset()); print(env.step('n=int(input()); print(n*2)'))"
```

Expected reward:

```text
1.0
```

## Use in Python

```python
from environment import AdaptEnv

env = AdaptEnv()

obs = env.reset()
print(obs)

code = "n=int(input()); print(n*2)"
result = env.step(code)

print(result)
assert result["reward"] == 1.0
```

## Check Failure Cases

Wrong answer:

```powershell
python3 -c "from environment import AdaptEnv; env=AdaptEnv(); env.reset(); print(env.step('print(0)'))"
```

Timeout:

```powershell
python3 -c "from environment import AdaptEnv; env=AdaptEnv(); env.reset(); print(env.step('while True: pass'))"
```

## Environment Contract

`reset()` returns:

```python
{
    "problem": "Given an integer n, return n * 2",
    "input": "5",
}
```

`step(action: str)` returns:

```python
{
    "observation": "<program output or error>",
    "reward": 1.0,
    "done": True,
    "info": {},
}
```

The implementation keeps the verifier pluggable so later versions can replace the single expected-output check with hidden tests, randomized inputs, or adaptive curriculum logic.
