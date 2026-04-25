---
title: ADAPT DSA Tutor OpenEnv
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - code-generation
---

# ADAPT DSA Tutor OpenEnv

ADAPT, the Adversarial DSA Tutor, is an OpenEnv-compliant RLVR environment for training code-generation agents on small DSA tasks. The agent receives a problem prompt, examples, and visible tests, then submits Python code. The environment runs the code against visible and hidden tests and returns reward, pass-rate metrics, execution status, and feedback.

This repo now focuses on the environment layer only. Verifier work and training scripts are owned separately.

## Why This Environment

The hackathon asks for OpenEnv environments that can improve LLM behavior through verifiable interaction. ADAPT targets a simple but useful skill loop:

```text
agent writes code -> environment executes it -> hidden tests and reward signals score it -> trainer improves the agent
```

The differentiator is curriculum-ready DSA practice: each episode carries a problem id and difficulty tier so training can track per-tier success instead of only aggregate reward.

## OpenEnv Interface

The environment uses the latest OpenEnv API shape:

- `AdaptEnvironment(Environment[AdaptAction, AdaptObservation, AdaptState])`
- `reset()` returns a typed observation.
- `step(action)` accepts an `AdaptAction` with a Python `code` string.
- `state` exposes episode id, step count, current problem id, difficulty, and recent metrics.

`openenv.yaml` points to:

```yaml
app: server.app:app
port: 7860
```

## Action

```python
{
    "code": "n = int(input())\nprint(n * 2)"
}
```

## Observation

Reset and step observations include:

- problem statement
- input format
- constraints
- examples
- visible tests
- problem id
- difficulty tier
- feedback
- pass rate, visible pass rate, and hidden pass rate
- syntax/runtime/timeout status
- reward components

Hidden test inputs and expected outputs are never returned in observations.

## Reward

Reward is clipped to `[0.0, 1.0]` and combines multiple environment-level signals:

- correctness from visible and hidden pass rate
- syntax validity
- clean execution
- output format compliance
- timeout penalty
- runtime error penalty
- static safety rejection for dangerous imports such as `os`, `subprocess`, `socket`, `pathlib`, and `shutil`

If `verifier.verifier.verify(code, test_cases)` exists, the environment can use it as an optional reward augmentation. If the verifier is absent, the environment still works using executor-derived reward.

## Local Setup

Use Python `3.10+`.

```powershell
cd C:\Users\kaust\PycharmProjects\meta-rl-dsa-solver
python -m venv .venv
.\.venv\Scripts\pip install -e .
```

For this local machine, the existing checked-out OpenEnv repo can also be used during development:

```powershell
$env:PYTHONPATH="C:\Users\kaust\PycharmProjects\OpenEnv\src;$PWD"
```

## Smoke Tests

Run the local smoke test:

```powershell
python test.py
```

Check syntax:

```powershell
python -m py_compile models.py env\adapt_env.py env\executor.py env\test_cases.py server\app.py
```

Start the OpenEnv server:

```powershell
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Useful endpoints:

- `GET /health`
- `GET /schema`
- `POST /reset`
- `POST /step`
- `GET /state`

Example step request:

```powershell
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d "{\"action\":{\"code\":\"n=int(input())\nprint(n*2)\"}}"
```

Validate with OpenEnv once dependencies are installed:

```powershell
openenv validate .
```

## Hugging Face Spaces

This repo is Docker Space ready:

```powershell
openenv push --repo-id <your-hf-username>/adapt-dsa-tutor
```

Before final submission, add:

- live Hugging Face Space link
- training reward/loss plots from Disha's run
- before/after code example showing a problem the model failed before training and solved after training
- mini-blog or short video link

## Current Problem Bank

The environment includes a lightweight curated bank:

- `easy_double`
- `easy_sum_two`
- `medium_maximum`
- `medium_count_even`
- `hard_reverse_words`

This is intentionally small for submission-minimum stability. Later work can expand it to 30-50 tiered problems without changing the OpenEnv API.
