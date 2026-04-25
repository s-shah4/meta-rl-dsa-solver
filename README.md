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
  - llm-training
---

# ADAPT: Adversarial DSA Programming Tutor

LLMs are getting better at one-shot code generation, but they still struggle with the thing real engineers do all day: read feedback, debug, and repair. ADAPT closes that gap by turning algorithm practice into a self-repair RL environment where the model must improve over multiple attempts instead of guessing once.

## Why ADAPT exists

Most code-generation benchmarks test whether a model can land the answer immediately. They do not test whether the model can recover from partial failure, use examples productively, or adapt as the task distribution changes.

ADAPT is built to stress exactly those capabilities:

- adaptive difficulty across easy, medium, and hard DSA families
- visible examples plus hidden evaluation tests
- multi-step repair with feedback between attempts
- reward-aware problem generation that shifts toward the most educational families
- optional dataset-backed problem loading for competitive-programming style tasks

## Architecture

```text
+------------+     +-----------+     +----------+     +-----------+
| Generator  | --> | Problem   | --> | Solver   | --> | Execution |
+------------+     +-----------+     +----------+     +-----------+
      ^                                                        |
      |                                                        v
      +------------- Curriculum <- Reward <- Verification -----+
```

## What the agent sees, does, and gets rewarded for

The agent sees a plain-English programming problem, the stdin format, constraints, and two worked examples. It writes Python code that reads from stdin and prints to stdout.

The environment executes that code on 10 tests per problem:

- 2 visible tests shown as examples
- 8 hidden tests used for the real pass-rate reward

After each attempt, the environment returns:

- hidden pass rate
- visible pass rate
- execution status such as `completed`, `wrong_answer`, `runtime_error`, or `timeout`
- a compact list of which tests failed
- enough context to try again on the same problem

## Multi-step repair loop

Each episode allows up to 3 attempts on the same problem.

1. Attempt 1: the agent submits a first solution.
2. Feedback: ADAPT reports the current execution status, hidden pass rate, visible pass rate, and which visible/hidden tests failed.
3. Attempt 2 or 3: the agent repairs its code using that feedback.
4. The episode ends early if all hidden tests pass and the solution clears the efficiency target.

Concrete example:

```text
Problem family: running_total

Attempt 1 code:
print(sum(nums))

Feedback:
Attempt 1/3
Previous attempt status: ready
Current execution status: wrong_answer
Hidden pass rate: 0.25
Visible pass rate: 0.50
Failed tests:
- Visible test #2: wrong_answer (expected=5 3 10, got=10)
- Hidden test #1: wrong_answer
- Hidden test #4: wrong_answer

Attempt 2 code:
running = 0
for x in nums:
    running += x
    out.append(str(running))
print(" ".join(out))
```

That repair loop is the core novelty of ADAPT: the model is rewarded for debugging, not just for lucky first drafts.

## Reward function

ADAPT uses hidden correctness as the base signal, then folds in efficiency once a submission is fully correct:

```python
if execution_status in {"syntax_error", "safety_violation", "timeout"}:
    reward = 0.0
elif hidden_pass_rate == 1.0:
    reward = step_discount * (0.6 + 0.4 * efficiency_score)
elif done:
    reward = 0.0
else:
    reward = 0.1 * max(0.0, hidden_pass_rate - previous_hidden_pass_rate)
```

Where:

- `step_discount = 1.00` on attempt 1
- `step_discount = 0.85` on attempt 2
- `step_discount = 0.70` on attempt 3
- `efficiency_score` is computed from time and space signals in `verifier/complexity.py`
- early completion requires `hidden_pass_rate == 1.0` and `efficiency_score >= 0.89`

Additional shaping for the repair loop:

- if a failed non-terminal attempt improves hidden pass rate, reward = `0.1 * delta_pass_rate`
- if the final attempt still fails, reward = `0.0`
- timeouts, syntax errors, and safety violations always get `0.0`
- a fully-correct but still-suboptimal solution is capped below the terminal max until it meets the efficiency target

Examples:

- attempt 1 solves all 8 hidden tests: reward = `1.0`
- attempt 2 solves all 8 hidden tests: reward = `0.85`
- attempt 1 improves from `0.25` to `0.50` hidden pass rate on a retry trajectory: reward = `0.025`
- attempt 3 still fails: reward = `0.0`

### Efficiency signal

When there are at least three probe inputs with distinct size hints, ADAPT measures empirical runtime and memory growth by replaying the submission on a small subset of tests.

- probe inputs are deduplicated, sorted by a numeric size hint parsed from the first token of the first non-empty line, and capped at 5 probes
- if no numeric size hint is available, ADAPT falls back to raw input length
- empirical complexity falls back to the static AST heuristic if the probe run errors, times out, or lacks enough size variation
- this keeps the reward signal stable for standard competitive-programming inputs where the first integer usually represents problem size

## Problem families

ADAPT now covers 20 algorithmic families instead of a tiny fixed bank:

- Easy: `sum_even_numbers`, `range_span`, `count_vowels`, `max_consecutive_ones`, `fizzbuzz_variant`, `running_total`
- Medium: `count_local_peaks`, `longest_non_decreasing_run`, `two_sum_count`, `max_subarray_sum`, `group_anagrams_count`, `balanced_brackets`, `matrix_diagonal_sum`
- Hard: `smallest_most_frequent`, `reverse_words`, `longest_common_subsequence`, `word_ladder_steps`, `merge_intervals`, `min_coins`, `rotate_matrix_90`

Every family has:

- its own randomized case generator
- 2 visible example tests
- 8 hidden evaluation tests
- a reference solver that auto-generates expected outputs

## Dataset-backed mode

ADAPT can also sample problems from a dataset-backed bank, such as `deepmind/code_contests`, instead of the built-in template families.

- dataset rows are normalized into the same canonical schema used by the deterministic generator
- expected outputs are never truncated; rows with outputs longer than `4096` characters are rejected during normalization
- duplicate normalized inputs are rejected so dataset problems still satisfy the environment validator
- the end-to-end smoke test for this path lives in `scripts/test_dataset_mode.py`

## Self-improving curriculum

ADAPT uses one curriculum authority in training: the `CurriculumManager` inside `training/train_grpo.py`.

- promote threshold: `0.70`
- demote threshold: `0.30`
- moving-average window: `10` episodes

On top of that, the generator tracks `family_productivity`, an EMA of how educational each family is:

```text
family_productivity[family] = 0.9 * old + 0.1 * generator_reward
```

Families that produce pass rates near the learning sweet spot, around `0.5`, become more likely to be sampled via a softmax distribution. This creates a closed loop:

```text
productive families -> more samples -> better learning signal -> updated family productivity
```

That makes ADAPT more than a static benchmark. The environment actively searches for the problems that teach the model the most.

## Results

[INSERT: reward curve plot]

[INSERT: baseline vs trained table]

Recommended artifacts to include here:

- reward curve from `training/reward_curve.csv`
- `reward_curve.png`
- `pass_rate_by_difficulty.png`
- `family_productivity.png`
- one before/after repair example from baseline vs trained evaluation

## How to run

### 1. Install dependencies

```powershell
cd C:\Users\kaust\PycharmProjects\meta-rl-dsa-solver
py -3.11 -m venv .venv
.\.venv\Scripts\pip install -e .
```

For training and plotting, also install your training extras:

```powershell
.\.venv\Scripts\pip install trl unsloth matplotlib wandb
```

Recommended training target:

- Python `3.11`
- Base model `Qwen/Qwen2.5-3B-Instruct`
- Single NVIDIA L4 with 4-bit LoRA + Unsloth GRPO

### 2. Start the OpenEnv server

```powershell
python server\app.py
```

### 3. Reset an environment session

```powershell
curl -X POST http://localhost:7860/reset ^
  -H "Content-Type: application/json" ^
  -d "{\"difficulty\":\"easy\"}"
```

The response includes a `session_id`. Reuse it for `step` and `state`.

### 4. Submit code to `/step`

```powershell
curl -X POST http://localhost:7860/step ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\":\"<SESSION_ID>\",\"code\":\"n=int(input())\nnums=list(map(int,input().split()))\nprint(sum(x for x in nums if x % 2 == 0))\"}"
```

### 5. Inspect current state

```powershell
curl "http://localhost:7860/state?session_id=<SESSION_ID>"
```

### 6. Run training

```powershell
python training\train_grpo.py ^
  --generator-mode reward_aware ^
  --baseline-eval ^
  --output-dir outputs_l4
```

To train against the dataset-backed bank instead of the local templates:

```powershell
python training\train_grpo.py ^
  --generator-mode reward_aware ^
  --use-dataset ^
  --dataset-name deepmind/code_contests ^
  --output-dir outputs_dataset
```

### 7. Plot the training curves

```powershell
python training\plot_results.py outputs_l4\reward_curve.csv
```

### 8. Run the dataset smoke test

```powershell
python scripts\test_dataset_mode.py
```

## Hugging Face Space

This repo is designed to be hosted as an OpenEnv FastAPI Space.

```powershell
openenv push --repo-id <your-hf-username>/adapt-dsa-tutor
```

## Submission checklist

- OpenEnv environment with `Environment`, `reset`, `step`, and `state`
- valid `openenv.yaml`
- Hugging Face Space deployment
- GRPO training script with Unsloth + TRL
- reward and pass-rate plots from a real run
- baseline vs trained evaluation summary
- Colab notebook link for reproducibility

## Links

- HuggingFace Space URL: [HuggingFace Space URL]
- Colab Training Notebook: [Colab Training Notebook]
- HF Blog Post: [HF Blog Post]
- YouTube Demo: [YouTube Demo]
