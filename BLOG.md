# ADAPT: training a model to debug algorithm solutions instead of guessing once

ADAPT is an OpenEnv environment for reinforcement learning on algorithmic problem solving. The core idea is simple: instead of rewarding a model only for a perfect first answer, we make it solve a programming task, inspect verifier feedback, and repair its code over up to three attempts.

That makes the environment closer to real software work. The agent sees a problem statement, input format, constraints, and visible examples. It writes Python code, gets hidden-test feedback, and is rewarded for improving hidden correctness and eventually meeting an efficiency target.

## What we built

- an OpenEnv-compatible environment with `reset`, `step`, and `state`
- a verifier that executes submitted Python safely and scores hidden correctness, visible correctness, and efficiency
- a reward-aware curriculum that shifts toward the most educational problem families
- a GRPO training pipeline built with Unsloth + Hugging Face TRL

## Why this environment is interesting

Most code-generation tasks only ask whether a model can get the answer immediately. ADAPT focuses on a harder and more realistic behavior: can the model recover from failure, use feedback, and converge toward a correct solution?

The environment includes `20` DSA problem families across easy, medium, and hard tiers, hidden evaluation tests, efficiency-aware rewards, and a curriculum that adapts based on what seems to teach the agent the most.

## Real training run

We trained `Qwen/Qwen2.5-3B-Instruct` with the overnight preset:

- Run ID: `15940d1d-7d8c-4253-8810-2ea934bedee4`
- Optimizer steps: `950`
- Training episodes logged: `7,600`
- Wall-clock time: `8.96 hours`
- Uploaded model revision: `6c957e7c6bdb25ff086775fb8692570aee4501c9`

From the recovered training logs:

- average reward improved from `0.4441` in the first `500` episodes to `0.5951` in the last `500`
- average hidden pass rate improved from `0.4625` to `0.6488`
- completion rate improved from `43.6%` to `59.6%`

Late in training, the model was solving many easy and medium tasks reliably and was noticeably stronger on hard tasks as well:

| Difficulty | Avg reward | Avg hidden pass rate | Completion rate |
| --- | ---: | ---: | ---: |
| Easy | 0.8390 | 0.8477 | 84.38% |
| Medium | 0.7892 | 0.8425 | 78.85% |
| Hard | 0.5182 | 0.5759 | 51.92% |

## Evidence

![Reward curve](artifacts/reward_curve.svg)

![Pass rate by difficulty](artifacts/pass_rate_by_difficulty.svg)

![Family productivity](artifacts/family_productivity.svg)

## Links

- Environment Space: [Dishaaa25/meta-rl-dsa-solver](https://huggingface.co/spaces/Dishaaa25/meta-rl-dsa-solver)
- Live app: [Dishaaa25-meta-rl-dsa-solver.hf.space](https://Dishaaa25-meta-rl-dsa-solver.hf.space)
- Training notebook: [Qwen2.5_(3B)-GRPO.ipynb](https://huggingface.co/spaces/Dishaaa25/meta-rl-dsa-solver/blob/main/Qwen2.5_%283B%29-GRPO.ipynb)
- Trained model: [Dishaaa25/adapt-dsa-tutor-model](https://huggingface.co/Dishaaa25/adapt-dsa-tutor-model)
