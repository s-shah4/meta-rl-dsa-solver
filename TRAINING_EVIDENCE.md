# Training Evidence

This project was trained directly on Hugging Face infrastructure rather than from a separate Colab notebook. For the hackathon submission form, the public Hugging Face Space repository can be used as the "Training Run Notebook URL" because it contains the runnable training script and the evidence artifacts below.

## Public training assets

- Space repository root: [Dishaaa25/meta-rl-dsa-solver](https://huggingface.co/spaces/Dishaaa25/meta-rl-dsa-solver/tree/main)
- Training entrypoint: [training/train_grpo.py](https://huggingface.co/spaces/Dishaaa25/meta-rl-dsa-solver/blob/main/training/train_grpo.py)
- Trained adapter repo: [Dishaaa25/adapt-dsa-tutor-model](https://huggingface.co/Dishaaa25/adapt-dsa-tutor-model)
- Model revision from this run: `6c957e7c6bdb25ff086775fb8692570aee4501c9`

## Confirmed run

- Run ID: `15940d1d-7d8c-4253-8810-2ea934bedee4`
- Status: `succeeded`
- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Optimizer steps: `950`
- Logged training episodes: `7,600`
- Duration: `8.964 hours`
- Generator mode: `reward_aware`
- Dataset size: `1024`
- Checkpoint upload enabled: `true`

## Evidence that training actually happened

- Reward curve image committed to the repo: `artifacts/reward_curve.svg`
- Pass-rate-by-difficulty image committed to the repo: `artifacts/pass_rate_by_difficulty.svg`
- Family-productivity image committed to the repo: `artifacts/family_productivity.svg`
- Structured summary committed to the repo: `artifacts/results_summary.json`

## Key outcomes from the recovered run logs

- Average reward improved from `0.4441` over the first `500` episodes to `0.5951` over the last `500`.
- Average hidden pass rate improved from `0.4625` to `0.6488`.
- Completion rate improved from `43.6%` to `59.6%`.

Final-500-episode breakdown:

| Difficulty | Episodes | Avg reward | Avg hidden pass rate | Completion rate |
| --- | ---: | ---: | ---: | ---: |
| Easy | 32 | 0.8390 | 0.8477 | 84.38% |
| Medium | 104 | 0.7892 | 0.8425 | 78.85% |
| Hard | 364 | 0.5182 | 0.5759 | 51.92% |

## Plots

![Reward curve](artifacts/reward_curve.svg)

![Pass rate by difficulty](artifacts/pass_rate_by_difficulty.svg)

![Family productivity](artifacts/family_productivity.svg)

## Training configuration snapshot

The full lightweight snapshot is committed in `artifacts/training_status_snapshot.json`. Key settings:

- learning rate: `5e-6`
- batch size: `1`
- gradient accumulation steps: `8`
- num generations: `4`
- max prompt length: `1024`
- max completion length: `384`
- LoRA rank: `16`
- LoRA alpha: `32`
- 4-bit loading: `true`

## Why the repo URL is the right form entry

The hackathon form explicitly allows a public Hugging Face repository URL in place of a Colab notebook. Since this team trained on Hugging Face directly, the repository URL is the most accurate submission target: it contains the training code, the evidence plots, and the run summary in one public place.
