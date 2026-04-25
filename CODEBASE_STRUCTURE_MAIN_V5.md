# ADAPT Codebase Structure (LLM Working Map)

## Scope And Assumption

This document was generated from the repository rooted at `/Users/dishagoyal/Documents/meta-rl-dsa-solver`.

Important assumption:
- There is no literal `main-v5/` directory in this workspace.
- This document therefore treats the current repository as the codebase you meant by "main-v5".

Goal of this document:
- give an LLM enough structure to navigate, modify, and extend the codebase safely
- explain not just file names, but runtime flow, ownership boundaries, data models, and cross-module dependencies
- surface invariants and hidden coupling points that matter when changing the code

Repository size snapshot:
- Python/source/documentation lines inspected: about 6,981 lines across the main tracked source files
- Core subsystems: `env`, `verifier`, `training`, `server`, top-level client/inference/test entrypoints, and `scripts`
- Empty or currently unused directories: `rewards/`, `outputs/` in this snapshot

## One-Screen Summary

This repo implements **ADAPT**, an adversarial DSA tutoring environment for reinforcement learning and evaluation of code-generating models.

The system is organized around one central loop:

1. A problem family is chosen by the generator.
2. The environment exposes only the public statement plus two visible examples.
3. A model submits Python code.
4. The verifier statically checks the code, runs it in a subprocess sandbox, scores visible and hidden tests, and computes correctness/efficiency signals.
5. The environment converts those signals into repair-oriented feedback and an episode reward.
6. Training uses this environment as a reward source inside a GRPO loop.
7. The server layer exposes reset/step/state plus training/model management endpoints.

The three most important files in the repo are:
- `env/adapt_env.py`: the runtime environment and episode state machine
- `env/generator.py`: the full problem-family registry plus deterministic problem generation
- `training/train_grpo.py`: prompt construction, curriculum logic, reward hookup, evaluation, and GRPO training

## High-Level Architecture

```text
Problem Generator -> AdaptEnvironment -> Verifier -> Reward/Feedback
       |                    |                |             |
       |                    |                |             |
       v                    v                v             v
  Problem family      Session state     Complexity +   Repair-aware
  sampling + cases    + observation     sandboxing     training signal

                Server API / Space Runtime
                         |
                         v
           Training jobs, model loading, code generation
```

## Top-Level Files And Directories

### Root

- `README.md`
  Project narrative, architecture overview, run instructions, and problem-family summary.

- `pyproject.toml`
  Main packaging and dependency definition.
  Registers the installable packages: `env`, `server`, `training`, `verifier`.
  Exposes script entrypoint `server = "server.app:main"`.

- `requirements.txt`
  Inference/server/runtime dependencies only.

- `openenv.yaml`
  OpenEnv metadata:
  - runtime: `fastapi`
  - app: `server.app:app`
  - port: `7860`

- `Dockerfile`
  Python 3.11 slim image, installs package with training extras, runs `uvicorn server.app:app`.

- `app.py`
  Thin re-export of `server.app`.

- `client.py`
  Local Python HTTP client for the environment and training endpoints.

- `models.py`
  Pydantic schemas for actions, observations, and state.

- `inference.py`
  Benchmark/evaluation runner that loops through tasks, calls an OpenAI-compatible API, and logs standardized `[START]/[STEP]/[END]` traces.

- `test.py`
  Repository smoke-test aggregator; calls several script-level tests plus a verifier check.

### `env/`

Owns problem generation, environment state, subprocess execution, and the local Gradio demo.

- `env/adapt_env.py`
- `env/generator.py`
- `env/executor.py`
- `env/test_cases.py`
- `env/app.py`

### `verifier/`

Owns submission validation, execution scoring, reward metrics, and lightweight complexity analysis.

- `verifier/verifier.py`
- `verifier/sandbox.py`
- `verifier/metrics.py`
- `verifier/complexity.py`

### `training/`

Owns prompt formatting, curriculum and generator control, GRPO training, evaluation, trace artifacts, and plotting.

- `training/train_grpo.py`
- `training/trace_logging.py`
- `training/plot_results.py`

### `server/`

Owns the FastAPI service layer and Hugging Face Space runtime/training orchestration.

- `server/app.py`
- `server/runtime.py`
- `server/requirements.txt`

### `scripts/`

Operational and smoke-test scripts.

- `scripts/deploy_and_smoke_train.py`
- `scripts/test_env.py`
- `scripts/test_space_api.py`
- `scripts/test_trace_logging.py`
- `scripts/test_training_config.py`
- `scripts/test_verifier.py`

### Other

- `notebooks/colab_kaggle_smoke_train.ipynb`
  Notebook-based smoke training support.

- `Qwen2.5_(3B)-GRPO.ipynb`
  Large notebook artifact related to GRPO experimentation.

- `rewards/`
  Present but empty in this snapshot.

- `outputs/`
  Present but empty in this snapshot.

## Core Runtime Concepts

### 1. Problem

A generated problem is a dictionary produced mainly by `GeneratorAgent.generate_problem(...)` in `env/generator.py`.

Expected keys:
- `problem_id`
- `problem_type`
- `difficulty`
- `difficulty_label`
- `problem`
- `input_format`
- `constraints`
- `test_cases`
- `visible_problem`
- `generation_mode`
- `validity_bonus`

Important split:
- `test_cases` contains both visible and hidden tests
- `visible_problem` is the sanitized public-facing subset
- hidden tests must never leak into the observation payload

### 2. Episode

An episode is one problem solved over up to `MAX_STEPS_PER_EPISODE = 3` attempts.

Attempt flow:
- attempt 1: initial submission
- attempt 2/3: repair attempt based on feedback
- episode ends early when hidden correctness reaches 1.0 and efficiency target is met, or after step 3

### 3. Reward

There are two related reward layers:

- verifier-level reward in `verifier.metrics.compute_pass_rate(...)`
  Used as raw execution-derived correctness reward.

- environment episode reward in `verifier.metrics.compute_episode_reward(...)`
  This is the more important repair-aware reward used by `AdaptEnvironment`.
  It accounts for:
  - hidden pass rate
  - step discount
  - improvement over previous attempt
  - efficiency score
  - optimization completion target

### 4. Generator Reward Signal

`env.generator.generator_reward(...)` measures how educational a generated problem is, targeting pass rates near 0.5.

This signal is used by training-side reward-aware family sampling:
- productive families become more likely to be sampled
- implemented by `GeneratorController.family_productivity`

## Shared Schemas (`models.py`)

### `AdaptAction`

Fields:
- `session_id`
- `code`

Meaning:
- server-routed step requests require a session id
- local environment calls may omit it because the environment can self-bootstrap

### `AdaptObservation`

This is the main environment-facing payload.

Key fields:
- identity/context:
  - `session_id`
  - `problem_id`
  - `problem_type`
  - `difficulty`
- episode progress:
  - `attempt_number`
  - `max_steps`
  - `done`
- public task content:
  - `problem`
  - `input_format`
  - `constraints`
  - `feedback`
- evaluation signals:
  - `pass_rate`
  - `visible_pass_rate`
  - `hidden_pass_rate`
  - `syntax_valid`
  - `execution_status`
  - `timeout_count`
  - `runtime_error_count`
  - `invalid_output_count`
  - `wrong_answer_count`
  - `format_compliance`
  - `reward_components`
  - `generator_reward_signal`
  - `reward`

### `AdaptState`

Server/state snapshot object; broader than the observation.

Important fields:
- active problem metadata
- generator mode
- generated public problem
- last reward / pass rate / feedback / execution status
- attempt history
- recent metrics
- generator reward signal

## Subsystem 1: Environment (`env/`)

### `env/adapt_env.py`

This is the environment state machine and the single most important runtime module.

#### Main class: `AdaptEnvironment`

Inheritance:
- attempts to inherit from `openenv.core.env_server.interfaces.Environment`
- falls back to a local generic stub if OpenEnv is unavailable

Important constants:
- `MAX_STEPS_PER_EPISODE = 3`
- `TARGET_EFFICIENCY_SCORE = 0.95`

Internal mutable state:
- `self.generator`
- `self.generator_mode`
- `self.session_id`
- `self.problem`
- `self.test_cases`
- `self.last_results`
- `self.history`
- `self.attempt_history`
- `self.previous_execution_status`
- `self.episode_done`
- `self._state` (`AdaptState`)

#### `reset(...)`

Responsibilities:
- optionally update session id / generator mode / difficulty
- choose or load a problem
- copy test cases into the environment
- clear prior attempt/results state
- initialize `AdaptState`
- return an initial observation with public problem text and instructions

Supports:
- forced `problem_id`
- forced `difficulty`
- externally injected `generated_problem`
- family weighting via `family_weights`

#### `step(action, ...)`

Responsibilities:
- lazy-reset if no problem is loaded
- return terminal "call reset" observation if episode already ended
- increment attempt number
- call `_verify_submission`
- compute hidden/visible pass rates and execution status
- compute repair-aware reward via `compute_episode_reward`
- build human-readable feedback
- record metrics/state
- finalize the episode if terminal

Terminal conditions:
- hidden pass rate is 1.0 and efficiency score >= 0.95
- or step number reaches 3

Important nuance:
- a fully correct but less efficient solution may produce `pass_rate == 1.0` but `done == False`
- the agent is then nudged to optimize further

#### `_verify_submission(...)`

Delegates to `verifier.verify(...)`.

Then augments the verifier metadata with:
- diversity bonus
- validity bonus
- generator reward signal

#### `_format_feedback(...)`

Produces repair-facing natural language feedback.

Behavior depends on status:
- syntax/safety violations get short static feedback
- other cases get:
  - attempt number
  - previous status
  - current status
  - hidden and visible pass rates
  - efficiency score
  - failed test summaries
  - optional optimization hints

#### `_public_problem_view(...)`

Critical leakage-control function.

It uses:
- `visible_problem`
- formatted visible examples from `_format_examples()`

This is what gets surfaced to the agent.
Hidden tests stay internal.

#### History and curriculum-relevant state

`AdaptEnvironment` tracks:
- recent pass rates
- recent problem families
- generator rewards
- problem signatures
- episode index

This history influences future generation diversity and can be used by higher-level curriculum logic.

### `env/generator.py`

This file is the full problem bank, generator, case synthesis engine, and reference-solver library.
It is the largest single file in the repo.

#### Core constants

- `VISIBLE_TEST_COUNT = 2`
- `HIDDEN_TEST_COUNT = 8`
- `TOTAL_TEST_CASES = 10`
- `MIN_TEST_CASES = 10`

#### `ProblemTemplate`

Dataclass fields:
- `problem_type`
- `difficulty_tier`
- `title`
- `input_format`
- `constraints`
- `statement_builder`
- `solver`
- `case_builder`

This is the canonical template unit from which all problems are built.

#### `generator_reward(pass_rate, diversity_bonus, validity_bonus)`

Educational-value reward:
- highest when pass rate is near 0.5
- clipped into `[0.0, 1.5]`
- later used by the reward-aware family sampler in training

#### `validate_problem(problem_dict)`

Hard validation gate for generated problems.

Checks:
- required keys exist
- public text fields are non-empty
- scalar difficulty is between 0 and 1
- exactly 10 test cases
- visible/hidden ordering and flags are correct
- test inputs are unique
- output diversity is high enough to avoid degenerate problems

This function is critical if changing generator behavior.

#### `normalize_problem(problem_dict)`

Normalizes string fields and shallow copies test cases / visible problem data.

#### `GeneratorAgent`

Dependency-free deterministic generator by default.

Main behaviors:
- `generate_problem(...)`
- `generate(...)` thin alias
- `_choose_template(...)`
- `_rng_for(...)`
- `_problem_signature(...)`
- `_tier_to_scalar(...)`

Determinism strategy:
- hashes a seed material bundle containing difficulty, recent pass rates, recent problem types, episode index, and family weights
- creates a `random.Random` instance from that hash

This means generation is reproducible for the same history and settings.

#### Template registry

`_build_templates()` registers 20 problem families:

Easy:
- `sum_even_numbers`
- `range_span`
- `count_vowels`
- `max_consecutive_ones`
- `fizzbuzz_variant`
- `running_total`

Medium:
- `count_local_peaks`
- `longest_non_decreasing_run`
- `two_sum_count`
- `max_subarray_sum`
- `group_anagrams_count`
- `balanced_brackets`
- `matrix_diagonal_sum`

Hard:
- `smallest_most_frequent`
- `reverse_words`
- `longest_common_subsequence`
- `word_ladder_steps`
- `merge_intervals`
- `min_coins`
- `rotate_matrix_90`

For each family, the file contains:
- statement text
- input format
- constraints
- visible/hidden randomized case builders
- a reference solver used to derive expected outputs

#### Bottom half of the file

The remainder of `env/generator.py` contains:
- case factory helpers for each family
- solver implementations for each family
- parsers for input formats
- random instance builders for bracket/path/word-ladder style tasks
- formatting helpers for arrays, targets, intervals, matrices, coins, etc.

Practical meaning:
- if you add a new problem family, most of the work belongs here
- the generator is intentionally self-contained; it does not depend on external datasets

### `env/executor.py`

Owns actual subprocess execution of user code.

Execution strategy:
- writes submission code to a temporary file under `.adapt_tmp` or `ADAPT_TMP_DIR`
- runs `python -I -S submission.py`
- feeds stdin
- captures stdout/stderr
- enforces timeout
- truncates outputs to `OUTPUT_LIMIT_BYTES`

Important constants:
- timeout: `1.0s`
- memory limit: `512 MB`
- output limit: `256 KB`

Linux-only hardening:
- RLIMIT for address space, CPU, file size, and file descriptors

Portable hardening:
- isolated Python mode
- stripped environment variables
- temp HOME/TMP

### `env/test_cases.py`

Convenience layer over the generator.

Main functions:
- `load_problem_bank()`
  Builds one representative problem per template for task listing.
- `load_problem(problem_id=None, difficulty=None)`
  Loads or samples a single problem.
- `get_test_cases(...)`
  Returns copied test cases.
- `split_test_cases(...)`
  Splits visible vs hidden tests.

Server and inference logic depend on this file for task enumeration.

### `env/app.py`

Gradio demo app, separate from the FastAPI/OpenEnv server.

Global state:
- `TRAINING_MANAGER = SpaceTrainingManager()`
- `SESSIONS: dict[str, AdaptEnvironment]`

Main interactive functions:
- `_get_env(...)`
- `_problem_markdown(...)`
- `sample_problem(...)`
- `evaluate_submission(...)`
- `model_attempt(...)`

Use case:
- manual exploration of sampled problems
- manual verification of code
- comparison against the current loaded model's generated code

Notable design choice:
- `model_attempt(...)` calls `TRAINING_MANAGER.generate_code(...)`, then immediately feeds that code back through `evaluate_submission(...)`
- so the UI path uses the exact same verifier/environment loop as manual submissions

## Subsystem 2: Verification (`verifier/`)

### `verifier/verifier.py`

Central verification orchestration.

Function: `verify(code, test_cases, step_number=1)`

Pipeline:
1. `validate_code(code)` from `verifier.sandbox`
2. `analyze_code_complexity(code)` from `verifier.complexity`
3. If syntax/safety fails:
   - compute status/reward with empty results
   - return immediately
4. Else execute against each test case via `verifier.sandbox.run_code(...)`
5. Build per-test result records
6. Compute pass-rate metrics with `compute_pass_rate(...)`
7. Build hidden-facing feedback summary

Per-test result record includes:
- index
- status
- passed
- format_ok
- stdout/stderr
- expected/input only for visible tests
- timed_out
- exit_code
- duration_ms
- sandbox metadata
- visibility

Security boundary:
- visible tests expose expected/actual/input in the result metadata
- hidden tests intentionally suppress those values

### `verifier/sandbox.py`

Static validation plus runtime delegation.

Forbidden imports:
- `ctypes`
- `os`
- `pathlib`
- `resource`
- `shutil`
- `signal`
- `socket`
- `subprocess`

Forbidden calls:
- `__import__`
- `breakpoint`
- `compile`
- `eval`
- `exec`
- `open`

Main functions:
- `validate_code(code)`
- `run_code(code, stdin, timeout=1)`

Important nuance:
- runtime isolation is actually implemented in `env.executor.run_code`
- this file adds AST-level policy checks before execution

### `verifier/metrics.py`

Owns reward math.

Constants:
- `STEP_DISCOUNTS = {1: 1.0, 2: 0.85, 3: 0.70}`
- `TERMINAL_ZERO_STATUSES = {"syntax_error", "safety_violation", "timeout"}`

Main functions:
- `step_discount(...)`
- `compute_reward(...)`
- `compute_pass_rate(...)`
- `compute_episode_reward(...)`

#### `compute_pass_rate(...)`

Produces:
- pass counts
- hidden/visible pass rates
- execution status classification
- format compliance
- reward components
- verifier components

Execution status priority order:
- syntax error
- safety violation
- precheck status
- timeout
- runtime error
- invalid output format
- wrong answer
- completed

#### `compute_episode_reward(...)`

This is the repair-aware environment reward.

Behavior:
- syntax/safety/timeout => `0.0`
- correct solution => discounted reward blended with efficiency score
- incorrect but non-terminal and improved => small progress reward (`0.1 * delta`)
- final failed attempt => `0.0`

This function is one of the most behavior-critical points in the codebase.

### `verifier/complexity.py`

Static efficiency heuristic.

Signals tracked:
- nested loop depth
- list/set/dict comprehensions
- generator expressions
- sorting calls
- materialized builtin inputs

Outputs:
- `time_complexity_score`
- `space_complexity_score`
- `efficiency_score`
- `optimization_hints`
- `complexity_signals`

Role in the system:
- not a formal complexity analyzer
- acts as a lightweight shaping mechanism for "correct but inefficient" solutions

## Subsystem 3: Training (`training/`)

### `training/train_grpo.py`

This is the second-most important file after `env/generator.py`.
It owns the full RL training path.

#### Global constants

- `SYSTEM_PROMPT`
  Shared solver instruction string.

- `CRITICAL_PROJECTION_NAMES`
  Used during precision audits.

- `SMOKE_PREFERRED_PRECISION = "fp16"`

#### `TrainingConfig`

Dataclass with the full trainable configuration surface.

Important fields:
- model/runtime:
  - `model_name`
  - `output_dir`
  - `load_in_4bit`
  - `gradient_checkpointing`
  - `bf16`
  - `save_merged_model`
- data/training:
  - `dataset_size`
  - `max_steps`
  - `batch_size`
  - `gradient_accumulation_steps`
  - `num_generations`
  - `learning_rate`
- token lengths:
  - `max_seq_length`
  - `max_prompt_length`
  - `max_completion_length`
- LoRA:
  - `lora_rank`
  - `lora_alpha`
- evaluation:
  - `baseline_eval`
  - `evaluation_episodes`
  - `eval_max_new_tokens`
- logging/control:
  - `disable_wandb`
  - `wandb_project`
  - `wandb_run_name`
  - `generator_mode`
  - `non_deterministic_generator`
  - `trace_logging_enabled`
  - `checkpoint_log_interval_steps`

#### Training presets

Defined in `TRAINING_PRESETS`:
- `smoke`
- `l4`
- `default`

`smoke`:
- tiny, CPU/fallback-friendly
- no 4-bit
- no gradient checkpointing
- short run

`l4` and `default`:
- intended for real LoRA + 4-bit training
- default model: `Qwen/Qwen2.5-3B-Instruct`

#### Prompt helpers

- `extract_code(...)`
- `format_examples(...)`
- `build_solver_prompt(...)`
- `build_prompt_from_problem(...)`

Important prompt shape:
- problem id/family/difficulty
- attempt number
- problem statement with examples
- input format
- constraints
- feedback

This exact shape is reused by training, evaluation, and server-side model generation.

#### Config helpers

- `build_training_config(...)`
- `namespace_to_config(...)`

These are used both by CLI and by the server runtime manager.

#### `CurriculumManager`

Adaptive difficulty controller.

Defaults:
- difficulties: `easy`, `medium`, `hard`
- moving window: `10`
- promote threshold: `0.70`
- demote threshold: `0.30`

Behavior:
- tracks recent episode pass rates
- promotes/demotes based on moving average
- clears history after each level change

#### `GeneratorController`

Training-side wrapper over `GeneratorAgent`.

Responsibilities:
- sample problems for a target difficulty
- register prompt -> problem mapping
- compute reward-aware family weights
- track family productivity EMA
- update generator-side history after each episode

Key feature:
- in `reward_aware` mode, family selection is biased by a softmax over `family_productivity / temperature`

#### `GeneratorRolloutDataset`

Minimal dataset adapter used by GRPO.

Each item:
- samples a new problem at the current curriculum difficulty
- returns a dict with only `{"prompt": prompt}`

This means the dataset is not static; prompts are generated online.

#### `TrainingLogger`

Owns:
- in-memory reward/event rows
- optional Weights & Biases logging
- optional trace artifacts through `TraceArtifactLogger`
- CSV writing

Every logged row includes:
- phase
- episode reward
- pass rate
- visible pass rate
- difficulty
- family
- curriculum level
- execution status
- attempt number
- family productivity columns
- extra metadata like prompt/completion/code/feedback

#### `build_reward_func(...)`

This is the bridge between GRPOTrainer and the environment.

For each `(prompt, completion)` pair:
- resolve original problem from prompt registry
- spin up `AdaptEnvironment`
- reset with the exact generated problem
- run one environment step using extracted code
- record reward
- update generator controller
- update curriculum
- log event
- emit progress callback payload

This function is the key glue layer for RL.

#### `generate_completion(...)`

Shared inference helper for local model objects.

Responsibilities:
- render chat template if tokenizer supports it
- choose generation device from HF device map or model device
- call `model.generate(...)`
- decode only newly generated tokens

Used by:
- evaluation
- server runtime generation

#### `run_policy_evaluation(...)`

Offline evaluator over a schedule of easy/medium/hard episodes.

Loop:
- sample a problem
- run up to 3 repair attempts
- log final result
- aggregate pass rates by tier

Outputs summary:
- easy
- medium
- hard
- overall

#### Precision/runtime utilities

- `get_runtime_versions(...)`
- `validate_runtime_versions(...)`
- `resolve_precision_policy(...)`
- `normalize_model_precision(...)`
- `audit_critical_module_precision(...)`

These guardrails exist because the training path is tuned around Unsloth + GRPO + mixed precision.

#### `run_training(...)`

The full training entrypoint.

Major stages:
1. Normalize config
2. Import `torch`, `trl`, `unsloth`, and transformers callback support
3. Create output directory
4. Patch Unsloth RL stack
5. Resolve runtime versions and precision policy
6. Load base model/tokenizer via `FastLanguageModel.from_pretrained`
7. Apply LoRA with `FastLanguageModel.get_peft_model`
8. Run precision audit(s)
9. Create curriculum, generator controller, and logger
10. Optionally run baseline evaluation
11. Build `GRPOConfig`
12. Build `GRPOTrainer` with online prompt dataset and environment reward function
13. Train
14. Save adapter or merged model
15. Optionally run trained evaluation
16. Write reward CSV and finalize trace artifacts
17. Return a summary payload containing paths and metrics

Output summary includes:
- config
- runtime versions
- precision mode/policy/audits
- `output_dir`
- `reward_curve_csv`
- trace artifact paths
- baseline/trained summaries
- completed steps

#### CLI

`build_parser()` exposes the training surface as a command-line interface.

Entrypoint:
- `python training/train_grpo.py ...`

### `training/trace_logging.py`

Structured artifact logger for Space/server integration and post-run analysis.

Main class: `TraceArtifactLogger`

Artifacts under `output_dir/logs/`:
- `run_manifest.json`
- `events.jsonl`
- `latest_checkpoint.json`
- `checkpoint_step_XXXXX.json`
- `run_summary.json`

Tracked rolling metrics:
- average reward
- average pass rate
- average efficiency score

Used by:
- `TrainingLogger` in `training/train_grpo.py`
- surfaced through `server.runtime.SpaceTrainingManager`

### `training/plot_results.py`

Offline plotting tool for `reward_curve.csv`.

Outputs:
- `reward_curve.png`
- `pass_rate_by_difficulty.png`
- `family_productivity.png`

Important note:
- `plot_pass_rate_by_difficulty(...)` groups rows by `difficulty_tier`
- the function then iterates over `"easy"`, `"medium"`, `"hard"`
- this assumes the CSV contains string difficulty labels, which it currently does

## Subsystem 4: Server And Space Runtime (`server/`)

### `server/app.py`

FastAPI surface for OpenEnv plus training/model utilities.

Global state:
- `SESSIONS`
- `SESSION_LAST_ACCESSED`
- `TRAINING_MANAGER = SpaceTrainingManager()`
- `TASKS` generated from `env.test_cases.load_problem_bank()`

Session policy:
- TTL: 30 minutes
- unknown/expired sessions raise 404

#### Request models

- `ResetRequest`
- `TrainRequest`
- `RunTrainedPolicyRequest`
- `GenerateCodeRequest`

#### Informational endpoints

- `GET /`
  Root metadata plus training/model status and active session count
- `GET /health`
- `GET /metadata`
- `GET /tasks`
- `GET /schema`
- `GET /train/status`
- `GET /model/status`

#### Environment endpoints

- `POST /reset`
  Creates a new `AdaptEnvironment`, stores it in `SESSIONS`, returns initial observation.

- `POST /step`
  Accepts either raw action JSON or `{ "action": ... }`.
  Requires `session_id`.
  Returns:
  - `observation`
  - `reward`
  - `done`
  - compact `info`

- `GET /state`
  Returns `AdaptState`.

#### Model/training endpoints

- `POST /train`
  Starts async training via `SpaceTrainingManager.start_training(...)`

- `POST /run-trained-policy`
  Runs current model through a full 3-attempt environment episode

- `POST /generate-code`
  One-shot generation for a supplied problem payload

#### MCP endpoint

- `POST /mcp`
  Stub only; always returns "not implemented"

### `server/runtime.py`

Owns asynchronous training and model-loading behavior for a Space-like deployment.

This file has two major classes:
- `SpaceModelRegistry`
- `SpaceTrainingManager`

It also defines two dataclasses used as persisted status:
- `ModelState`
- `TrainingJobState`

#### `ModelState`

Tracks:
- whether a model is loaded
- whether the active generation source is `trained`, `base`, or `unavailable`
- repo id / local path / revision
- base model name
- load time
- last error

#### `TrainingJobState`

Tracks:
- job status and run id
- config
- timestamps
- artifact paths
- uploaded model revision
- logs location
- current phase and progress
- precision/runtime audit metadata
- last seen train metrics
- baseline/trained summaries
- error and traceback

This is what backs `/train/status`.

#### `SpaceModelRegistry`

Primary responsibilities:
- load base model when no trained model is available
- load trained artifact from local disk or Hugging Face Hub
- choose generation stack
- fallback from trained model to base model if trained generation fails
- run policy episodes and one-shot code generation

Important methods:
- `_require_runtime_dependencies(...)`
- `load_base_model()`
- `load_from_local(...)`
- `load_latest_from_hub()`
- `run_policy(...)`
- `generate_code(...)`

Environment variables it cares about:
- `HF_MODEL_REPO_ID`
- `HF_TOKEN`
- `BASE_MODEL_NAME`
- `MODEL_NAME`

Fallback behavior:
- if no trained model is loadable, base-model generation can still work
- if trained-model generation errors, it can fall back to base generation and record the fallback reason

#### `SpaceTrainingManager`

Primary responsibilities:
- persist training job state to disk
- prevent concurrent training runs
- build output directory for each run
- spawn background training thread
- upload artifacts to the Hub
- delete local logs after upload
- refresh active model after successful upload

State files:
- status persisted in `training_status.json` under `SPACE_OUTPUT_ROOT` or `/tmp/adapt-space`

Key methods:
- `_restore_status()`
- `_persist_status()`
- `_update_progress(...)`
- `status_payload()`
- `start_training(...)`
- `_run_training_job(...)`
- `_upload_artifacts(...)`
- `_cleanup_local_logs(...)`
- `load_latest_model()`
- `run_trained_policy(...)`
- `generate_code(...)`

Background training behavior:
- `start_training(...)` creates a `TrainingConfig` using `build_training_config(...)`
- output dir is namespaced by `run_id`
- training happens on a daemon thread
- progress updates come from `run_training(..., progress_callback=...)`

Successful completion path:
1. run training
2. upload artifacts to HF model repo
3. delete local logs
4. load latest trained model from Hub
5. mark status as `succeeded`

Failure path:
- mark status as `failed`
- persist traceback
- optionally clean logs first

## Top-Level Utility Entrypoints

### `client.py`

`AdaptEnvClient` wraps HTTP operations for:
- `reset`
- `step`
- `state`
- `train`
- `train_status`
- `model_status`
- `run_trained_policy`
- `generate_code`

Useful for:
- local automation
- notebooks
- scripted experimentation against the FastAPI app

### `inference.py`

OpenAI-compatible external inference runner.

Key behaviors:
- loads tasks from `env.test_cases.load_problem_bank()`
- resets an `AdaptEnvironment` per task
- builds a JSON user payload from the observation
- calls `OpenAI(...).chat.completions.create(...)`
- extracts code
- runs up to 3 repair attempts
- logs machine-readable stdout lines

Important env vars:
- `HF_TOKEN`
- `API_BASE_URL` default `https://router.huggingface.co/v1`
- `MODEL_NAME` default `openai/gpt-oss-120b`

This file is separate from the local model runtime in `server/runtime.py`.
It is for remote API-based evaluation, not local HF model objects.

### `test.py`

Simple "run several smokes in sequence" script.

Calls:
- `scripts.test_env.main`
- `scripts.test_space_api.main`
- `scripts.test_training_config.main`
- `scripts.test_trace_logging.main`
- then a direct verifier correctness check

## Scripts (`scripts/`)

### `scripts/deploy_and_smoke_train.py`

Operational helper for pushing to a Space and smoke-running training remotely.

Capabilities:
- optionally auto-commit and push repo changes
- wait for `/health`
- start `/train`
- poll `/train/status`
- follow an already-running job if requested

Key modes/flags:
- `--skip-push`
- `--skip-health-check`
- `--trigger-only`
- `--status-only`
- `--follow-running`

This is a deployment/ops script, not part of core runtime.

### `scripts/test_env.py`

Best environment behavior reference in the repo.

It verifies:
- public examples are exposed
- hidden tests are not exposed
- correct first-attempt solution gets full reward
- repair trajectory produces discounted reward
- correct-but-less-efficient solution remains open for optimization
- syntax/runtime/timeout/safety states behave as expected

If you need to understand intended environment semantics, start here after `adapt_env.py`.

### `scripts/test_space_api.py`

FastAPI smoke tests with patched training manager behavior.

Verifies:
- root/model/train status endpoints
- reset/step/state flow
- expected 409 when no trained model is available for policy run
- generate-code path
- training start and conflict handling

### `scripts/test_trace_logging.py`

Smoke tests trace artifact creation.

Checks:
- manifest creation
- event JSONL writing
- checkpoint writing
- summary finalization

### `scripts/test_training_config.py`

Tests:
- preset construction
- precision policy resolution under fake torch/cuda conditions
- reward behavior for fully correct but sub-target efficiency case

### `scripts/test_verifier.py`

Manual verification sanity suite over:
- correct
- wrong
- less optimized
- invalid output
- timeout
- runtime error
- safety violation

Also confirms the complexity/efficiency signal differentiates optimized from less-optimized code.

## End-To-End Flows

### Flow A: API Environment Interaction

1. Client calls `POST /reset`
2. `server.app.reset(...)` creates a new `AdaptEnvironment`
3. `AdaptEnvironment.reset(...)` loads/generates a problem and returns public observation
4. Client submits code via `POST /step`
5. `server.app.step(...)` validates `AdaptAction`
6. `AdaptEnvironment.step(...)` calls `verify(...)`
7. `verify(...)` validates AST safety, runs sandboxed code on tests, computes metrics
8. Environment converts metrics into feedback + reward + updated observation
9. `GET /state` returns broader session state if needed

### Flow B: Local Training

1. CLI or server builds `TrainingConfig`
2. `run_training(...)` loads base model and applies LoRA
3. `GeneratorRolloutDataset` emits prompts for current curriculum difficulty
4. GRPO generates completions
5. `build_reward_func(...)` routes completions through `AdaptEnvironment`
6. Environment returns reward
7. `CurriculumManager` updates difficulty
8. `GeneratorController` updates family productivity
9. `TrainingLogger` writes CSV/events/checkpoints
10. Model and tokenizer are saved
11. Optional trained evaluation runs

### Flow C: Server-Side Training On Space

1. `POST /train`
2. `SpaceTrainingManager.start_training(...)`
3. Background thread calls `_run_training_job(...)`
4. `run_training(...)` emits progress updates
5. Status persisted to `training_status.json`
6. Artifacts uploaded to HF repo
7. Local logs optionally deleted
8. Latest model reloaded from Hub
9. `/train/status` and `/model/status` reflect final state

### Flow D: Model-Powered Generation In Demo Or API

1. A problem payload is turned into a solver prompt with `build_solver_prompt(...)`
2. `SpaceModelRegistry.generate_code(...)` picks trained or base model
3. `generate_completion(...)` runs local generation
4. Returned completion is cleaned with `extract_code(...)`
5. In Gradio demo, generated code is immediately evaluated in the same environment loop

## Important Invariants And Hidden Couplings

### Hidden tests must never leak

Critical files:
- `env/adapt_env.py`
- `verifier/verifier.py`
- `scripts/test_env.py`

If you change observation formatting, server payloads, or debug output, verify that hidden tests are still suppressed.

### Prompt text is reused across training, evaluation, and generation

Critical file:
- `training/train_grpo.py`

If you change `SYSTEM_PROMPT`, `build_solver_prompt(...)`, or `extract_code(...)`, you affect:
- GRPO training
- baseline/trained evaluation
- `server.runtime` generation
- `env.app` model attempt path

### Reward semantics are distributed

Files:
- `verifier/metrics.py`
- `env/adapt_env.py`
- `env/generator.py`

There are three conceptually different signals:
- verifier correctness reward
- episode reward
- generator reward signal

Do not change one assuming it is the only reward in the system.

### The generator is both data source and answer key

`env/generator.py` does all of these:
- defines task families
- generates inputs
- defines reference solvers
- defines expected outputs

If a reference solver is wrong, the entire environment silently becomes wrong.

### Correctness is based on hidden tests, not visible tests

Visible tests are for examples and repair feedback.
Core reward/curriculum decisions are driven by hidden pass rate.

### Efficiency can block episode completion

An all-tests-passing answer can still have `done == False` if efficiency target is not met.
This is intentional and affects reward, feedback, and repair behavior.

### Space runtime is designed to degrade gracefully

`server/runtime.py` allows:
- base model use when trained model is unavailable
- trained-to-base fallback on generation error

This is useful operationally, but it means "generation works" does not always imply "trained model is active".

## Change Impact Guide

### If you want to add a new problem family

Primary file:
- `env/generator.py`

You will usually need to add:
- template entry in `_build_templates()`
- case builder(s)
- reference solver
- any parser/helper utilities

Then verify:
- `validate_problem(...)` still passes
- `load_problem_bank()` includes the new family
- Gradio dropdown in `env/app.py` optionally includes it
- README family list stays aligned

### If you want to change reward behavior

Primary files:
- `verifier/metrics.py`
- `env/adapt_env.py`
- maybe `env/generator.py` if generator curriculum shaping should change

Then re-check:
- `scripts/test_env.py`
- `scripts/test_training_config.py`
- training summaries and trace logging assumptions

### If you want to change prompt style or model behavior

Primary file:
- `training/train_grpo.py`

Also impacts:
- `server/runtime.py`
- `inference.py`
- `env/app.py`

### If you want to harden execution/security

Primary files:
- `verifier/sandbox.py`
- `env/executor.py`

Be careful to preserve:
- valid competitive-programming style solutions
- output capture behavior
- timeout and isolation semantics

### If you want to modify server payloads or API

Primary files:
- `models.py`
- `server/app.py`
- `client.py`
- possibly `scripts/test_space_api.py`

## Recommended Reading Order For A New LLM

1. `README.md`
2. `models.py`
3. `env/adapt_env.py`
4. `verifier/verifier.py`
5. `verifier/metrics.py`
6. `env/generator.py`
7. `training/train_grpo.py`
8. `server/app.py`
9. `server/runtime.py`
10. `scripts/test_env.py`
11. `scripts/test_space_api.py`

Why this order:
- it starts with conceptual context
- then schemas
- then the core environment loop
- then generation and reward internals
- then training and serving
- then tests as executable specification

## File-By-File Responsibility Index

### Root

- `README.md`: project rationale, architecture, usage docs
- `pyproject.toml`: package/dependency metadata
- `requirements.txt`: lighter runtime dependencies
- `openenv.yaml`: OpenEnv deployment metadata
- `Dockerfile`: container entrypoint
- `app.py`: re-export server app
- `client.py`: Python HTTP client wrapper
- `models.py`: shared action/observation/state schemas
- `inference.py`: remote API-based benchmark runner
- `test.py`: combined smoke test launcher

### `env/`

- `env/__init__.py`: re-exports generator utilities
- `env/adapt_env.py`: main environment state machine
- `env/app.py`: Gradio demo app
- `env/executor.py`: subprocess execution sandbox
- `env/generator.py`: problem registry, generators, solvers, helpers
- `env/test_cases.py`: problem bank and split helpers

### `verifier/`

- `verifier/__init__.py`: re-exports `verify`
- `verifier/complexity.py`: static complexity heuristics
- `verifier/metrics.py`: reward and scoring math
- `verifier/sandbox.py`: AST safety checks and runtime bridge
- `verifier/verifier.py`: verification orchestration

### `training/`

- `training/__init__.py`: package marker
- `training/plot_results.py`: offline plotting utilities
- `training/trace_logging.py`: structured logs/checkpoints/summary artifacts
- `training/train_grpo.py`: training, evaluation, prompts, config, curriculum

### `server/`

- `server/__init__.py`: re-export app/main
- `server/app.py`: FastAPI routes and session handling
- `server/requirements.txt`: server/runtime dependency subset
- `server/runtime.py`: model registry and training job manager

### `scripts/`

- `scripts/deploy_and_smoke_train.py`: deploy-and-train automation
- `scripts/test_env.py`: environment smoke spec
- `scripts/test_space_api.py`: API smoke spec
- `scripts/test_trace_logging.py`: trace logger smoke spec
- `scripts/test_training_config.py`: config/precision/reward smoke spec
- `scripts/test_verifier.py`: verifier behavior sanity script

## Known Gaps Or Non-Core Artifacts

- No `main-v5/` directory exists in this workspace.
- `rewards/` is empty in this snapshot.
- `outputs/` is empty in this snapshot, so no completed run artifacts were available to inspect.
- The notebook artifacts were not treated as the source of truth; the Python modules are the canonical implementation.

## Bottom Line

If another LLM needs to work effectively in this repo, it should think of the codebase as five tightly linked layers:

1. `env/generator.py` defines what the tasks are.
2. `verifier/*` defines how code is checked and scored.
3. `env/adapt_env.py` turns scoring into an interactive repair environment.
4. `training/train_grpo.py` plugs that environment into GRPO and curriculum learning.
5. `server/*` makes the environment, model runtime, and training lifecycle available over HTTP and in a Hugging Face Space.

That is the true structural spine of the repository.
