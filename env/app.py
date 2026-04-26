from __future__ import annotations

from html import escape
from uuid import uuid4

import gradio as gr

from env.adapt_env import AdaptEnvironment
from models import AdaptAction
from server.runtime import SpaceTrainingManager

TRAINING_MANAGER = SpaceTrainingManager()
SESSIONS: dict[str, AdaptEnvironment] = {}

PLAYGROUND_DEFAULT_PROBLEM = (
    "Given an array of integers, return the length of the longest contiguous subarray "
    "whose sum is divisible by k."
)
PLAYGROUND_DEFAULT_INPUT = (
    "The first line contains two integers n and k. The second line contains n space-separated integers."
)
PLAYGROUND_DEFAULT_CONSTRAINTS = "1 <= n <= 2 * 10^5, 1 <= k <= 10^9, array values fit in 32-bit signed integers."


def _get_env(session_id: str | None) -> AdaptEnvironment:
    if not session_id or session_id not in SESSIONS:
        session_id = str(uuid4())
        SESSIONS[session_id] = AdaptEnvironment(session_id=session_id)
    return SESSIONS[session_id]


def _problem_markdown(observation: dict) -> str:
    return (
        f"### {observation['problem_type']} ({observation['difficulty']})\n\n"
        f"{observation['problem']}\n\n"
        f"**Input Format**\n{observation['input_format']}\n\n"
        f"**Constraints**\n{observation['constraints']}"
    )


def sample_problem(problem_id: str, difficulty: str) -> tuple[str, str, str, str, dict]:
    env = _get_env(None)
    observation = env.reset(
        problem_id=problem_id or None,
        difficulty=difficulty or None,
        session_id=env.session_id,
    )
    payload = observation.model_dump()
    return (
        env.session_id,
        _problem_markdown(payload),
        payload["feedback"],
        "",
        payload,
    )


def evaluate_submission(session_id: str, code: str) -> tuple[str, str, str, dict]:
    env = _get_env(session_id)
    observation = env.step(AdaptAction(session_id=env.session_id, code=code))
    payload = observation.model_dump()
    status = (
        f"Reward: {payload['reward']:.2f} | Hidden pass rate: {payload['hidden_pass_rate']:.2f} | "
        f"Visible pass rate: {payload['visible_pass_rate']:.2f} | Status: {payload['execution_status']}"
    )
    return payload["feedback"], status, code, payload


def model_attempt(session_id: str) -> tuple[str, str, str, dict]:
    env = _get_env(session_id)
    if not env.problem:
        observation = env.reset(session_id=env.session_id)
    else:
        observation = env._build_observation(
            reward=float(env.state.last_reward or 0.0),
            done=bool(env.episode_done),
            feedback=env.state.last_feedback or "No attempt yet.",
            pass_rate=float(env.state.last_pass_rate or 0.0),
            visible_pass_rate=float(env.state.recent_metrics.get("visible_pass_rate", 0.0)),
            hidden_pass_rate=float(env.state.last_pass_rate or 0.0),
            syntax_valid=env.state.last_execution_status != "syntax_error",
            execution_status=env.state.last_execution_status or "ready",
            timeout_count=int(env.state.recent_metrics.get("timeout_count", 0)),
            runtime_error_count=int(env.state.recent_metrics.get("runtime_error_count", 0)),
            invalid_output_count=int(env.state.recent_metrics.get("invalid_output_count", 0)),
            wrong_answer_count=int(env.state.recent_metrics.get("wrong_answer_count", 0)),
            format_compliance=float(env.state.recent_metrics.get("format_compliance", 0.0)),
            reward_components=dict(env.state.recent_metrics.get("reward_components", {})),
            generator_reward_signal=float(env.state.generator_reward_signal or 0.0),
        )

    try:
        generation = TRAINING_MANAGER.generate_code(
            problem=observation.problem,
            input_format=observation.input_format,
            constraints=observation.constraints,
            feedback=observation.feedback,
            problem_id=observation.problem_id,
            problem_type=observation.problem_type,
            difficulty=observation.difficulty,
            attempt_number=observation.attempt_number + 1,
            max_steps=observation.max_steps,
        )
    except Exception as exc:
        return str(exc), "Model generation unavailable", "", {"error": str(exc)}

    return evaluate_submission(session_id, generation["code"])


def _format_accuracy(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    return f"{value * 100:.1f}%"


def _format_progress(completed_steps: int, total_steps: int, progress_ratio: float) -> str:
    if total_steps > 0:
        return f"{completed_steps}/{total_steps} steps ({progress_ratio * 100:.1f}%)"
    return "No active training run"


def _render_metrics_card(metrics: dict) -> str:
    overall_accuracy = metrics.get("overall_accuracy")
    baseline_accuracy = metrics.get("baseline_accuracy")
    live_pass_rate = metrics.get("live_pass_rate")
    metric_source = metrics.get("metric_source", "unavailable")
    training_status = escape(str(metrics.get("training_status", "unknown")).replace("_", " ").title())
    phase = escape(str(metrics.get("phase", "idle")).replace("_", " ").title())
    progress = escape(
        _format_progress(
            int(metrics.get("completed_steps", 0) or 0),
            int(metrics.get("total_steps", 0) or 0),
            float(metrics.get("progress_ratio", 0.0) or 0.0),
        )
    )
    source_label = {
        "trained_eval": "Final evaluation metric",
        "rolling_pass_rate": "Live rolling pass rate",
        "unavailable": "Metric unavailable",
    }.get(metric_source, "Metric unavailable")
    source_label = escape(source_label)
    live_pass_rate_text = escape(_format_accuracy(live_pass_rate))
    overall_accuracy_text = escape(_format_accuracy(overall_accuracy))
    baseline_accuracy_text = escape(_format_accuracy(baseline_accuracy))

    return f"""
    <div class="panel metrics-card">
      <div class="metrics-header">
        <div>
          <div class="metrics-eyebrow">Live Demo Dashboard</div>
          <h3>Model Accuracy Tracker</h3>
        </div>
        <div class="metrics-chip">{source_label}</div>
      </div>
      <div class="metrics-grid">
        <div class="metric-tile">
          <span>Overall Accuracy</span>
          <strong>{overall_accuracy_text}</strong>
        </div>
        <div class="metric-tile">
          <span>Base Accuracy</span>
          <strong>{baseline_accuracy_text}</strong>
        </div>
        <div class="metric-tile">
          <span>Live Pass Rate</span>
          <strong>{live_pass_rate_text}</strong>
        </div>
        <div class="metric-tile">
          <span>Training Phase</span>
          <strong>{phase}</strong>
        </div>
      </div>
      <div class="metrics-footer">
        <span>Status: <strong>{training_status}</strong></span>
        <span>Progress: <strong>{progress}</strong></span>
      </div>
    </div>
    """


def playground_metrics_view() -> str:
    payload = TRAINING_MANAGER.status_payload()
    metrics = payload.get("demo_metrics", {})
    return _render_metrics_card(metrics if isinstance(metrics, dict) else {})


def _model_badge(result: dict | None, label: str, error: str | None = None) -> str:
    if error:
        return f"### {label}\nStatus: {error}"
    if not result:
        return f"### {label}\nStatus: No generation yet."
    model = result.get("model", {}) if isinstance(result.get("model"), dict) else {}
    effective_model = str(result.get("effective_model", model.get("active_model_kind", "unavailable"))).replace("_", " ")
    source = model.get("source_repo_id") or model.get("base_model_name") or model.get("local_path") or "unknown source"
    details = [f"Status: {effective_model.title()}"]
    if model.get("fallback_reason"):
        details.append("Fallback: trained model unavailable, using base model")
    details.append(f"Source: {source}")
    if model.get("revision"):
        details.append(f"Revision: {model['revision']}")
    return f"### {label}\n" + "\n".join(details)


def compare_models(problem: str, input_format: str, constraints: str) -> tuple[str, str, str, str]:
    if not problem.strip():
        message = "Please enter a problem statement before generating solutions."
        return "", "", _model_badge(None, "Base Model", message), _model_badge(None, "ADAPT-Trained Model", message)
    if not input_format.strip():
        message = "Please provide the input format so the model prompt is well-formed."
        return "", "", _model_badge(None, "Base Model", message), _model_badge(None, "ADAPT-Trained Model", message)
    if not constraints.strip():
        message = "Please provide constraints so the model can target the expected solution shape."
        return "", "", _model_badge(None, "Base Model", message), _model_badge(None, "ADAPT-Trained Model", message)

    base_result: dict | None = None
    trained_result: dict | None = None
    base_error: str | None = None
    trained_error: str | None = None

    try:
        base_result = TRAINING_MANAGER.generate_code(
            problem=problem,
            input_format=input_format,
            constraints=constraints,
            target_model="base",
        )
    except Exception as exc:
        base_error = str(exc)

    try:
        trained_result = TRAINING_MANAGER.generate_code(
            problem=problem,
            input_format=input_format,
            constraints=constraints,
            target_model="current",
        )
    except Exception as exc:
        trained_error = str(exc)

    return (
        base_result.get("code", "") if base_result else "",
        trained_result.get("code", "") if trained_result else "",
        _model_badge(base_result, "Base Model", base_error),
        _model_badge(trained_result, "ADAPT-Trained Model", trained_error),
    )


with gr.Blocks(
    title="ADAPT DSA Tutor Demo",
    css="""
    .panel {border: 1px solid #d7d3c9; border-radius: 18px; background: #fffaf2;}
    .hero {background: linear-gradient(135deg, #f7eedb, #f3f8ef); border-radius: 22px; padding: 18px;}
    .metrics-card {padding: 18px;}
    .metrics-header {display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 14px;}
    .metrics-header h3 {margin: 4px 0 0; font-size: 1.2rem;}
    .metrics-eyebrow {font-size: 0.8rem; letter-spacing: 0.08em; text-transform: uppercase; color: #816b45;}
    .metrics-chip {background: #efe3c6; color: #6f5529; border-radius: 999px; padding: 6px 10px; font-size: 0.85rem;}
    .metrics-grid {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px;}
    .metric-tile {background: #fff; border: 1px solid #eadfc9; border-radius: 14px; padding: 12px;}
    .metric-tile span {display: block; font-size: 0.82rem; color: #7a6541; margin-bottom: 6px;}
    .metric-tile strong {font-size: 1.1rem; color: #2f2412;}
    .metrics-footer {display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; color: #5c4c30;}
    @media (max-width: 900px) {
      .metrics-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}
    }
    @media (max-width: 640px) {
      .metrics-grid {grid-template-columns: minmax(0, 1fr);}
    }
    """,
) as demo:
    with gr.Tab("Verifier Sandbox"):
        session_id = gr.Textbox(label="Session ID", interactive=False)
        state_payload = gr.JSON(label="Observation Payload")

        gr.Markdown(
            """
            # ADAPT DSA Tutor
            Sample a problem, inspect the verifier feedback, and compare your repair attempt with the currently loaded model path.
            """,
            elem_classes=["hero"],
        )

        with gr.Row():
            problem_id = gr.Dropdown(
                choices=[
                    "",
                    "sum_even_numbers",
                    "range_span",
                    "count_vowels",
                    "max_consecutive_ones",
                    "fizzbuzz_variant",
                    "running_total",
                    "count_local_peaks",
                    "longest_non_decreasing_run",
                    "two_sum_count",
                    "max_subarray_sum",
                    "group_anagrams_count",
                    "balanced_brackets",
                    "matrix_diagonal_sum",
                    "smallest_most_frequent",
                    "reverse_words",
                    "longest_common_subsequence",
                    "word_ladder_steps",
                    "merge_intervals",
                    "min_coins",
                    "rotate_matrix_90",
                ],
                value="",
                label="Problem Family",
                info="Leave blank to sample automatically.",
            )
            difficulty = gr.Radio(choices=["easy", "medium", "hard"], value="easy", label="Difficulty")
            sample_btn = gr.Button("Sample Problem", variant="primary")

        problem_view = gr.Markdown(elem_classes=["panel"])
        with gr.Row():
            code = gr.Textbox(
                label="Python Submission",
                lines=18,
                max_lines=24,
                placeholder="Write code that reads stdin and prints stdout.",
            )
            with gr.Column():
                feedback = gr.Textbox(label="Verifier Feedback", lines=14)
                status = gr.Textbox(label="Scorecard", lines=4)
                with gr.Row():
                    verify_btn = gr.Button("Verify Submission", variant="primary")
                    model_btn = gr.Button("Run Current Model", variant="secondary")

        sample_btn.click(
            fn=sample_problem,
            inputs=[problem_id, difficulty],
            outputs=[session_id, problem_view, feedback, code, state_payload],
        )
        verify_btn.click(
            fn=evaluate_submission,
            inputs=[session_id, code],
            outputs=[feedback, status, code, state_payload],
        )
        model_btn.click(
            fn=model_attempt,
            inputs=[session_id],
            outputs=[feedback, status, code, state_payload],
        )

    with gr.Tab("Model Playground"):
        gr.Markdown(
            """
            # Model Playground
            Paste a problem and compare how the base model stacks up against the current ADAPT-powered solver while training metrics refresh live below.
            """,
            elem_classes=["hero"],
        )
        playground_timer = gr.Timer(value=5)

        problem_text = gr.Textbox(
            label="Problem",
            lines=8,
            value=PLAYGROUND_DEFAULT_PROBLEM,
            placeholder="Paste the full problem statement here.",
        )
        with gr.Row():
            input_format_text = gr.Textbox(
                label="Input Format",
                lines=5,
                value=PLAYGROUND_DEFAULT_INPUT,
                placeholder="Describe how stdin is formatted.",
            )
            constraints_text = gr.Textbox(
                label="Constraints",
                lines=5,
                value=PLAYGROUND_DEFAULT_CONSTRAINTS,
                placeholder="Add the important bounds and edge constraints.",
            )
        with gr.Row():
            generate_btn = gr.Button("Generate Solutions", variant="primary")
            clear_btn = gr.Button("Clear", variant="secondary")

        with gr.Row():
            with gr.Column():
                base_code = gr.Code(label="Base Model Output", language="python", lines=20)
                base_status = gr.Markdown(_model_badge(None, "Base Model"))
            with gr.Column():
                trained_code = gr.Code(label="ADAPT-Trained Model Output", language="python", lines=20)
                trained_status = gr.Markdown(_model_badge(None, "ADAPT-Trained Model"))

        metrics_dashboard = gr.HTML(playground_metrics_view())

        generate_btn.click(
            fn=compare_models,
            inputs=[problem_text, input_format_text, constraints_text],
            outputs=[base_code, trained_code, base_status, trained_status],
        )
        clear_btn.click(
            fn=lambda: (
                "",
                "",
                "",
                "",
                "",
                _model_badge(None, "Base Model"),
                _model_badge(None, "ADAPT-Trained Model"),
                playground_metrics_view(),
            ),
            outputs=[
                problem_text,
                input_format_text,
                constraints_text,
                base_code,
                trained_code,
                base_status,
                trained_status,
                metrics_dashboard,
            ],
        )
        playground_timer.tick(fn=playground_metrics_view, outputs=[metrics_dashboard])
        demo.load(fn=playground_metrics_view, outputs=[metrics_dashboard])


if __name__ == "__main__":
    demo.launch()
