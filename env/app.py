from __future__ import annotations

from uuid import uuid4

import gradio as gr

from env.adapt_env import AdaptEnvironment
from models import AdaptAction
from server.runtime import SpaceTrainingManager

TRAINING_MANAGER = SpaceTrainingManager()
SESSIONS: dict[str, AdaptEnvironment] = {}


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


with gr.Blocks(
    title="ADAPT DSA Tutor Demo",
    css="""
    .panel {border: 1px solid #d7d3c9; border-radius: 18px; background: #fffaf2;}
    .hero {background: linear-gradient(135deg, #f7eedb, #f3f8ef); border-radius: 22px; padding: 18px;}
    """,
) as demo:
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
        code = gr.Textbox(label="Python Submission", lines=18, max_lines=24, placeholder="Write code that reads stdin and prints stdout.")
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


if __name__ == "__main__":
    demo.launch()
