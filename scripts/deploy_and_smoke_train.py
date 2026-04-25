#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_REMOTE = "space"
DEFAULT_REMOTE_BRANCH = "main"
DEFAULT_POLL_INTERVAL_SECONDS = 10
DEFAULT_DEPLOY_TIMEOUT_SECONDS = 60 * 20
DEFAULT_TRAIN_TIMEOUT_SECONDS = 60 * 30
DEFAULT_REQUIRED_HEALTHY_CHECKS = 3
DEFAULT_MIN_DEPLOY_WAIT_SECONDS = 30
DEFAULT_COMMIT_MESSAGE = "Smoke-train deployment update"


class ScriptError(RuntimeError):
    pass


def print_info(message: str) -> None:
    print(f"[info] {message}", flush=True)


def print_warn(message: str) -> None:
    print(f"[warn] {message}", flush=True)


def print_error(message: str) -> None:
    print(f"[error] {message}", file=sys.stderr, flush=True)


def pretty_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total_seconds = max(float(seconds), 0.0)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = round(total_seconds % 60, 1)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def http_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url=url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            if not raw.strip():
                return {}
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ScriptError(f"HTTP {exc.code} for {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ScriptError(f"Request to {url} failed: {exc}") from exc


@dataclass(frozen=True)
class GitOptions:
    repo_root: Path
    remote: str
    remote_branch: str
    commit_message: str
    skip_commit: bool
    skip_push: bool


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def ensure_success(result: subprocess.CompletedProcess[str], action: str) -> None:
    if result.returncode == 0:
        return
    message = result.stderr.strip() or result.stdout.strip() or f"{action} failed with exit code {result.returncode}"
    raise ScriptError(f"{action} failed: {message}")


def has_uncommitted_changes(repo_root: Path) -> bool:
    result = run_command(["git", "status", "--porcelain"], repo_root)
    ensure_success(result, "git status")
    return bool(result.stdout.strip())


def current_head_sha(repo_root: Path) -> str:
    result = run_command(["git", "rev-parse", "HEAD"], repo_root)
    ensure_success(result, "git rev-parse HEAD")
    return result.stdout.strip()


def remote_branch_sha(repo_root: Path, remote: str, remote_branch: str) -> str | None:
    result = run_command(["git", "ls-remote", remote, f"refs/heads/{remote_branch}"], repo_root)
    ensure_success(result, f"git ls-remote {remote} refs/heads/{remote_branch}")
    line = result.stdout.strip()
    if not line:
        return None
    return line.split()[0]


def commit_changes_if_needed(options: GitOptions) -> str:
    repo_root = options.repo_root
    head_before = current_head_sha(repo_root)
    if not has_uncommitted_changes(repo_root):
        print_info(f"Working tree is clean at {head_before}.")
        return head_before

    if options.skip_commit:
        raise ScriptError(
            "Working tree has uncommitted changes, but --skip-commit was set. "
            "Commit the changes manually or remove --skip-commit."
        )

    print_info("Staging and committing local changes before deploy.")
    ensure_success(run_command(["git", "add", "-A"], repo_root), "git add -A")
    commit_result = run_command(["git", "commit", "-m", options.commit_message], repo_root)
    ensure_success(commit_result, "git commit")
    if commit_result.stdout.strip():
        print(commit_result.stdout.strip(), flush=True)
    head_after = current_head_sha(repo_root)
    print_info(f"Created commit {head_after}.")
    return head_after


def deployment_needed(options: GitOptions, local_head_sha: str) -> bool:
    remote_sha = remote_branch_sha(options.repo_root, options.remote, options.remote_branch)
    if remote_sha is None:
        print_info(f"Remote branch {options.remote}/{options.remote_branch} does not exist yet.")
        return True
    if remote_sha == local_head_sha:
        print_info(f"Remote {options.remote}/{options.remote_branch} already points to {local_head_sha}.")
        return False
    print_info(
        f"Remote {options.remote}/{options.remote_branch} is at {remote_sha}; "
        f"local HEAD is {local_head_sha}."
    )
    return True


def push_current_head(options: GitOptions) -> None:
    print_info(f"Pushing HEAD to {options.remote}/{options.remote_branch}.")
    result = run_command(["git", "push", options.remote, f"HEAD:{options.remote_branch}"], options.repo_root)
    ensure_success(result, f"git push {options.remote} HEAD:{options.remote_branch}")
    summary = result.stdout.strip() or result.stderr.strip()
    if summary:
        print(summary, flush=True)


def fetch_health(base_url: str) -> dict[str, Any]:
    return http_json("GET", api_url(base_url, "/health"))


def wait_for_health(
    base_url: str,
    *,
    timeout_seconds: int,
    poll_interval_seconds: int,
    required_healthy_checks: int,
    min_deploy_wait_seconds: int,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    push_started_at = time.time()
    prior_payload: dict[str, Any] | None = None
    transition_observed = False
    healthy_streak = 0

    while time.time() < deadline:
        try:
            payload = fetch_health(base_url)
        except ScriptError as exc:
            healthy_streak = 0
            transition_observed = True
            print_info(f"Waiting for service health: {exc}")
            time.sleep(poll_interval_seconds)
            continue

        if prior_payload is not None and payload != prior_payload:
            transition_observed = True

        elapsed = time.time() - push_started_at
        if elapsed < min_deploy_wait_seconds:
            remaining = max(0, int(min_deploy_wait_seconds - elapsed))
            print_info(f"Health endpoint reachable. Waiting {remaining}s for deployment stabilization.")
            prior_payload = payload
            time.sleep(poll_interval_seconds)
            continue

        if payload.get("status") == "healthy":
            healthy_streak += 1
            print_info(
                f"Health check {healthy_streak}/{required_healthy_checks}: "
                f"training={payload.get('training')} model_loaded={payload.get('model_loaded')}"
            )
            if healthy_streak >= required_healthy_checks:
                if not transition_observed and prior_payload is not None:
                    print_warn("No payload transition was observed after deploy; continuing because health stabilized.")
                return payload
        else:
            healthy_streak = 0
            print_info(f"Health payload not ready yet: {pretty_json(payload)}")

        prior_payload = payload
        time.sleep(poll_interval_seconds)

    raise ScriptError(f"Service did not report healthy within {timeout_seconds} seconds.")


def fetch_training_status(base_url: str) -> dict[str, Any]:
    return http_json("GET", api_url(base_url, "/train/status"))


def summarize_training_status(payload: dict[str, Any]) -> str:
    status = payload.get("status")
    phase = payload.get("phase")
    completed = payload.get("completed_steps")
    total = payload.get("total_steps")
    difficulty = payload.get("current_difficulty")
    problem_family = payload.get("last_problem_family")
    reward = payload.get("last_reward")
    elapsed_minutes = payload.get("elapsed_minutes")
    pieces = [
        f"status={status}",
        f"phase={phase}",
        f"steps={completed}/{total}",
    ]
    if difficulty:
        pieces.append(f"difficulty={difficulty}")
    if problem_family:
        pieces.append(f"family={problem_family}")
    if reward is not None:
        pieces.append(f"reward={reward}")
    if elapsed_minutes is not None:
        pieces.append(f"elapsed={elapsed_minutes}m")
    return " ".join(pieces)


def print_runtime_baseline(payload: dict[str, Any]) -> None:
    timing = payload.get("timing_summary") or {}
    wall_clock_seconds = timing.get("wall_clock_seconds")
    if wall_clock_seconds is None:
        wall_clock_seconds = payload.get("elapsed_seconds")
    if wall_clock_seconds is None:
        return

    avg_seconds_per_step = timing.get("avg_seconds_per_step")
    avg_seconds_per_episode = timing.get("avg_seconds_per_episode")
    steps_per_hour = timing.get("steps_per_hour")
    episodes_per_hour = timing.get("episodes_per_hour")

    print_info(
        "Smoke runtime baseline: "
        f"wall_clock={format_duration(float(wall_clock_seconds))}, "
        f"avg_step={avg_seconds_per_step}s, "
        f"avg_episode={avg_seconds_per_episode}s"
    )

    if steps_per_hour is not None or episodes_per_hour is not None:
        six_hour_steps = round(float(steps_per_hour) * 6) if steps_per_hour is not None else None
        seven_hour_steps = round(float(steps_per_hour) * 7) if steps_per_hour is not None else None
        six_hour_episodes = round(float(episodes_per_hour) * 6) if episodes_per_hour is not None else None
        seven_hour_episodes = round(float(episodes_per_hour) * 7) if episodes_per_hour is not None else None
        print_info(
            "Window estimate at current smoke throughput: "
            f"6h -> steps={six_hour_steps}, episodes={six_hour_episodes}; "
            f"7h -> steps={seven_hour_steps}, episodes={seven_hour_episodes}"
        )


def ensure_no_active_training(base_url: str) -> None:
    payload = fetch_training_status(base_url)
    if payload.get("status") == "running":
        raise ScriptError(
            "The remote training manager already reports an active run. "
            f"Current status: {summarize_training_status(payload)}"
        )


def start_training(base_url: str, train_payload: dict[str, Any]) -> dict[str, Any]:
    print_info(f"Starting training with payload: {pretty_json(train_payload)}")
    payload = http_json("POST", api_url(base_url, "/train"), payload=train_payload)
    print_info(f"Training accepted: {pretty_json(payload)}")
    return payload


def poll_training_status(
    base_url: str,
    *,
    poll_interval_seconds: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_signature: tuple[Any, ...] | None = None

    while time.time() < deadline:
        payload = fetch_training_status(base_url)
        signature = (
            payload.get("status"),
            payload.get("phase"),
            payload.get("completed_steps"),
            payload.get("total_steps"),
            payload.get("last_problem_family"),
            payload.get("last_reward"),
            payload.get("error"),
        )
        if signature != last_signature:
            print_info(summarize_training_status(payload))
            last_signature = signature

        status = payload.get("status")
        if status == "failed":
            print_error("Training failed. Final payload follows.")
            print(pretty_json(payload), flush=True)
            return payload
        if status == "succeeded":
            print_info("Training succeeded. Final payload follows.")
            print(pretty_json(payload), flush=True)
            print_runtime_baseline(payload)
            return payload

        time.sleep(poll_interval_seconds)

    raise ScriptError(f"Training did not finish within {timeout_seconds} seconds.")


def build_train_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {"preset": args.preset}
    if args.train_payload_json:
        extra_payload = json.loads(args.train_payload_json)
        if not isinstance(extra_payload, dict):
            raise ScriptError("--train-payload-json must decode to a JSON object.")
        payload.update(extra_payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy the current repo to a Hugging Face Space and run a smoke training job.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL of the running server, for example https://<space>.hf.space or http://localhost:7860",
    )
    parser.add_argument(
        "--preset",
        default="smoke",
        help="Training preset to send to /train. Defaults to smoke.",
    )
    parser.add_argument(
        "--train-payload-json",
        default=None,
        help="Optional JSON object merged into the /train request body.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help="How often to poll /health and /train/status.",
    )
    parser.add_argument(
        "--deploy-timeout-seconds",
        type=int,
        default=DEFAULT_DEPLOY_TIMEOUT_SECONDS,
        help="Maximum time to wait for the service to become healthy after push.",
    )
    parser.add_argument(
        "--train-timeout-seconds",
        type=int,
        default=DEFAULT_TRAIN_TIMEOUT_SECONDS,
        help="Maximum time to wait for the smoke train run to finish.",
    )
    parser.add_argument(
        "--required-healthy-checks",
        type=int,
        default=DEFAULT_REQUIRED_HEALTHY_CHECKS,
        help="Number of consecutive healthy /health checks required before training starts.",
    )
    parser.add_argument(
        "--min-deploy-wait-seconds",
        type=int,
        default=DEFAULT_MIN_DEPLOY_WAIT_SECONDS,
        help="Minimum time to wait after push before treating health as stable.",
    )
    parser.add_argument(
        "--remote",
        default=DEFAULT_REMOTE,
        help="Git remote to push to when deploying.",
    )
    parser.add_argument(
        "--remote-branch",
        default=DEFAULT_REMOTE_BRANCH,
        help="Remote branch to push HEAD to when deploying.",
    )
    parser.add_argument(
        "--commit-message",
        default=DEFAULT_COMMIT_MESSAGE,
        help="Commit message to use if local changes need to be committed before push.",
    )
    parser.add_argument(
        "--skip-commit",
        action="store_true",
        help="Do not auto-commit local changes before deploy.",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Skip git deploy entirely and just hit the running server.",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip waiting on /health before training.",
    )
    parser.add_argument(
        "--trigger-only",
        action="store_true",
        help="Start the smoke run and exit without polling to completion.",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Do not start a new run; just print /train/status and optionally poll it.",
    )
    parser.add_argument(
        "--follow-running",
        action="store_true",
        help="If /train/status already reports a running job, follow it instead of failing.",
    )
    return parser


def maybe_deploy(args: argparse.Namespace, repo_root: Path) -> None:
    if args.skip_push:
        print_info("Skipping git deploy because --skip-push was set.")
        return

    git_options = GitOptions(
        repo_root=repo_root,
        remote=args.remote,
        remote_branch=args.remote_branch,
        commit_message=args.commit_message,
        skip_commit=args.skip_commit,
        skip_push=args.skip_push,
    )
    local_head_sha = commit_changes_if_needed(git_options)
    if not deployment_needed(git_options, local_head_sha):
        print_info("Skipping push because the remote is already on the current local HEAD.")
        return

    push_current_head(git_options)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    try:
        if args.status_only:
            payload = fetch_training_status(args.base_url)
            print(pretty_json(payload), flush=True)
            if args.follow_running and payload.get("status") == "running":
                final_status = poll_training_status(
                    args.base_url,
                    poll_interval_seconds=args.poll_interval_seconds,
                    timeout_seconds=args.train_timeout_seconds,
                )
                return 0 if final_status.get("status") == "succeeded" else 1
            return 0

        maybe_deploy(args, repo_root)

        if not args.skip_health_check:
            wait_for_health(
                args.base_url,
                timeout_seconds=args.deploy_timeout_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
                required_healthy_checks=args.required_healthy_checks,
                min_deploy_wait_seconds=args.min_deploy_wait_seconds,
            )
        else:
            print_info("Skipping health wait because --skip-health-check was set.")

        current_status = fetch_training_status(args.base_url)
        if current_status.get("status") == "running":
            if args.follow_running:
                print_warn(
                    "A training job is already running; following the existing run instead of starting a new one."
                )
                final_status = poll_training_status(
                    args.base_url,
                    poll_interval_seconds=args.poll_interval_seconds,
                    timeout_seconds=args.train_timeout_seconds,
                )
                return 0 if final_status.get("status") == "succeeded" else 1
            ensure_no_active_training(args.base_url)

        train_payload = build_train_payload(args)
        start_training(args.base_url, train_payload)
        if args.trigger_only:
            print_info("Training was triggered successfully; exiting because --trigger-only was set.")
            return 0

        final_status = poll_training_status(
            args.base_url,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout_seconds=args.train_timeout_seconds,
        )
        return 0 if final_status.get("status") == "succeeded" else 1
    except Exception as exc:
        print_error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
