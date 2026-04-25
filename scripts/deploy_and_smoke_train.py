#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_REMOTE = "space"
DEFAULT_REMOTE_BRANCH = "main"
DEFAULT_POLL_INTERVAL_SECONDS = 10
DEFAULT_TIMEOUT_SECONDS = 60 * 30
DEFAULT_REQUIRED_HEALTHY_CHECKS = 3
DEFAULT_MIN_DEPLOY_WAIT_SECONDS = 30


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
    raise RuntimeError(f"{action} failed: {message}")


def has_uncommitted_changes(repo_root: Path) -> bool:
    result = run_command(["git", "status", "--porcelain"], repo_root)
    ensure_success(result, "git status")
    return bool(result.stdout.strip())


def commit_changes(repo_root: Path, message: str) -> str | None:
    if not has_uncommitted_changes(repo_root):
        print("No uncommitted changes found. Reusing the current HEAD commit.", flush=True)
        return None

    add_result = run_command(["git", "add", "-A"], repo_root)
    ensure_success(add_result, "git add")

    commit_result = run_command(["git", "commit", "-m", message], repo_root)
    ensure_success(commit_result, "git commit")
    print(commit_result.stdout.strip(), flush=True)

    rev_result = run_command(["git", "rev-parse", "HEAD"], repo_root)
    ensure_success(rev_result, "git rev-parse")
    return rev_result.stdout.strip()


def push_to_space(repo_root: Path, remote: str, remote_branch: str) -> None:
    push_result = run_command(["git", "push", remote, f"HEAD:{remote_branch}"], repo_root)
    ensure_success(push_result, f"git push {remote} HEAD:{remote_branch}")
    print(push_result.stdout.strip() or push_result.stderr.strip(), flush=True)


def http_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc


def try_get_space_health(health_url: str) -> dict[str, Any] | None:
    try:
        return http_json("GET", health_url)
    except RuntimeError:
        return None


def wait_for_space_health(
    base_url: str,
    timeout_seconds: int,
    poll_interval_seconds: int,
    required_healthy_checks: int,
    min_deploy_wait_seconds: int,
) -> None:
    health_url = f"{base_url.rstrip('/')}/health"
    deadline = time.time() + timeout_seconds
    push_started_at = time.time()
    previous_payload = try_get_space_health(health_url)
    deployment_transition_seen = previous_payload is None
    consecutive_healthy_checks = 0

    while time.time() < deadline:
        try:
            payload = http_json("GET", health_url)
        except RuntimeError as exc:
            deployment_transition_seen = True
            consecutive_healthy_checks = 0
            print(f"Waiting for Space deployment: {exc}", flush=True)
            time.sleep(poll_interval_seconds)
            continue

        if previous_payload != payload:
            deployment_transition_seen = True

        if not deployment_transition_seen:
            print(
                "Space is still serving the pre-push health payload; waiting for the new deployment to take over.",
                flush=True,
            )
            previous_payload = payload
            time.sleep(poll_interval_seconds)
            continue

        waited_long_enough = (time.time() - push_started_at) >= min_deploy_wait_seconds
        if not waited_long_enough:
            remaining = max(0, min_deploy_wait_seconds - int(time.time() - push_started_at))
            print(
                f"Deployment transition detected. Waiting {remaining}s more for the Space to stabilize.",
                flush=True,
            )
            previous_payload = payload
            time.sleep(poll_interval_seconds)
            continue

        if payload.get("status") == "healthy":
            consecutive_healthy_checks += 1
            print(
                f"Space healthy check {consecutive_healthy_checks}/{required_healthy_checks}: {payload}",
                flush=True,
            )
            if consecutive_healthy_checks >= required_healthy_checks:
                print("Space deployment looks stable. Starting smoke training.", flush=True)
                return
        else:
            consecutive_healthy_checks = 0
            print(f"Space health not ready yet: {payload}", flush=True)

        previous_payload = payload
        time.sleep(poll_interval_seconds)

    raise TimeoutError(f"Space did not finish deploying within {timeout_seconds} seconds.")


def start_smoke_training(base_url: str) -> dict[str, Any]:
    train_url = f"{base_url.rstrip('/')}/train"
    payload = http_json("POST", train_url, {"preset": "smoke"})
    print(f"Started smoke training: {json.dumps(payload, indent=2)}", flush=True)
    return payload


def poll_training_status(base_url: str, poll_interval_seconds: int, timeout_seconds: int) -> dict[str, Any]:
    status_url = f"{base_url.rstrip('/')}/train/status"
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        payload = http_json("GET", status_url)
        status = payload.get("status")
        phase = payload.get("phase")
        completed_steps = payload.get("completed_steps")
        total_steps = payload.get("total_steps")
        print(
            f"Training status: status={status} phase={phase} completed_steps={completed_steps}/{total_steps}",
            flush=True,
        )

        if status == "failed":
            print("Training failed. Full error payload:", flush=True)
            print(json.dumps(payload, indent=2), flush=True)
            return payload

        if status == "succeeded":
            print("Training succeeded. Final payload:", flush=True)
            print(json.dumps(payload, indent=2), flush=True)
            return payload

        time.sleep(poll_interval_seconds)

    raise TimeoutError(f"Training did not finish within {timeout_seconds} seconds.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Commit, push, trigger smoke training, and poll HF Space status.")
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL of the deployed Hugging Face Space, for example https://<space>.hf.space",
    )
    parser.add_argument(
        "--commit-message",
        default="Automated smoke training update",
        help="Commit message to use when local changes are present.",
    )
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="Git remote for the Hugging Face Space.")
    parser.add_argument(
        "--remote-branch",
        default=DEFAULT_REMOTE_BRANCH,
        help="Remote branch that deploys the Hugging Face Space.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help="How often to poll /train/status.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Overall timeout for Space health and smoke training completion.",
    )
    parser.add_argument(
        "--required-healthy-checks",
        type=int,
        default=DEFAULT_REQUIRED_HEALTHY_CHECKS,
        help="How many consecutive healthy checks to require before starting training.",
    )
    parser.add_argument(
        "--min-deploy-wait-seconds",
        type=int,
        default=DEFAULT_MIN_DEPLOY_WAIT_SECONDS,
        help="Minimum number of seconds to wait after the push before treating the Space as fully deployed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    try:
        commit_hash = commit_changes(repo_root, args.commit_message)
        if commit_hash:
            print(f"Committed changes at {commit_hash}", flush=True)

        push_to_space(repo_root, args.remote, args.remote_branch)
        wait_for_space_health(
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            required_healthy_checks=args.required_healthy_checks,
            min_deploy_wait_seconds=args.min_deploy_wait_seconds,
        )
        start_smoke_training(args.base_url)
        final_status = poll_training_status(
            base_url=args.base_url,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        print(f"Deployment/training loop failed: {exc}", file=sys.stderr, flush=True)
        return 1

    return 0 if final_status.get("status") == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
