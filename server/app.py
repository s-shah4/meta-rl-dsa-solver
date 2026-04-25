from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core>=0.2.3 is required. Install with: pip install -e ."
    ) from exc

from env.adapt_env import AdaptEnvironment
from models import AdaptAction, AdaptObservation


app = create_app(
    AdaptEnvironment,
    AdaptAction,
    AdaptObservation,
    env_name="adapt_dsa_tutor",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
