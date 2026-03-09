from __future__ import annotations

from backend.app.config import load_settings
from backend.app.main import app


def main() -> None:
    settings = load_settings()
    debug = settings.app_env.lower() in {"dev", "development"}
    app.run(host=settings.host, port=settings.port, debug=debug)


if __name__ == "__main__":
    main()

