from __future__ import annotations

from dotenv import load_dotenv

_loaded = False


def load_env() -> None:
    """Загружает переменные окружения из .env."""
    global _loaded
    if _loaded:
        return
    load_dotenv()
    _loaded = True


__all__ = ["load_env"]