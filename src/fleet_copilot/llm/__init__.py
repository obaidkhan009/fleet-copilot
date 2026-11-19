from .base import LLMBackend, LLMTurn


def make_backend(name: str) -> LLMBackend:
    if name == "offline":
        from .offline import OfflineBackend
        return OfflineBackend()
    if name == "claude":
        from .claude import ClaudeBackend
        return ClaudeBackend()
    raise ValueError(f"unknown backend '{name}' (expected 'claude' or 'offline')")


__all__ = ["LLMBackend", "LLMTurn", "make_backend"]
