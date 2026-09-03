"""infervolt: measure -> diagnose -> fix -> verify -> recipe -> remember."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("infervolt")
except PackageNotFoundError:  # pragma: no cover - editable install without metadata
    __version__ = "0.0.0"

__all__ = ["__version__"]
