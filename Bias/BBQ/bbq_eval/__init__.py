"""BBQ-Eval: bias and fairness assessment for question answering."""

from importlib.metadata import PackageNotFoundError, version

from .main import main

try:
    __version__ = version("bbq-eval")
except PackageNotFoundError:
    __version__ = "1.0.0"

__email__ = ""


def evaluate_main(*args, **kwargs):
    from .evaluate import main as evaluate

    return evaluate(*args, **kwargs)


def get_accuracy(*args, **kwargs):
    from .evaluate import get_accuracy as accuracy

    return accuracy(*args, **kwargs)


__all__ = ["main", "evaluate_main", "get_accuracy"]
