from .base import RunnerBase
from .subprocess_runner import SubprocessRunner
from .polling_runner import PollingRunner


def get_runner(runner_type: str, **kwargs) -> RunnerBase:
    """Return a runner instance for the given runner type name."""
    registry = {
        'subprocess': SubprocessRunner,
        'polling': PollingRunner,
    }
    if runner_type not in registry:
        raise ValueError(
            f"Unknown runner '{runner_type}'. "
            f"Choose from: {list(registry)}"
        )
    return registry[runner_type](**kwargs)
