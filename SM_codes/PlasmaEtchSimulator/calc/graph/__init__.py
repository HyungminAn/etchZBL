from .base import GraphBuilderBase
from .networkx_builder import NetworkXBuilder
from .scipy_builder import ScipyBuilder


def get_builder(backend: str = 'networkx') -> GraphBuilderBase:
    """Return a graph builder instance for the given backend name."""
    registry = {
        'networkx': NetworkXBuilder,
        'scipy': ScipyBuilder,
    }
    if backend not in registry:
        raise ValueError(
            f"Unknown graph backend '{backend}'. "
            f"Available backends: {list(registry)}"
        )
    return registry[backend]()
