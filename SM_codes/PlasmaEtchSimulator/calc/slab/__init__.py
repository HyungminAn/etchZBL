from .height import HeightBasedModifier
from .pendepth import PenDepthModifier


def get_modifier(strategy: str):
    """Return a slab modifier instance for the given strategy name."""
    registry = {
        'height': HeightBasedModifier,
        'pen_depth': PenDepthModifier,
    }
    if strategy not in registry:
        raise ValueError(
            f"Unknown slab_modify_crit '{strategy}'. "
            f"Choose from: {list(registry)}"
        )
    return registry[strategy]()
