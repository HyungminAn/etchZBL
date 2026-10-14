"""
NVT thermostat blocks for the cooling phase of MD simulations.
"""
from .base import LammpsBlock


class NoseHooverBlock(LammpsBlock):
    """Nosé-Hoover NVT with a temperature halt condition."""

    def __init__(self, temperature: float, halt_temperature: float, timestep: float) -> None:
        self.temperature = temperature
        self.halt_temperature = halt_temperature
        self.timestep = timestep

    def render(self) -> str:
        tau = self.timestep * 100
        return (
            f'fix fNVT gUnfixed nvt temp {self.temperature:.1f} {self.temperature:.1f} {tau}\n'
            f'fix F_halt all halt 200 v_myTEMP <= {self.halt_temperature} error continue\n'
            'run 100000\n'
        )


class LangevinBlock(LammpsBlock):
    """Langevin thermostat (NVE + Langevin) with a temperature halt condition."""

    def __init__(self, temperature: float, halt_temperature: float, timestep: float) -> None:
        self.temperature = temperature
        self.halt_temperature = halt_temperature
        self.timestep = timestep

    def render(self) -> str:
        tau = self.timestep * 100
        return (
            f'fix fNVE_for_langevin gUnfixed nve\n'
            f'fix fNVT gUnfixed langevin {self.temperature:.1f} {self.temperature:.1f}'
            f' {tau} ${{seeds}}\n'
            f'fix F_halt all halt 200 v_myTEMP <= {self.halt_temperature} error continue\n'
            'run 100000\n'
        )


class LangevinFixedTBlock(LammpsBlock):
    """Langevin thermostat that runs for a fixed duration (no halt condition)."""

    def __init__(self, temperature: float, halt_temperature: float, timestep: float,
                 duration_ps: float = 2.0) -> None:
        self.temperature = temperature
        self.halt_temperature = halt_temperature
        self.timestep = timestep
        self.duration_ps = duration_ps

    def render(self) -> str:
        tau = self.timestep * 100
        n_steps = int(self.duration_ps / self.timestep)
        return (
            f'fix fNVE_for_langevin gUnfixed nve\n'
            f'fix fNVT gUnfixed langevin {self.temperature:.1f} {self.temperature:.1f}'
            f' {tau} ${{seeds}}\n'
            f'fix F_halt all halt 200 v_myTEMP <= {self.halt_temperature} error continue\n'
            f'run {n_steps}\n'
        )


def get_thermostat_block(method: str,
                         temperature: float,
                         halt_temperature: float,
                         timestep: float) -> LammpsBlock:
    """Return a thermostat block for the named NVT method."""
    registry = {
        'nose-hoover': NoseHooverBlock,
        'langevin': LangevinBlock,
        'langevin_fixed_t': LangevinFixedTBlock,
    }
    if method not in registry:
        raise ValueError(
            f"Unknown nvt_method '{method}'. "
            f"Choose from: {list(registry)}"
        )
    return registry[method](temperature, halt_temperature, timestep)
