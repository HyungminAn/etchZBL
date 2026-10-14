"""
LAMMPSInputBuilder: assembles a LAMMPS input file from an ordered list of blocks.

Usage example:

    from PlasmaEtchSimulator.lmp.builder import LAMMPSInputBuilder
    from PlasmaEtchSimulator.lmp.blocks import BpnnPotentialBlock, RegionBlock, EtchRunBlock

    plan = [
        BpnnPotentialBlock(elmlist, potloc),
        RegionBlock(timestep, fix_height, tempheight),
        EtchRunBlock(...),
    ]
    LAMMPSInputBuilder(plan).write('lammps.in')

To add a new routine, define a new LammpsBlock subclass and insert it into the plan.
"""
import os

from .blocks.base import LammpsBlock
from .blocks.potential import BpnnPotentialBlock, E3gnnPotentialBlock, SevenNetPotentialBlock
from .blocks.regions import RegionBlock
from .blocks.etch import EtchRunBlock
from .blocks.anneal import AnnealRunBlock


class LAMMPSInputBuilder:
    """Renders a LAMMPS input file from an ordered sequence of LammpsBlock objects."""

    def __init__(self, plan: list[LammpsBlock]) -> None:
        self.plan = plan

    def render(self) -> str:
        return ''.join(block.render() for block in self.plan)

    def write(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.render())

    @classmethod
    def etch(cls, config: dict) -> 'LAMMPSInputBuilder':
        """Build a standard etch simulation input from a config dict."""
        pot = config['pot_params']
        ep = config['etch_params']
        return cls([
            _make_potential_block(pot),
            RegionBlock(ep['timestep'], ep['fix'], ep['tempheight']),
            EtchRunBlock(
                elmlist=pot['elmlist'],
                ion=ep['ion'],
                energy=ep['energy'],
                angle=ep['angle'],
                incident_height=ep['incident_height'],
                incident_range=ep['incident_range'],
                evap_height=ep['evapheight'],
                slab_temperature=ep['slab_temperature'],
                halt_temperature=ep['halt_temperature'],
                nvt_method=ep['nvt_method'],
                timestep=ep['timestep'],
                log_step=ep['log_step'],
                compress=ep['compress'],
                print_velocity=ep['print_velocity'],
            ),
        ])

    @classmethod
    def anneal(cls, config: dict) -> 'LAMMPSInputBuilder':
        """Build a standard annealing input from a config dict."""
        pot = config['pot_params']
        ep = config['etch_params']
        return cls([
            _make_potential_block(pot),
            RegionBlock(ep['timestep'], ep['fix'], ep['tempheight']),
            AnnealRunBlock(
                elmlist=pot['elmlist'],
                slab_temperature=ep['slab_temperature'],
                nvt_method=ep['nvt_method'],
                timestep=ep['timestep'],
                equilibration_step=ep['equilibration_step'],
                log_step=ep['log_step'],
                compress=ep['compress'],
                print_velocity=ep['print_velocity'],
            ),
        ])


def _make_potential_block(pot: dict) -> LammpsBlock:
    pottype = pot['pottype']
    elmlist = pot['elmlist']
    potloc = pot['potloc']
    if pottype == 'bpnn':
        return BpnnPotentialBlock(elmlist, potloc)
    if pottype == 'e3gnn':
        return E3gnnPotentialBlock(elmlist, potloc, d3=pot.get('d3', False))
    if pottype == 'sevennet':
        return SevenNetPotentialBlock(elmlist, potloc)
    raise ValueError(f"Unknown pottype '{pottype}'. Choose from: 'bpnn', 'e3gnn', 'sevennet'.")
