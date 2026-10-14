"""
LAMMPS region and group setup block shared by etch and anneal simulations.
"""
from .base import LammpsBlock


class RegionBlock(LammpsBlock):
    """
    Defines the fixed bottom layer, the unfixed slab, and the temperature
    measurement region. Used identically in both etch and anneal inputs.
    """

    def __init__(self, timestep: float, fix_height: float, tempheight: float) -> None:
        self.timestep = timestep
        self.fix_height = fix_height
        self.tempheight = tempheight

    def render(self) -> str:
        return f"""\
timestep                        {self.timestep}
variable   fix_height equal     {self.fix_height}
variable   temp_height equal    {self.tempheight}

neighbor       2 bin
neigh_modify   every 10 delay 0 check yes
compute        _rg all gyration

# Bottom fixed layer
region  rFixed  block   INF INF INF INF 0.0 $(v_fix_height)
group gBottom region rFixed
velocity gBottom set 0.0 0.0 0.0
fix freeze_bottom gBottom setforce 0.0 0.0 0.0

# Free slab
region  rUnfixed  block   INF INF INF INF $(v_fix_height) INF
group gUnfixed  dynamic  all  region rUnfixed

# Temperature measurement region
region  rtemp block   INF INF INF INF ${{temp_height}} INF
group gtemp  dynamic  all  region rtemp

## Adaptive load balancing and variable timestep
comm_style tiled
fix f_balance  all balance 0 1.05 rcb
fix dt_fix all dt/reset 1   NULL {self.timestep}  0.1 emax 1 units box

"""
