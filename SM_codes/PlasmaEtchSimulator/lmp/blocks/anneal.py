"""
LAMMPS annealing block: NVT equilibration without ion deposition.
"""
import numpy as np

from .base import LammpsBlock


class AnnealRunBlock(LammpsBlock):
    """NVT equilibration script for slab relaxation after atom addition."""

    def __init__(
        self,
        elmlist: list,
        slab_temperature: float,
        nvt_method: str,
        timestep: float,
        equilibration_step: int,
        log_step: int,
        compress: bool,
        print_velocity: bool,
    ) -> None:
        self.elmlist = elmlist
        self.slab_temperature = slab_temperature
        self.nvt_method = nvt_method
        self.timestep = timestep
        self.equilibration_step = equilibration_step
        self.log_step = log_step
        self.compress = compress
        self.print_velocity = print_velocity

    def render(self) -> str:
        count_expr = ''.join(f' $(c_atom_count[{i + 1}])' for i in range(len(self.elmlist)))
        elm_str = ' '.join(self.elmlist)
        seeds = np.random.randint(0, 1_000_000)
        tau = self.timestep * 100

        dump_style = 'custom/gz' if self.compress else 'custom'
        dump_path = r'dump_${i}_anneal.gz' if self.compress else r'dump_${i}_anneal.lammps'
        thermo_path = r'thermo_${i}_anneal.dat'

        if self.print_velocity:
            dump_fields = 'id type element xu yu zu fx fy fz vx vy vz'
            dump_fmt = '%d %d %s %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f'
        else:
            dump_fields = 'id type element xu yu zu fx fy fz'
            dump_fmt = '%d %d %s %.2f %.2f %.2f %.2f %.2f %.2f'

        if self.nvt_method == 'nose-hoover':
            nvt_block = (
                f'fix fNVT gUnfixed nvt temp {self.slab_temperature:.1f}'
                f' {self.slab_temperature:.1f} {tau}\n'
            )
        elif self.nvt_method in ('langevin', 'langevin_fixed_t'):
            nvt_block = (
                f'fix fNVE_for_langevin gUnfixed nve\n'
                f'fix fNVT gUnfixed langevin {self.slab_temperature:.1f}'
                f' {self.slab_temperature:.1f} {tau} {seeds}\n'
            )
        else:
            raise ValueError(f"Unknown nvt_method '{self.nvt_method}'")

        nvt_block += f'run {self.equilibration_step}\n'

        return f"""\
compute POT_E all pe
compute TEMP gtemp temp
compute KIN_E gtemp ke
variable myTEMP equal c_TEMP
variable seeds equal {seeds}
variable total_step equal {self.log_step}

compute atom_count all count/type atom

thermo 0
thermo_modify lost ignore flush yes

dump dMovie all {dump_style} ${{total_step}} {dump_path} {dump_fields}
dump_modify dMovie sort id element {elm_str} time yes format line "{dump_fmt}"

fix PrintTherm all print 10 "" file {thermo_path} screen no title "STEP TIME TEMP POT_E KIN_E etotal{' '.join([' num_' + e for e in self.elmlist])}"
unfix PrintTherm
fix PrintTherm all print ${{total_step}} "$(step) $(time) $(c_TEMP) $(c_POT_E) $(c_KIN_E) $(etotal){count_expr}" append {thermo_path} screen no

{nvt_block}
unfix fNVT
undump dMovie
write_data ${{strout}}
"""
