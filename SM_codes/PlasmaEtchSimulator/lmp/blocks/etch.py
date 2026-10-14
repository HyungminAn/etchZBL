"""
LAMMPS etch simulation block: ion deposition + NVE collision + NVT cooling.
"""
import re

import numpy as np
from ase.data import atomic_masses, atomic_numbers

from .base import LammpsBlock
from .thermostat import get_thermostat_block


def _ion_mass_expr(ion: str) -> str:
    """Build a LAMMPS arithmetic expression for the total ion mass."""
    ndict: dict[str, int] = {}
    for match in re.finditer(r'([A-Z][a-z]*)(\d*)', ion):
        elm, cnt = match.group(1), match.group(2)
        ndict[elm] = ndict.get(elm, 0) + (int(cnt) if cnt else 1)
    parts = [
        f'{cnt}*{atomic_masses[atomic_numbers[elm]]}'
        for elm, cnt in ndict.items()
    ]
    return '+'.join(parts)


class EtchRunBlock(LammpsBlock):
    """Full etch MD script: variable setup, ion deposition, NVE + NVT phases."""

    def __init__(
        self,
        elmlist: list,
        ion: str,
        energy: float,
        angle: float,
        incident_height: float,
        incident_range: float,
        evap_height: float,
        slab_temperature: float,
        halt_temperature: float,
        nvt_method: str,
        timestep: float,
        log_step: int,
        compress: bool,
        print_velocity: bool,
    ) -> None:
        self.elmlist = elmlist
        self.ion = ion
        self.energy = energy
        self.angle = angle
        self.incident_height = incident_height
        self.incident_range = incident_range
        self.evap_height = evap_height
        self.slab_temperature = slab_temperature
        self.halt_temperature = halt_temperature
        self.nvt_method = nvt_method
        self.timestep = timestep
        self.log_step = log_step
        self.compress = compress
        self.print_velocity = print_velocity

    def render(self) -> str:
        ion_mass = _ion_mass_expr(self.ion)
        count_expr = ''.join(f' $(c_atom_count[{i + 1}])' for i in range(len(self.elmlist)))

        dump_style = 'custom/gz' if self.compress else 'custom'
        dump_path = r'dump_${i}.gz' if self.compress else r'dump_${i}.lammps'

        if self.print_velocity:
            dump_fields = 'id type element xu yu zu fx fy fz vx vy vz'
            dump_fmt = '%d %d %s %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f'
        else:
            dump_fields = 'id type element xu yu zu fx fy fz'
            dump_fmt = '%d %d %s %.2f %.2f %.2f %.2f %.2f %.2f'

        thermostat = get_thermostat_block(
            self.nvt_method,
            self.slab_temperature,
            self.halt_temperature,
            self.timestep,
        )

        elm_str = ' '.join(self.elmlist)
        seeds = np.random.randint(0, 1_000_000)

        return f"""\
variable ion_mass   equal       {ion_mass}
variable ion_energy equal       {self.energy}
variable angle_target equal     {self.angle}
molecule mol_file               {self.ion}.dat toff 0

variable   incident_height equal   {self.incident_height}
variable   incident_range equal    {self.incident_range}
variable   evap_height equal    {self.evap_height}
variable   slab_temperature equal {self.slab_temperature}

# Incident region
region rIncident block   EDGE EDGE EDGE EDGE $(v_incident_height) $(v_incident_height + v_incident_range)

# Evaporation region
region  rUpper block   INF INF INF INF $(v_evap_height) INF
fix fEvap all evaporate 500 100 rUpper 1

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

fix PrintTherm all print 10 "" file thermo_${{i}}.dat screen no title "STEP TIME TEMP POT_E KIN_E etotal{' '.join([' num_' + e for e in self.elmlist])}"
unfix PrintTherm
fix PrintTherm all print ${{total_step}} "$(step) $(time) $(c_TEMP) $(c_POT_E) $(c_KIN_E) $(etotal){count_expr}" append thermo_${{i}}.dat screen no

variable    V_initial       equal   $(sqrt((v_ion_energy*1.6e-19)/(0.5*v_ion_mass*1e-3/6.02e23))*1e10/1e12)
variable angle_incidence equal $(abs(normal(v_angle_target,5,v_seeds)/180*PI))
variable angle_phi equal $(random(0,360,v_seeds)/180*PI)
variable vel_Z equal $(-v_V_initial*cos(v_angle_incidence))
variable vel_X equal $(-v_V_initial*sin(v_angle_incidence)*cos(v_angle_phi))
variable vel_Y equal $(-v_V_initial*sin(v_angle_incidence)*sin(v_angle_phi))
variable pos_seed equal $(round(random(1,10000,v_seeds)))
fix  fDepo  all  deposit  1  0  1000000  ${{pos_seed}} region  rIncident local 6 6 2.5 near  6.0  &
    mol mol_file vx $(v_vel_X) $(v_vel_X) vy $(v_vel_Y) $(v_vel_Y) vz $(v_vel_Z)  $(v_vel_Z) attempt 100

# NVE collision phase: 2 ps
fix fNVE all nve
variable start_time equal $(time)
label run_loop
variable run_i loop 50
run 200
if "$(time-v_start_time) >= 2" then "jump SELF break"
next run_i
jump SELF run_loop
label break

# NVT cooling phase
variable run_i delete
unfix fNVE
{thermostat.render()}
unfix fNVT
unfix fDepo
undump dMovie
write_data ${{strout}}
"""
