import numpy as np
from ase.data import atomic_masses, atomic_numbers

from PlasmaEtchSimulator.lmp.functions import _mole_to_ndict
from PlasmaEtchSimulator.lmp.etch import LAMMPSetch


class LAMMPSanneal(LAMMPSetch):
    def __init__(self, pot_params, etch_params, run_params, logger=None):
        super().__init__(pot_params, etch_params, run_params, logger)
        self.lmpname = 'lammps_anneal.in'

    def _anneal_run(self):
        lines = ''
        # Get ion type
        ion_type : str = self.ion
        ion_num_dict = _mole_to_ndict(ion_type)
        ion_mass = ''
        for  elm, num in ion_num_dict.items():
            ion_mass += f'{num}*{atomic_masses[atomic_numbers[elm]]}+'
        ion_mass = ion_mass[:-1]
        # etch_setting include the energy, angle, ion, ...
        anneal_setting = f'''
timestep                        {self.timestep}
variable   fix_height equal     {self.fix}

variable   temp_height equal    {self.tempheight}
variable   slab_temperature equal {self.slab_temperature}
'''

        # Slab region is setting script
        anneal_region = f'''
neighbor       2 bin
neigh_modify   every 10 delay 0 check yes
compute        _rg all gyration

# Bottom fix layer
region  rFixed  block   INF INF INF INF 0.0 $(v_fix_height)
group gBottom region rFixed
velocity gBottom set 0.0 0.0 0.0
fix freeze_bottom gBottom setforce 0.0 0.0 0.0

# Incident region
region  rUnfixed  block   INF INF INF INF $(v_fix_height) INF
group gUnfixed  dynamic  all  region rUnfixed

# Temperature region
region  rtemp block   INF INF INF INF ${{temp_height}} INF
group gtemp  dynamic  all  region rtemp

## Balance of simulation & variable time step when high energy transfer
comm_style tiled
fix f_balance  all balance 0 1.05 rcb
fix dt_fix all dt/reset 1   NULL {self.timestep}  0.1 emax 1 units box
'''
        count = ''
        for it, elm in enumerate(self.elmlist):
            count += f' $(c_atom_count[{it+1}])'

        if self.compress:
            dump_style = 'custom/gz'
            dump_path = r'dump_${i}_anneal.gz'
        else:
            dump_style = 'custom'
            dump_path = r'dump_${i}_anneal.lammps'

        if self.etch_params['print_velocity']:
            dump_print_format = 'id type element xu yu zu fx fy fz vx vy vz'
            dump_modify_format = '%d %d %s %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f'
        else:
            dump_print_format = 'id type element xu yu zu fx fy fz'
            dump_modify_format = '%d %d %s %.2f %.2f %.2f %.2f %.2f %.2f'

        thermo_path = r'thermo_${i}_anneal.dat'
        anneal_logging = f'''
compute POT_E all pe
compute TEMP gtemp temp
compute KIN_E gtemp ke
variable myTEMP equal c_TEMP
variable seeds equal {np.random.randint(0, 1000000)}
variable total_step equal {self.log_step}

# "compute count/type" is available after LAMMPS version 15Jun2023.
# See "https://docs.lammps.org/compute_count_type.html"
compute atom_count all count/type atom

thermo 0
thermo_modify lost ignore flush yes

dump dMovie all {dump_style} ${{total_step}} {dump_path} {dump_print_format}
dump_modify dMovie sort id element {' '.join(self.elmlist)} time yes format line "{dump_modify_format}"

fix PrintTherm all print 10 "" file {thermo_path} screen no title "STEP TIME TEMP POT_E KIN_E etotal num_Si num_N num_C num_H num_F"
unfix PrintTherm
fix PrintTherm all print ${{total_step}} "$(step) $(time) $(c_TEMP) $(c_POT_E) $(c_KIN_E) $(etotal) {count}" append {thermo_path} screen no
'''
        if self.nvt_method == 'nose-hoover':
            line_nvt = (
                f'fix fNVT gUnfixed nvt temp {self.slab_temperature:.1f} {self.slab_temperature:.1f} {self.timestep*100}\n'
                )
        elif self.nvt_method == 'langevin':
            line_nvt = (
                f'fix fNVE_for_langevin gUnfixed nve\n'
                f'fix fNVT gUnfixed langevin {self.slab_temperature:.1f} {self.slab_temperature:.1f} {self.timestep*100} ${{seeds}}\n'
                )
        elif self.nvt_method == 'langevin_fixed_t':
            langevin_fixed_mdtime = 2  # ps
            langevin_fixed_t = int(langevin_fixed_mdtime / self.timestep)
            line_nvt = (
                f'fix fNVE_for_langevin gUnfixed nve\n'
                f'fix fNVT gUnfixed langevin {self.slab_temperature:.1f} {self.slab_temperature:.1f} {self.timestep*100} ${{seeds}}\n'
                )
        else:
            raise ValueError(f'NVT method {self.nvt_method} is not supported')

        line_nvt += f'run {self.equilibration_step}\n'
        anneal_run = f'''
{line_nvt}

unfix fNVT
undump dMovie
write_data ${{strout}}
'''
        lines += anneal_setting + anneal_region + anneal_logging + anneal_run
        return lines

    def generate_lammps_input(self, lmpdir):
        location = lmpdir +'/'+self.lmpname
        header = self._pot_header()
        outlines = header

        body = self._anneal_run()
        outlines += body

        with open(location, 'w') as f:
            f.write(outlines)
            self.lmploc = location
            self.print(f"LAMMPS input {location} written")
