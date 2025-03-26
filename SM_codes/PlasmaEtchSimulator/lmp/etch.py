import numpy as np
from ase.data import atomic_masses, atomic_numbers

from PlasmaEtchSimulator.lmp.functions import _mole_to_ndict
from PlasmaEtchSimulator.params import D2Params, ZBLParams


class LAMMPSetch:
    def __init__(self,
                pot_params : dict,
                etch_params : dict,
                etc_params : dict,
                logger = None) -> None:
        """Initilize the LAMMPSetch class

        Args:
            pot_params (dict): parameters related to potential definition
            etch_params (dict): parameters related to etching simulation like molecule, slab, energy, angle ...
            etc_params (dict): parameters related to etc like log file, dump file, ...

        self.elmlist    : list
        self.pottype    : str
        self.potloc     : str
        self.d3         : bool
        ## Time step for the simulation
        self.timestep     : float
        ##  Maximum and minimum incidenth height of the ion
        self.incident_height : float
        self.incident_range  : float
        ## Height of the slab where the temperature is calculated
        self.tempheight   : float
        ## Height of the slab where the evaporation is occur
        self.evapheight   : float
        ## Ion, energy, angle to use in the simulation
        self.ion            : str
        self.energy         : float
        self.angle          : float
        ## The slab temperature
        self.slab_temperature     : float
        ## The upshift is mearning that this region have any ensemble (NVT, NVE) to prevent the injection to below PBC
        # If this occur, the error occur in the simulation when using e3gnn !
        self.upshift      : float
        # The fix is the number of the fix to use in the simulation
        self.fix          : float
        self.vaccum       : float
        self.log_step     : int
        self.compress     : bool

        """
        self.etc_params = etc_params

        self.pot_params = pot_params
        self.etch_params = etch_params

        for key, value in pot_params.items():
            setattr(self, key, value)

        for key, value in etch_params.items():
            setattr(self, key, value)

        for key, value in etc_params.items():
            setattr(self, key, value)

        self.log = logger
        self.lmpname = 'lammps.in' ## Inner variable to store the lammps input file name
        self.lmploc  = None ## LAMMPS script location which use later or other function

    def _update_self(self):
        update_list = [
                'incident_height',
                'evapheight',
                ]
        params_list = [
                self.pot_params,
                self.etch_params,
                self.etc_params,
                ]
        for my_key in update_list:
            for params in params_list:
                if my_key in params:
                    setattr(self, my_key, params[my_key])

    def _pot_header(self):
        mass = ''
        pair_coeff = ''

        if self.pottype == 'e3gnn':
            potin, pair_coeff = self._get_pair_coeff_e3gnn()
        elif self.pottype == 'bpnn':
            potin, pair_coeff = self._get_pair_coeff_bpnn()

        for it, elm in enumerate(self.elmlist):
            atn = atomic_numbers[elm]
            mass += f'mass  {it+1} {atomic_masses[atn]} \n'

        front = f'''units           metal
newton          on
dimension       3
boundary        p p p
atom_style 	atomic
box tilt large

read_data       ${{strin}}
'''
        return front + '\n' + mass + '\n' + potin + '\n\n' + pair_coeff + '\n\n'

    def _get_pair_coeff_e3gnn(self):
        ndict = {}
        if self.d3 == True:
            potin = f'pair_style hybrid/overlay e3gnn  d3 9000 1600 damp_bj pbe'
            pair_coeff =    f"pair_coeff * * e3gnn {self.potloc}  {' '.join(self.elmlist)}\n"
            pair_coeff +=    f"pair_coeff * * d3  {' '.join(self.elmlist)}\n"
        elif self.pottype == 'e3gnn':
            potin = f'pair_style e3gnn'
            pair_coeff =    f"pair_coeff * *  {self.potloc}  {' '.join(self.elmlist)}\n"
        else:
            potin = f'pair_style e3gnn/zbl'
            pair_coeff =    f"pair_coeff * *  {self.potloc}  {' '.join(self.elmlist)}\n"
            for it, elm in enumerate(self.elmlist):
                for init, inelm in enumerate(self.elmlist[it:]):
                    if elm not in ndict:
                        ndict[elm] = atomic_numbers[elm]
                    if inelm not in ndict:
                        ndict[inelm] = atomic_numbers[inelm]
                    pair_coeff += f'pair_coeff {it+1} {it+init+1} {ndict[elm]} {ndict[inelm]}\n'
        return potin, pair_coeff

    def _get_pair_coeff_bpnn(self):
        ndict = {}
        d2_params = D2Params.values
        zbl_params = ZBLParams.values
        d2_key = list(d2_params.keys())
        for key in d2_key:
            d2_params[(key[1], key[0])] = d2_params[key]
            zbl_params[(key[1], key[0])] = zbl_params[key]
        potin = f'pair_style hybrid/overlay nn/intel   d2 15 0.75 20 zbl/pair'
        pair_coeff =    f"pair_coeff * * nn/intel {self.potloc}  {' '.join(self.elmlist)}\n"
        for it, elm in enumerate(self.elmlist):
            for init, inelm in enumerate(self.elmlist[it:]):
                if elm not in ndict:
                    ndict[elm] = atomic_numbers[elm]
                if inelm not in ndict:
                    ndict[inelm] = atomic_numbers[inelm]
                zbl_param = zbl_params[(ndict[elm], ndict[inelm])]
                pair_coeff += f'pair_coeff {it+1} {it+init+1} zbl/pair {ndict[elm]} {ndict[inelm]} {zbl_param[0]} {zbl_param[1]}\n'

        for it, elm in enumerate(self.elmlist):
            for init, inelm in enumerate(self.elmlist[it:]):
                if elm not in ndict:
                    ndict[elm] = atomic_numbers[elm]
                if inelm not in ndict:
                    ndict[inelm] = atomic_numbers[inelm]
                d2_param = d2_params[(ndict[elm], ndict[inelm])]
                pair_coeff += f'pair_coeff {it+1} {it+init+1} d2 {d2_param[0]} {d2_param[1]}\n'

        return potin, pair_coeff

    def _etch_run(self):
        lines = ''
        # Get ion type
        ion_type : str = self.ion
        ion_num_dict = _mole_to_ndict(ion_type)
        ion_mass = ''
        for  elm, num in ion_num_dict.items():
            ion_mass += f'{num}*{atomic_masses[atomic_numbers[elm]]}+'
        ion_mass = ion_mass[:-1]
        # etch_setting include the energy, angle, ion, ...
        etch_setting = f'''
variable ion_mass   equal       {ion_mass}
variable ion_energy equal       {self.energy}
variable angle_target equal     {self.angle}
molecule mol_file               {f'{ion_type}.dat'} toff 0

timestep                        {self.timestep}
variable   fix_height equal     {self.fix}

variable   incident_height equal   {self.incident_height}
variable   incident_range equal    {self.incident_range}

variable   temp_height equal    {self.tempheight}
variable   evap_height equal    {self.evapheight}
variable   slab_temperature equal {self.slab_temperature}

'''

        # Slab region is setting script
        etch_region = f'''
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

region rIncident block   EDGE EDGE EDGE EDGE $(v_incident_height) $(v_incident_height + v_incident_range)

# Temperature region
region  rtemp block   INF INF INF INF ${{temp_height}} INF
group gtemp  dynamic  all  region rtemp

# Evaporate region
region  rUpper block   INF INF INF INF $(v_evap_height) INF
fix fEvap all evaporate 500 100 rUpper 1

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
            dump_path = r'dump_${i}.gz'
        else:
            dump_style = 'custom'
            dump_path = r'dump_${i}.lammps'

        if self.etch_params['print_velocity']:
            dump_print_format = 'id type element xu yu zu fx fy fz vx vy vz'
            dump_modify_format = '%d %d %s %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f'
        else:
            dump_print_format = 'id type element xu yu zu fx fy fz'
            dump_modify_format = '%d %d %s %.2f %.2f %.2f %.2f %.2f %.2f'

        etch_logging = f'''
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

fix PrintTherm all print 10 "" file thermo_${{i}}.dat screen no title "STEP TIME TEMP POT_E KIN_E etotal num_Si num_N num_C num_H num_F"
unfix PrintTherm
fix PrintTherm all print ${{total_step}} "$(step) $(time) $(c_TEMP) $(c_POT_E) $(c_KIN_E) $(etotal) {count}" append thermo_${{i}}.dat screen no
'''
        etch_sputter = r'''
variable    V_initial       equal   $(sqrt((v_ion_energy*1.6e-19)/(0.5*v_ion_mass*1e-3/6.02e23))*1e10/1e12)

variable angle_incidence equal $(abs(normal(v_angle_target,5,v_seeds)/180*PI))

variable angle_phi equal $(random(0,360,v_seeds)/180*PI)
variable vel_Z equal $(-v_V_initial*cos(v_angle_incidence))
variable vel_X equal $(-v_V_initial*sin(v_angle_incidence)*cos(v_angle_phi))
variable vel_Y equal $(-v_V_initial*sin(v_angle_incidence)*sin(v_angle_phi))
variable pos_seed equal $(round(random(1,10000,v_seeds)))
fix  fDepo  all  deposit  1  0  1000000  ${pos_seed} region  rIncident local 6 6 2.5 near  6.0  &
    mol mol_file vx $(v_vel_X) $(v_vel_X) vy $(v_vel_Y) $(v_vel_Y) vz $(v_vel_Z)  $(v_vel_Z) attempt 100
'''
        if self.nvt_method == 'nose-hoover':
            line_nvt = (
                f'fix fNVT gUnfixed nvt temp {self.slab_temperature:.1f} {self.slab_temperature:.1f} {self.timestep*100}\n'
                f'fix F_halt all halt 200 v_myTEMP <= {self.halt_temperature} error continue\n'
                'run 100000\n'
                )
        elif self.nvt_method == 'langevin':
            line_nvt = (
                f'fix fNVE_for_langevin gUnfixed nve\n'
                f'fix fNVT gUnfixed langevin {self.slab_temperature:.1f} {self.slab_temperature:.1f} {self.timestep*100} ${{seeds}}\n'
                f'fix F_halt all halt 200 v_myTEMP <= {self.halt_temperature} error continue\n'
                'run 100000\n'
                )
        elif self.nvt_method == 'langevin_fixed_t':
            langevin_fixed_mdtime = 2  # ps
            langevin_fixed_t = int(langevin_fixed_mdtime / self.timestep)
            line_nvt = (
                f'fix fNVE_for_langevin gUnfixed nve\n'
                f'fix fNVT gUnfixed langevin {self.slab_temperature:.1f} {self.slab_temperature:.1f} {self.timestep*100} ${{seeds}}\n'
                f'fix F_halt all halt 200 v_myTEMP <= {self.halt_temperature} error continue\n'
                f'run {langevin_fixed_t}\n'
                )
        else:
            raise ValueError(f'NVT method {self.nvt_method} is not supported')
        etch_run = f'''
fix fNVE all nve

variable start_time equal $(time)
label run_loop
variable run_i loop 50
run 200
if "$(time-v_start_time) >= 2" then "jump SELF break"
next run_i
jump SELF run_loop
label break

#NVT cooling
variable run_i delete
unfix fNVE
{line_nvt}

unfix fNVT
unfix fDepo
undump dMovie
write_data ${{strout}}
'''
        lines += etch_setting + etch_region + etch_logging + etch_sputter + etch_run
        return lines

    def generate_lammps_input(self, lmpdir):
        location = lmpdir +'/'+self.lmpname
        header = self._pot_header()
        outlines = header

        body = self._etch_run()
        outlines += body

        with open(location, 'w') as f:
            f.write(outlines)
            self.lmploc = location
            self.print(f"LAMMPS input {location} written")

    def get_lammps_input(self):
        return self.lmploc

    def print(self, line : str):
        self.log.print(line)
