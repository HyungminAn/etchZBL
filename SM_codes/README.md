# PlasmaEtchSimulator
A python code to run ion beam etch simulation.

# How to install
```python
pip install -e . # (requires setup.py)
```

# Inputs
- `input.yaml`: input parameters
- `str_shoot_0.coo`: initial structure file

# How to run
```shell
python main.py input.yaml
```

# Input parameters
Please refet to `PlasmaEtchSimulator/input.yaml`:
```yaml
pot_params:
  elmlist: # List of elements in the system
    - Si
    - O
    - C
    - H
    - F
  pottype: 'bpnn'  # potential type; 'bpnn' or 'e3gnn'
  # path to the potential file
  potloc: '/data2/andynn/Etch/05_EtchingMD/ver2/potential_saved_bestmodel'
  d3: false  # whether to use D3 dispersion correction (only for e3nn)

etch_params:
  ion: 'CF3'            # Ion species; 'CF', 'CF2', 'CF3', 'CH2F', or 'CHF2'
  energy: 200           # Ion energy; unit: eV
  timestep: 0.002       # timestep; unit: ps
  angle: 0              # Ion incident angle; unit: degree
  fix: 6                # Thickness of the fixed region at the bottom of slab; unit: Angstrom

  # Ions will be inserted at this height; unit: Angstrom
  # This is initial value; it will be updated if the slab addition occurs.
  incident_height: 80
  incident_range: 4     # Thickness of the ion-insertion region; unit: Angstrom
  tempheight: 6         # Temperature of the slab will be measured for atoms above this height; unit: Angstrom

  # Ions above this height would evaporate (deleted); unit: Angstrom
  # This is initial value; it will be updated if the slab addition occurs.
  evapheight: 85

  # After each MD run, remove clusters whose height is higher than slab by this
  # amount; unit: Angstrom
  remove_high_clutser_height_crit: 10

  # When slab is added, run MD by this many steps for equilibration; unit: step
  equilibration_step: 1000

  # after NVE, the NVT-MD would be run at this temperature; unit: K
  slab_temperature: 300
  # NVT MD would be halted if the slab temperature goes below this value; unit: K
  halt_temperature: 300

  nvt_method: 'langevin'  # 'nose-hoover' or 'langevin'
  log_step: 40 # step
  compress : True # Compress the dump file (needs COMPRESS package in LAMMPS compiling)
  print_velocity: False # Print the velocity of the atoms in the dump file

run_params:
  nstart: 1 # Starting index of ion shooting
  nshoot: 4000 # Total number of ion shootings

  # pen_depth: slab addition/subtraction based on penetration depth
  # (the lowest height among all C,H,F atoms)
  slab_modify_crit: 'pen_depth'  # 'pen_depth' or 'height'

  n_atoms_limit: 2000  # The maximum number of atom allowed in the simulation.
  # Only the atoms above this height would be considered for checking C,H,F ratio; unit: Angstrom
  h_CHF_crit: 6.0
  # If the ratio of C,H,F atoms among all atoms is above this value,
  # the simulation cell is thought to be under deposition condition,
  # and slab subtraction would be checked; unit: fraction (0.7=70%)
  CHF_ratio_crit: 0.7
  # Slab addition would occur if the lowest C,H,F atom is below this height
  h_penetrate_crit: 6.0
  # Thickness of the slab to be added/subtracted; unit: Angstrom
  h_sub_amount: 3.0
  h_add_amount: 3.0

  box_height: 200 # height of the simulation box; unit: Angstrom

  # check slab update (add, subtract) every this many shots
  slab_update_iteration: 10
  # Location of amorphous bulk and slab structures
  bulk_loc: '/data2_1/andynn/Etch/05_EtchingMD/ver2_2/01_AmorphousSlab/223cell/01_Bulk/FINAL.coo'
  slab_loc: '/data2_1/andynn/Etch/05_EtchingMD/ver2_2/01_AmorphousSlab/223cell/02_slab/relaxed.coo'

  # List of species considered as byproducts
  byproduct_list:
    - 'SiF4'
    - 'SiHF3'
    - 'SiH2F2'
    - 'SiH3F'
    - 'SiH4'

    - 'SiF2'

    - 'O2'
    - 'H2'
    - 'F2'
    - 'CO'
    - 'HF'

    - 'CF4'
    - 'CHF3'
    - 'CH2F2'
    - 'CH3F'
    - 'CH4'

    - 'CF2'

    - 'H2O'
    - 'OF2'
    - 'OHF'

    - 'CH2O'
    - 'CHFO'
    - 'COF2'
    - 'CO2'

    # (unsaturated species are not considered in the original paper)
    # - 'CF3'
    # - 'CH2F'
    # - 'CHF2'
    # - 'CH3'
    # - 'CHF'
    # - 'CH2'
    # - 'CF'
    # - 'CH'

etc_params:
  lmp_loc: '/home/andynn/lammps/build_etch_d2_loki34_compress/lmp' # path to lammps binary
  nprocs: 'auto' # number of processors; 'auto' or integer
  device: 'cpu' # 'cpu' or 'gpu'
  log: 'auto' # log file; 'auto' or filename
  run_loc:  '.' # working directory
```
