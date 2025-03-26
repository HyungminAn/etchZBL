from dataclasses import dataclass

@dataclass
class KEY:
    # Collect all key varialbes
    POT_PARAMS  = 'pot_params'
    ELMLIST     = 'elmlist'
    POT_TYPE    = 'pottype'
    POT_LOC     = 'potloc'
    USE_VDW     = 'd3'

    ETCH_PARAMS = 'etch_params'
    TIMESTEP = 'timestep'
    ION = 'ion'

    ENERGY = 'energy'
    ANGLE  = 'angle'
    UPSHIFT = 'upshift'
    FIX = 'fix'
    VACUUM              = 'vacuum'
    TEMP_HEIGHT         = 'tempheight'
    EVAP_HEIGHT         = 'evapheight'
    SLAB_TEMPERATURE    = 'slab_temperature'
    INCIDENT_HEIGHT     = 'incident_height'
    INCIDENT_RANGE      = 'incident_range'
    LOG_STEP = 'log_step'
    COMPRESS = 'compress'

    RUN_PARAMS = 'run_params'
    NSTART = 'nstart'
    NSHOOT = 'nshoot'
    SLAB_MAX_HEIGHT = 'slab_max_height'
    SLAB_MIN_HEIGHT = 'slab_min_height'
    SLAB_CENTER_HEIGHT = 'slab_center_height'
    SLAB_UPDATE_ITERATION = 'slab_update_iteration'
    BULK_LOC = 'bulk_loc'
    SLAB_LOC = 'slab_loc'
    BOX_HEIGHT = 'box_height'
    # BOND_SCALE = 'bond_scale'
    BYPRODUCT_LIST = 'byproduct_list'

    ETC_PARAMS = 'etc_params'
    LOG = 'log'
    LMP_LOC = 'lmp_loc'
    RUN_LOC = 'run_loc'
    # MODULE_LOC = 'module_loc'
    NPROCS = 'nprocs'
    DEVICE = 'device'
    DEBUG  = 'debug'
