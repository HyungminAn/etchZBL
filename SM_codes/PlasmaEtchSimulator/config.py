"""
Configuration loading and default values.

The YAML config is merged with DEFAULTS so that every key always has a value.
Users only need to specify what differs from the defaults.
"""
import yaml


DEFAULTS: dict = {
    'pot_params': {
        'elmlist': ['Si', 'O', 'C', 'H', 'F'],
        'd3': False,
    },
    'etch_params': {
        'timestep': 0.002,
        'angle': 0,
        'incident_range': 4,
        'tempheight': 6,
        'nvt_method': 'langevin',
        'log_step': 40,
        'compress': True,
        'print_velocity': False,
        'equilibration_step': 1000,
        'halt_temperature': 300,
        'remove_high_cluster_height_crit': 10,
    },
    'run_params': {
        'nstart': 1,
        'slab_modify_crit': 'pen_depth',
        'slab_update_iteration': 10,
        'box_height': 200,
        'n_atoms_limit': 2000,
        'h_CHF_crit': 6.0,
        'CHF_ratio_crit': 0.7,
        'h_penetrate_crit': 6.0,
        'h_sub_amount': 3.0,
        'h_add_amount': 3.0,
        'byproduct_list': [],
    },
    'etc_params': {
        'nprocs': 'auto',
        'device': 'cpu',
        'log': 'auto',
        'run_loc': '.',
        'graph_backend': 'networkx',
        'runner': 'subprocess',
        'poll_interval': 10,
    },
}


def load_config(path: str) -> dict:
    """Load YAML config and merge with defaults."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _merge_defaults(raw, DEFAULTS)


def _merge_defaults(config: dict, defaults: dict) -> dict:
    """Deep-merge config onto defaults (config values win)."""
    merged: dict = {}
    for section, section_defaults in defaults.items():
        user_section = config.get(section, {})
        if isinstance(section_defaults, dict):
            merged[section] = {**section_defaults, **user_section}
        else:
            merged[section] = user_section if section in config else section_defaults
    for section in config:
        if section not in merged:
            merged[section] = config[section]
    return merged
