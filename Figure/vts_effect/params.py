from dataclasses import dataclass

@dataclass
class PARAMS:
    src = "/home/andynn/02_Etch/05_vts_effect_check/04_SmallcellMD"
    pot_type = ["pot_0", "pot_1", "pot_2"]
    ion_list = ["CF", "CF3", "CH2F", "CHF2"]
    energy_list = [10, 30]
    # ion_list = ["CHF2"]
    # energy_list = [30]
    incidences = [i+1 for i in range(50)]

    LAMMPS_READ_OPTS = {
            "format": "lammps-dump-text",
            "index": ":",
            }

    sample_interval = 0.5 # ps
    n_max_sample = 5

    STYLE_DICT = {
            'pot_0': {
                'linestyle': 'solid',
                },
            'pot_1': {
                'linestyle': 'dotted',
                },
            'pot_2': {
                'linestyle': (0, (1, 10)),
                'alpha': 0.5,
                },
            }

    COLOR_DICT = {
        'Si': 'blue',
        'O': 'orange',
        'C': 'green',
        'H': 'red',
        'F': 'purple',
    }

    ION_CONVERT_DICT = {
            'CF': 'CF',
            'CF3': 'CF$_3$',
            'CH2F': 'CH$_2$F',
            'CHF2': 'CHF$_2$',
            }

    POT_CONVERT_DICT = {
            'pot_0': 'primitive',
            'pot_1': 'NoVtS',
            'pot_2': 'NoVtsNoCHF',
            }

    LAMMPS_SAVE_OPTS = {
            "format": "lammps-data",
            "specorder": ["Si", "O", "C", "H", "F"],
            }

    VASP_SAVE_OPTS = {
            "format": "vasp",
            "sort": True,
            }

    parity_plot_color_dict = {
            'pot_0': 'blue',
            'pot_1': 'orange',
            'pot_2': 'green',
            }
