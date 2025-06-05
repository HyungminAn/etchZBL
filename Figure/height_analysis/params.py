from dataclasses import dataclass
@dataclass
class PARAMS:
    @dataclass
    class LAMMPS:
        ATOM_IDX_Si, ATOM_NUM_Si = 1, 14
        ATOM_IDX_O, ATOM_NUM_O = 2, 8
        ATOM_IDX_C, ATOM_NUM_C = 3, 6
        ATOM_IDX_H, ATOM_NUM_H = 4, 1
        ATOM_IDX_F, ATOM_NUM_F = 5, 9

        READ_OPTS = {
            'format': 'lammps-data',
            'Z_of_type': {
                ATOM_IDX_Si: ATOM_NUM_Si,
                ATOM_IDX_O: ATOM_NUM_O,
                ATOM_IDX_C: ATOM_NUM_C,
                ATOM_IDX_H: ATOM_NUM_H,
                ATOM_IDX_F: ATOM_NUM_F,
            }
        }

    @dataclass
    class PLOT:
        @dataclass
        class COLORS:
            COLOR_LIST = {
                'default': '#609af7',

                '1': '#075c29',
                '2': '#609af7',
                '3': '#de3535',

                'layer': {
                    'mixed': '#d1ae4d',
                    'film': '#3d2e04',
                },

                'density': {
                    'mixed': '#d1ae4d',
                    'film': '#3d2e04',
                },

                'fc_ratio': {
                    'mixed': '#d1ae4d',
                    'film': '#3d2e04',
                },

                'spx_ratio': {
                    'sp3': '#075c29',
                    'sp2': '#609af7',
                    'sp': '#de3535',
                    'others': '#d1ae4d',
                },

                'neighbor': {
                    'SiC_cluster': 'orange',
                    'Fluorocarbon': 'green',
                    'C4': 'pink',
                    'C3': 'red',
                    'C2': 'blue',
                    'etc': 'purple',
                    }

            }

            COLORS = {
                'CF_500': COLOR_LIST['1'],
                'CF_750': COLOR_LIST['1'],
                'CF_1000': COLOR_LIST['1'],

                'CF2_75': COLOR_LIST['1'],
                'CF2_100': COLOR_LIST['1'],
                'CF2_250': COLOR_LIST['1'],
                'CF2_500': COLOR_LIST['1'],
                'CF2_750': COLOR_LIST['1'],
                'CF2_1000': COLOR_LIST['1'],

                'CF3_25': COLOR_LIST['1'],
                'CF3_50': COLOR_LIST['1'],
                'CF3_100': COLOR_LIST['1'],
                'CF3_250': COLOR_LIST['1'],
                'CF3_500': COLOR_LIST['1'],
                'CF3_750': COLOR_LIST['1'],
                'CF3_1000': COLOR_LIST['1'],

                'CH2F_750': COLOR_LIST['1'],
                'CH2F_1000': COLOR_LIST['1'],

                'CHF2_250': COLOR_LIST['1'],
                'CHF2_500': COLOR_LIST['1'],
                'CHF2_750': COLOR_LIST['1'],
                'CHF2_1000': COLOR_LIST['1'],

                'CF3_10': COLOR_LIST['3'],
                'CF2_25': COLOR_LIST['3'],
                'CH2F_100': COLOR_LIST['3'],
                }

        @dataclass
        class HEIGHT:
            READ_INTERVAL = 10
            CUTOFF_PERCENTILE = 98  # percentile
            SHIFT = 6.0
            CARBON_FILM_CUTOFF = (85, 15)  # percentile
            TRUNCATE_INITIAL_REGION = 0.2  # 10^16 cm-2

        @dataclass
        class ATOMDENSITY:
            SPACING = 0.1  # density profile
            BW_WIDTH = 0.2  # density profile
            CUTOFF_RATIO_FILM = 0.6  # density profile
            CUTOFF_RATIO_MIXED = 0.1  # density profile
            CUTOFF_RATIO_FILM_UPPER = 0.5  # density profile
            ELEM_LIST = ['Si', 'O', 'C', 'H', 'F']

    @dataclass
    class CONVERT:
        ANGST_TO_NM = 0.1
        CONV_FACTOR_TO_CM2 = 1/9000  # 9000 incidence corresponds to 10^17 cm^2
        ION_CONVERT_DICT = {
                'CF': 'CF$^{+}$',
                'CF2': 'CF${}_{2}^{+}$',
                'CF3': 'CF${}_{3}^{+}$',
                'CH2F': 'CH${}_{2}$F$^{+}$',
                'CHF2': 'CHF${}_{2}^{+}$',
                }
