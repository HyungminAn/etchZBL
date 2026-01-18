import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt


class NVE_NVT_Analyzer:
    def __init__(self, src, path_output):
        self.src = src
        self.path_output = path_output

    def run(self):
        path_save = f'{self.path_output}.pickle'
        if os.path.exists(path_save):
            with open(path_save, 'rb') as f:
                data = pickle.load(f)
            self.plot(data, f'{self.path_output}.png')
            return

        file_list = []
        for root, _, files in os.walk(self.src):
            for file in files:
                if not file.startswith('thermo_'):
                    continue
                if not os.path.exists(os.path.join(root, f'str_shoot_{self.strip_name(file)}_after_mod.coo')):
                    continue
                file_list.append(os.path.join(root, file))
        file_list = sorted(file_list, key=lambda x: self.strip_name(x))
        result_dict = {}
        for file in file_list:
            TEMP_NVE_END, NVT_TIME = self.read_time(file)
            result_dict[self.strip_name(file)] = (TEMP_NVE_END, NVT_TIME)
        with open(path_save, 'wb') as f:
            pickle.dump(result_dict, f)
        self.plot(result_dict, f'{self.path_output}.png')

    def read_time(self, src_thermo):
        '''
        In order of STEP, TIME, TEMP
        '''
        IDX_STEP, IDX_TIME, IDX_TEMP = 0, 1, 2
        NVE_TIME = 2.0
        matrix = np.loadtxt(src_thermo, skiprows=2, usecols=(IDX_STEP, IDX_TIME, IDX_TEMP))
        _, unique_indices = np.unique(matrix[:, IDX_STEP], return_index=True)
        reduced_matrix = matrix[unique_indices]
        TEMP_NVE_END = reduced_matrix[reduced_matrix[:, IDX_TIME] > NVE_TIME][0, IDX_TEMP]
        NVT_TIME = reduced_matrix[-1, IDX_TIME] - NVE_TIME
        return TEMP_NVE_END, NVT_TIME

    def strip_name(self, file):
        return int(file.split('/')[-1].split('_')[1].replace('.dat', ''))

    def plot(self, data, path_output):
        key = sorted(list(data.keys()), key=lambda x: int(x))
        TEMP_NVE_END = [data[k][0] for k in key]
        NVT_TIME = [data[k][1] for k in key]

        plot_opts = {
                'linestyle': '-',
                'alpha': 0.7,
                'markersize': 2,
                'linewidth': 1,
                }

        plt.rcParams.update({'font.size': 21})
        fig, (ax_temp, ax_time) = plt.subplots(2, 1, figsize=(10, 10))
        ax_temp.plot(key, TEMP_NVE_END, color='orange', **plot_opts)
        ax_temp.set_xlabel('Incidence')
        ax_temp.set_ylabel('Temperature (K)')
        ax_temp.set_title('Temperature at the beginning of NVT')

        ax_time.plot(key, NVT_TIME, color='black', **plot_opts)
        ax_time.set_xlabel('Incidence')
        ax_time.set_ylabel('Time (ps)')
        ax_time.set_title('NVT Time')

        title = path_output
        plt.suptitle(title)

        fig.tight_layout()
        fig.savefig(path_output)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python checkNVTtime.py [src] [path_output]')
        sys.exit(1)

    src = sys.argv[1]
    path_output = sys.argv[2]
    analyzer = NVE_NVT_Analyzer(src, path_output)
    analyzer.run()
