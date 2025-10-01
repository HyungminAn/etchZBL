import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

class DataLoader:
    def run(self, path_root, key):
        path_save = f'data_{key}.dat'
        if os.path.exists(path_save):
            with open(path_save, 'rb') as f:
                return pickle.load(f)

        file_list = []
        for root, dirs, files in os.walk(path_root):
            for file in files:
                if not file.endswith('.lammps') or 'anneal' in file:
                    continue
                file_list.append(os.path.join(root, file))
        file_list = sorted(file_list,
            key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
        result = {}
        for file in file_list:
            time = self.read_time(file)
            key = file.split('/')[-1].split('.')[0].split('_')[-1]
            if time is not None:
                result[key] = time
                print(file, key, time)
        with open(path_save, 'wb') as f:
            pickle.dump(result, f)

        return result

    def read_time(self, path_file):
        with open(path_file, 'r') as f:
            lines = f.readlines()
        line = lines[-1]
        if not line.startswith('Total wall time'):
            return None
        time = self.string_to_time(line.split()[-1])
        return time

    def string_to_time(self, string):
        h, m, s = map(float, string.split(':'))
        return h * 3600 + m * 60 + s

class DataPlotter:
    def run(self, data):
        plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        for key, data in data.items():
            x = np.array(sorted([int(k) for k in data.keys()]))
            y = np.array([data[k] for k in sorted(data.keys())])

            for x_, y_ in zip(x, y):
                if y_ > 500:
                    print(f'Warning: {key} at step {x_} has a large time {y_} s')
            ax.plot(x, y, label=key, alpha=0.5)
        ax.legend(loc='upper right', fontsize=8, frameon=False)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.set_xlabel('Step')
        ax.set_ylabel('Time (s)')
        fig.tight_layout()
        fig.savefig('result.png')

def main():
    src_dict = {
        'primitive': '/home/andynn/02_Etch/05_vts_effect_check/05_LargecellMD/pot_0/CF_100',
        'NoVts': '/home/andynn/02_Etch/05_vts_effect_check/05_LargecellMD/pot_1/CF_100',
        'NoVtsNoCHF': '/home/andynn/02_Etch/05_vts_effect_check/05_LargecellMD/pot_2/CF_100',
        }
    data = {}
    dl = DataLoader()
    for key, path in src_dict.items():
        print(f'Processing {key}...')
        result = dl.run(path, key)
        data[key] = result

    dp = DataPlotter()
    dp.run(data)


if __name__ == '__main__':
    main()
