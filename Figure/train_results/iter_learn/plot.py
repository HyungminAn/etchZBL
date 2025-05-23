import time
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_absolute_error as MAE

mpl.rcParams['font.family'] = 'Arial'

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'{func.__name__} takes {time.time()-start:.2f} sec')
        return result
    return wrapper

@timeit
def get_data():
    with open('input.yaml') as f:
        tot_config = yaml.load(f, Loader=yaml.FullLoader)

    result_total = {}
    for system_type, config in tot_config.items():
        result_total[system_type] = {}
        for iter_count, energy_dict in config.items():
            result = {}
            with open(energy_dict['dft'], 'r') as f:
                data_dft = f.readlines()
            with open(energy_dict['nnp'], 'r') as f:
                data_nnp = f.readlines()

            for line_dft, line_nnp in zip(data_dft, data_nnp):
                try:
                    ion_type, ion_E, incidence, count, energy, nions = line_dft.split()
                    energy_dft = float(energy)
                    energy_nnp = float(line_nnp.strip())
                    nions = int(nions)
                except:
                    ion_type, ion_E, incidence, count = line_dft.split()
                    energy_dft = None
                    energy_nnp = None
                    nions = None

                key = f'{ion_type} {ion_E}eV'
                if key not in result:
                    result[key] = {'dft': [], 'nnp': [], 'nions': []}
                result[key]['dft'].append(energy_dft)
                result[key]['nnp'].append(energy_nnp)
                result[key]['nions'].append(nions)

            result_total[system_type][iter_count] = result

    return result_total


@timeit
def print_MAE(mae_dict):
    type_list = [
        'CF 10eV',
        'CF 30eV',
        'CF3 10eV',
        'CF3 30eV',
        'CH2F 10eV',
        'CH2F 30eV',
        'CHF2 10eV',
        'CHF2 30eV',
    ]
    iter_list = [i for i in range(4)]

    for iter_count in iter_list:
        print(f'Iter {iter_count}: ', end='')
        for my_type in type_list:
            key = f'{my_type} Iter_{iter_count}'
            mae = mae_dict[key]
            print(f'{mae*1000:.2f}', end=' & ')
        print('')

@timeit
def plot(result_total, name, fig_label):
    n_row, n_col = 4, 2
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(n_row, n_col, figsize=(5 * n_col, 3.5 * n_row))
    ax_dict = {
            'CF 10eV': ax[0, 0],
            'CF 30eV': ax[0, 1],
            'CF3 10eV': ax[1, 0],
            'CF3 30eV': ax[1, 1],
            'CH2F 10eV': ax[2, 0],
            'CH2F 30eV': ax[2, 1],
            'CHF2 10eV': ax[3, 0],
            'CHF2 30eV': ax[3, 1],
            }

    COLOR_ITER = {
            0: 'orange',
            1: 'green',
            2: 'blue',
            3: 'black',
            }

    mae_dict = {}

    for iter_count, result in result_total.items():
        for key, value in result.items():
            ax = ax_dict[key]
            y_dft = np.array(value['dft'])
            y_nnp = np.array(value['nnp'])
            nions = np.array(value['nions'])

            y_dft_per_ion = np.array([i/nion
                                      for i, nion in zip(y_dft, nions)
                                      if i is not None and nion is not None])
            y_nnp_per_ion = np.array([i/nion
                                      for i, nion in zip(y_nnp, nions)
                                      if i is not None and nion is not None ])
            mae = MAE(y_dft_per_ion, y_nnp_per_ion)

            y_diff = [(dft - nnp)/nion * 1000
                      if dft is not None and nnp is not None and nion is not None
                      else None
                      for dft, nnp, nion in zip(y_dft, y_nnp, nions)
                      ]
            y_diff = np.array(y_diff)

            x = np.array([i for i in range(len(y_dft))])
            ax.plot(x, y_diff,
                    # label=f'{iter_count} (MAE: {mae*1000:.2f} meV/atom)',
                    label=f'{iter_count}',
                    color=COLOR_ITER[int(iter_count.split('_')[-1])],
                    alpha=0.7)

            mae_dict[f'{key} {iter_count}'] = mae

    y_min, y_max = 0, 0
    for ax in ax_dict.values():
        y_min = min(y_min, ax.get_ylim()[0])
        y_max = max(y_max, ax.get_ylim()[1])

    for idx, (key, ax) in enumerate(ax_dict.items(), start=1):
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlabel('Index')
        ax.set_ylabel('Energy diff (meV/atom)')

        title = fig_label.split()[0].replace(')', f'-{idx})') \
                + f' {fig_label.split()[1]}' \
                + f', {convert_key(key)}'
        ax.text(-0.2, 1.2, title, transform=ax.transAxes, fontsize=14, va='top', ha='left')

        ax.set_xlim(0, 250)
        ax.set_ylim(y_min, y_max)
        ax.axhline(y=0, color='grey', linestyle='--')

    fig.tight_layout(pad=2.0)
    fig.savefig(f'{name}.png')
    fig.savefig(f'{name}.pdf')
    fig.savefig(f'{name}.eps')
    print(f'{name}.png is saved.')

    print_MAE(mae_dict)

def convert_key(key):
    if 'CF3' in key:
        key = key.replace('CF3', 'CF${}_{3}^{+}$')
    elif 'CH2F' in key:
        key = key.replace('CH2F', 'CH$_2$F$^{+}$')
    elif 'CHF2' in key:
        key = key.replace('CHF2', 'CHF${}_{2}^{+}$')
    elif 'CF ' in key:
        key = key.replace('CF', 'CF$^{+}$')
    return key


@timeit
def main():
    result_total = get_data()
    mapping = {
            0: '(a) SiO$_2$',
            1: '(b) Si$_3$N$_4$',
            }
    for idx, (name, result) in enumerate(result_total.items()):
        plot(result, name, mapping[idx])


if __name__ == '__main__':
    main()
