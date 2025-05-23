import pickle
import time
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
from sklearn.metrics import mean_absolute_error as MAE

import datashader as ds
import datashader.transfer_functions as tf
from datashader.mpl_ext import dsshow
import pandas as pd


mpl.rcParams['font.family'] = 'Arial'

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'{func.__name__} took {time.time()-start:.2f} seconds')
        return result
    return wrapper

class DataLoader:
    @timeit
    def get_data(self, path_yaml):
        with open(path_yaml, 'rb') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        tot_data, tot_data_split = {}, {}
        for system_type, config in configs.items():
            print(f'Loading data for {system_type}')

            with open(config['structure_idx'], 'rb') as f:
                structure_idx = pickle.load(f)

            energy_dft = np.loadtxt(config['energy']['dft'])
            energy_nnp = np.loadtxt(config['energy']['nnp'])
            nions = np.loadtxt(config['nions'])
            print('Energy loaded')

            energy_dft = energy_dft / nions
            energy_nnp = energy_nnp / nions

            with open(config['force']['dft'], 'rb') as f:
                force_dft = pickle.load(f)
            with open(config['force']['nnp'], 'rb') as f:
                force_nnp = pickle.load(f)
            print('Force loaded')
            mae_energy = MAE(energy_dft, energy_nnp)
            mae_force = MAE(force_dft, force_nnp)

            data = {
                    'energy': {
                        'dft': energy_dft,
                        'nnp': energy_nnp,
                        'mae': mae_energy
                        },
                    'force': {
                        'dft': force_dft,
                        'nnp': force_nnp,
                        'mae': mae_force
                        },
                    }

            data_split = self.split_data(data, structure_idx, nions)
            tot_data[system_type] = data
            tot_data_split[system_type] = data_split

        return tot_data, tot_data_split

    @timeit
    def split_data(self, data, structure_idx, nions):
        data_split = {}
        nions_cumsum = np.cumsum([0] + [i for i in nions], dtype=int)
        for key, idx_list in structure_idx.items():
            idx_list = np.array(idx_list)
            e_dft = data['energy']['dft'][idx_list]
            e_nnp = data['energy']['nnp'][idx_list]

            idx_list_force = []
            for idx in idx_list:
                my_list = np.arange(nions_cumsum[idx], nions_cumsum[idx+1])
                idx_list_force.extend(my_list)
            idx_list_force = np.array(idx_list_force)

            f_dft = data['force']['dft'][idx_list_force]
            f_nnp = data['force']['nnp'][idx_list_force]
            mae_energy = MAE(e_dft, e_nnp)
            mae_force = MAE(f_dft, f_nnp)
            key = self.modify_key(key)

            if key not in data_split:
                data_split[key] = {
                        'energy': {
                            'dft': e_dft,
                            'nnp': e_nnp,
                            'mae': mae_energy,
                            },
                        'force': {
                            'dft': f_dft,
                            'nnp': f_nnp,
                            'mae': mae_force,
                            },
                        }
            else:
                # Add to existing data
                data_split[key]['energy']['dft'] = np.concatenate((data_split[key]['energy']['dft'], e_dft))
                data_split[key]['energy']['nnp'] = np.concatenate((data_split[key]['energy']['nnp'], e_nnp))
                data_split[key]['force']['dft'] = np.concatenate((data_split[key]['force']['dft'], f_dft))
                data_split[key]['force']['nnp'] = np.concatenate((data_split[key]['force']['nnp'], f_nnp))
                data_split[key]['energy']['mae'] = MAE(data_split[key]['energy']['dft'], data_split[key]['energy']['nnp'])
                data_split[key]['force']['mae'] = MAE(data_split[key]['force']['dft'], data_split[key]['force']['nnp'])
        return data_split

    @staticmethod
    def modify_key(key):
        '''
        modify some keys (typos)
        '''
        if key == "01_bulk:bulk_MD:alpha" or key == "01_bulk:bulk_MD:beta":
            key = 'bulk_md'
        elif key == "slab_chaneling":
            key = "slab_channeling"
        return key

class DataPlotter:
    @timeit
    def plot_total(self, tot_data, fig_name):
        plt.rcParams.update({'font.size': 18})
        n_row, n_col = len(tot_data), 2
        fig, axes = plt.subplots(n_row, n_row, figsize=(6*n_col, 6*n_row))

        for ((ax_energy, ax_force), (system_type, data)) in zip(axes, tot_data.items()):
            e_dft, e_nnp, e_mae = data['energy']['dft'], data['energy']['nnp'], data['energy']['mae']
            f_dft, f_nnp, f_mae = data['force']['dft'], data['force']['nnp'], data['force']['mae']
            f_dft, f_nnp = f_dft.flatten(), f_nnp.flatten()
            print('Plotting...')

            mask = (e_dft < 0)
            ax_energy.scatter(e_dft[mask], e_nnp[mask], s=5, c='black', alpha=0.3)
            # df_e = pd.DataFrame({'x': e_dft[mask], 'y': e_nnp[mask]})
            # dsartist = dsshow(
            #         df_e,
            #         ds.Point("x", "y"),
            #         ds.count(),
            #         cmap='plasma',
            #         norm='linear',
            #         plot_width=100,
            #         plot_height=100,
            #         ax=ax_energy,
            #         )
            # fig.colorbar(dsartist, ax=ax_energy, label='count')

            force_cut = 50  # eV/A
            mask = np.logical_and(f_dft > -force_cut, f_dft < force_cut)
            df_f = pd.DataFrame({ 'x': f_dft[mask], 'y': f_nnp[mask], })
            dsartist = dsshow(
                    df_f,
                    ds.Point("x", "y"),
                    ds.count(),
                    cmap='plasma',
                    norm='log',
                    plot_width=200,
                    plot_height=200,
                    ax=ax_force,
                    )
            fig.colorbar(dsartist, ax=ax_force, label='count')

            self.decorate_axes(ax_energy,
                               ax_force,
                               e_mae,
                               f_mae,
                               e_dft,
                               f_dft,
                               f_cut=force_cut)

        for ((ax_energy, ax_force), (system_type)) in zip(axes, tot_data.keys()):
            if system_type == 'SiO2':
                text = '(a) SiO$_2$'
            elif system_type == 'Si3N4':
                text = '(b) Si$_3$N$_4$'
            else:
                raise ValueError(f'Unknown system type: {system_type}')

            ax_energy.text(-0.1, 1.2, text, transform=ax_energy.transAxes,
                           ha='left', va='top', fontsize=20)

        fig.tight_layout(pad=1.0)
        fig.savefig(f'{fig_name}.png')
        fig.savefig(f'{fig_name}.pdf')
        # fig.savefig('result.eps')

    def decorate_axes(self,
                      ax_energy,
                      ax_force,
                      e_mae,
                      f_mae,
                      e_dft,
                      f_dft,
                      f_cut):
        ax_energy.set_xlabel(r'E$_{\text{DFT}}$ (eV/atom)')
        ax_energy.set_ylabel(r'E$_{\text{NNP}}$ (eV/atom)')

        ax_force.set_xlabel(r'F$_{\text{DFT}}$' + ' (eV/$\mathrm{\AA}$)')
        ax_force.set_ylabel(r'F$_{\text{NNP}}$' + ' (eV/$\mathrm{\AA}$)')

        ax_force.set_xlim(-f_cut, f_cut)
        ax_force.set_ylim(-f_cut, f_cut)

        ax_energy.set_aspect('equal')
        ax_force.set_aspect('equal')

        ax_energy.axline([0, 0], [1, 1], color='grey', linestyle='--', linewidth=0.5)
        ax_force.axline([0, 0], [1, 1], color='grey', linestyle='--', linewidth=0.5)

        ax_energy.set_title(f'Energy ({len(e_dft)} data points)', fontsize=14)
        ax_force.set_title(f'Force ({len(f_dft) // 3} data points)', fontsize=14)

        text_energy = f"MAE: {e_mae*1000:.2f} meV/atom"
        text_force = f"MAE: {f_mae:.2f}" + " eV/$\mathrm{\AA}$"
        box_options = {
            'ha': 'left',
            'va': 'top',
            # 'bbox': {
            #     'boxstyle': 'round',
            #     'facecolor': 'wheat',
            #     'alpha': 0.5,
            #     },
            'fontsize': 18,
        }
        ax_energy.text(0.05, 0.95, text_energy, transform=ax_energy.transAxes, **box_options)
        ax_force.text(0.05, 0.95, text_force, transform=ax_force.transAxes, **box_options)

    @timeit
    def summarize_data(self, tot_data, tot_data_split):
        results = {}
        for system_type in tot_data.keys():
            data = tot_data[system_type]
            data_split = tot_data_split[system_type]

            result = [('Total',
                       data['energy']['mae'],
                       len(data['energy']['dft']),
                       data['force']['mae'],
                       len(data['force']['dft']))
                      ]
            for key, data in data_split.items():
                result.append((key,
                               data['energy']['mae'],
                               len(data['energy']['dft']),
                               data['force']['mae'],
                               len(data['force']['dft'])
                               ))

            result = sorted(result, key=lambda x: x[0])
            results[system_type] = result

        keys = []
        for system_type, result in results.items():
            for i in [key for key, _, _, _, _ in result]:
                if i not in keys:
                    keys.append(i)

        print(f"{'structure':<20} ", f"{'E MAE (meV/atom)':<15} {'n_data':<10} {'F MAE (eV/A)':<15} {'n_data':<10}" * len(results))
        for key in keys:
            print(f"{key:<20} ", end='')
            for system_type, result in results.items():
                result = [r for r in result if r[0] == key]
                if result:
                    _, mae_energy, n_energy, mae_force, n_force = result[0]
                    print(f"{mae_energy*1000:<15.2f} {n_energy:<10d} {mae_force:<15.3f} {n_force:<10d}", end='')
                else:
                    mae_energy = n_energy = mae_force = n_force = '_'
                    print(f"{mae_energy:<15} {n_energy:<10} {mae_force:<15} {n_force:<10}", end='')
            print()


@timeit
def main():
    tot_data, tot_data_split = DataLoader().get_data('input.yaml')
    DataPlotter().plot_total(tot_data, 'result')
    DataPlotter().summarize_data(tot_data, tot_data_split)

    # system = 'Si3N4'
    # for sub_system in tot_data_split[system].keys():
    #     DataPlotter().plot_total({system: tot_data_split[system][sub_system],
    #                               f'{system}_': tot_data_split[system][sub_system]}, sub_system)


if __name__ == '__main__':
    main()
