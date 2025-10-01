import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde


class DataLoader:
    def run(self, path_log):
        with open(path_log, 'r') as f:
            lines = f.readlines()
        x, y = [], []
        for line in lines:
            if line.startswith('*') or 'E_rxn' in line:
                continue
            *_, E_dft, E_nnp = line.split()
            if E_dft == '-' or E_nnp == '-':
                continue
            E_dft, E_nnp = float(E_dft), float(E_nnp)
            x.append(E_dft)
            y.append(E_nnp)
        x, y = np.array(x), np.array(y)
        return (x, y)

class DataPlotter:
    def __init__(self, cal_type_1, cal_type_2):
        self.cal_type_1 = cal_type_1
        self.cal_type_2 = cal_type_2

    def run(self, data):
        print("plotting...", end='')
        fig, axes = self.generate_figure()
        for (system, rxn_xy), ax in zip(data.items(), axes):
            rxn_1, rxn_2, Z = self.sort_data(rxn_xy)
            self.plot(rxn_1, rxn_2, Z, ax)
            self.add_statistics(rxn_xy, ax)
            self.set_xy_limits(ax, rxn_1, rxn_2)
            self.decorate(rxn_xy, ax, system)
        self.save(fig)
        print("done")

    def generate_figure(self):
        plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(9.0, 3.5)
        for ax in axes:
            ax.set_aspect('equal','box')
        return fig, axes

    def sort_data(self, data):
        E_1, E_2 = data
        E_1, E_2 = np.array(E_1), np.array(E_2)
        values = np.vstack((E_1, E_2))
        kernel = gaussian_kde(values)
        kernel.set_bandwidth(bw_method=kernel.factor*3)
        Z = kernel(values)
        Z /= np.min(Z[np.nonzero(Z)])
        idx = Z.argsort()
        DFT_rxn, NNP_rxn, Z = E_1[idx], E_2[idx], Z[idx]

        return DFT_rxn, NNP_rxn, Z

    def plot(self, rxn_1, rxn_2, Z, ax):
        cmap = plt.get_cmap('Blues')
        norm = matplotlib.colors.LogNorm(vmin=1)
        plot_options = {
            'c': Z+1,
            'cmap': cmap,
            's': 1,
            'norm': norm,
            'zorder': 2,
            }
        sc = ax.scatter(rxn_1, rxn_2, **plot_options)
        cax = inset_axes(ax,
                         width="5%",
                         height="100%",
                         loc='center left',
                         bbox_to_anchor=(1.02, 0.0, 1, 1),
                         bbox_transform=ax.transAxes,
                         borderpad=0.1)
        cb = plt.colorbar(sc, cax=cax)
        cb.set_label('Density')

    def add_statistics(self, data, ax):
        E_1, E_2 = data
        mae = np.mean(np.abs(E_1-E_2))
        pearson = np.corrcoef(E_1,E_2)[0,1]
        r2 = 1-np.sum(np.power(E_1-E_2,2))/np.sum(np.power(E_1-np.mean(E_1),2))
        text = ""
        # text += f"MAE: {mae:.2f} eV\n"
        # text += f"Pearson: {pearson:.4f}\n"
        text += f"R$^2$: {r2:.4f}"
        box_options = {
            'transform': ax.transAxes,
            'ha': 'left',
            'va': 'top',
            # 'bbox': {
            #     'boxstyle': 'round',
            #     'facecolor': 'wheat',
            #     'alpha': 0.5,
            #     },
            # 'fontsize': 5,
        }
        ax.text(0.05, 0.95, text, **box_options)

    def set_xy_limits(self, ax, rxn_1, rxn_2):
        x_min = np.min([np.min(rxn_1), np.min(rxn_2)])
        x_max = np.max([np.max(rxn_1), np.max(rxn_2)])
        if x_min < 0:
            x_min *= 1.1
        else:
            x_min *= 0.9
        if x_max < 0:
            x_max *= 0.9
        else:
            x_max *= 1.1
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min, x_max)

    def decorate(self, data, ax, title):
        ax.axline((0, 0), slope=1, linestyle='--', zorder=1, color='grey')
        ax.set_aspect('equal')

        cal_type_1 = self.cal_type_1.upper()
        cal_type_2 = self.cal_type_2.upper()
        x_label = fr"$E^{{\rm{{{cal_type_1}}}}}_{{\rm{{rxn}}}}$ (eV)"
        y_label = fr"$E^{{\rm{{{cal_type_2}}}}}_{{\rm{{rxn}}}}$ (eV)"
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        E_1, _ = data
        n_data = len(E_1)
        # title = f"{title} ({n_data} rxns)"
        # ax.set_title(title)
        text_convert_dict = {
                'SiO2': f'(a) SiO$_2$',
                'Si3N4': f'(b) Si$_3$N$_4$',
                }
        text = text_convert_dict.get(title) + f' ({n_data} rxns)'
        ax.text(-0.2, 1.1, text, transform=ax.transAxes, ha='left', va='top')

    def save(self, fig):
        name = 'rxn_energy_metric'
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2)
        fig.savefig(f"{name}.png", dpi=200)
        fig.savefig(f"{name}.pdf")
        fig.savefig(f"{name}.eps")

def main():
    if len(sys.argv) != 2:
        print("Usage: python plot.py input.yaml")
        sys.exit(1)

    path_yaml = sys.argv[1]
    with open(path_yaml, 'r') as f:
        paths = yaml.safe_load(f)

    cal_type_1 = 'dft'
    cal_type_2 = 'nnp'

    dl = DataLoader()
    data = {}
    for system, path in paths.items():
        data[system] = dl.run(path)

    pl = DataPlotter(cal_type_1, cal_type_2)
    pl.run(data)


if __name__ == "__main__":
    main()
