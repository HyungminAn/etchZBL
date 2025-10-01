import sys
import yaml
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class PARAMS:
    colors = {
        'SiO2': 'red',
        'Si3N4': 'blue',
    }

    convert_dict = {
        'SiO2': 'SiO$_2$',
        'Si3N4': 'Si$_3$N$_4$',
        'CF': 'CF$^{+}$',
        'CF2': 'CF${}_{2}^{+}$',
        'CF3': 'CF${}_{3}^{+}$',
        'CH2F': 'CH$_2$F$^{+}$',
        'CHF2': 'CHF${}_{2}^{+}$',
    }

class IonEnergyPairGenerator:
    def run(self, paths):
        ion_set = set()
        energy_set = set()
        for system in paths.keys():
            for ion in paths[system].keys():
                for energy in paths[system][ion].keys():
                    ion_set.add(ion)
                    energy_set.add(energy)
        ion_set = sorted(list(ion_set))
        energy_set = sorted(list(energy_set))
        return ion_set, energy_set

class FigureGenerator:
    def run(self, paths):
        plt.rcParams.update({'font.size': 10, 'font.family': 'arial'})
        iepg = IonEnergyPairGenerator()
        ion_set, energy_set = iepg.run(paths)
        n_system = len(paths.keys())
        n_ion = len(ion_set)
        n_energy = len(energy_set)

        n_row = n_ion * n_system
        n_col = n_energy
        fig, axes = plt.subplots(n_row, n_col, figsize=(7.1 * 1.5, n_row * 1.5))
        ax_dict = {}

        for system_idx, system in enumerate(paths.keys()):
            for ion in paths[system].keys():
                for energy in paths[system][ion].keys():
                    key = (system, ion, energy)
                    ion_index = ion_set.index(ion)
                    energy_index = energy_set.index(energy)
                    row = n_ion * system_idx + ion_index
                    col = energy_index
                    ax_dict[key] = axes[row, col]

        for system_idx, system in enumerate(paths.keys()):
            for ion in ion_set:
                for energy in energy_set:
                    key = (system, ion, energy)
                    if key not in ax_dict:
                        ion_index = ion_set.index(ion)
                        energy_index = energy_set.index(energy)
                        row = n_ion * system_idx + ion_index
                        col = energy_index
                        axes[row, col].set_visible(False)

        return fig, ax_dict

class AxisPlotter:
    def run(self, path_yield, dst, ax, color, decorate_x=False, decorate_y=False):
        x, y = self.truncate_data(path_yield)
        ax.plot(x, y, color=color)
        self.decorate(ax, y, dst,
                      decorate_x=decorate_x,
                      decorate_y=decorate_y,
                      )

    def truncate_data(self, path_yield, truncate=4500):
        data = np.loadtxt(path_yield, skiprows=2)
        norm_factor = 10 / 9000
        x = data[:, 0] * norm_factor
        y = data[:, 2]

        x = x[:truncate]
        y = y[:truncate]
        return x, y

    def decorate(self, ax, etch_yield, dst, decorate_x=False, decorate_y=False):
        if decorate_x:
            # ax.set_xlabel(r"Ion dose ($ \times 10^{16} \mathrm{cm}^{-2}$)")
            pass
        else:
            ax.xaxis.set_ticklabels([])
        if decorate_y:
            # ax.set_ylabel("Etch yield (Si/ion)")
            pass
        else:
            ax.yaxis.set_ticklabels([])

        yield_avg = etch_yield[-1]
        textbox = f"yield = {yield_avg:.3f}"
        if yield_avg > 0.5:
            x_text, y_text, ha, va = 0.95, 0.05, 'right', 'bottom'
        else:
            x_text, y_text, ha, va = 0.95, 0.95, 'right', 'top'

        ax.text(x_text, y_text, textbox, ha=ha, va=va, transform=ax.transAxes)

        ax.set_xlim(0, 5)
        ax.set_ylim(0, 2.0)

        system, ion, energy = dst.split('_')
        system = PARAMS.convert_dict[system]
        ion = PARAMS.convert_dict[ion]
        title = f"{system}, {ion}, {energy}"
        ax.set_title(title, fontsize=10)

class EtchYieldPlotterTotal():
    def run(self, paths):
        fg = FigureGenerator()
        fig, ax_dict = fg.run(paths)
        ap = AxisPlotter()
        for system in paths.keys():
            color = PARAMS.colors.get(system, 'black')
            for ion in paths[system].keys():
                for energy, file in paths[system][ion].items():
                    if (system, ion) == ('Si3N4', 'CHF2'):
                        decorate_x = True
                    else:
                        decorate_x = False

                    if (system, energy) == ('SiO2', 100) \
                       or (system, energy) == ('Si3N4', 250):
                        decorate_y = True
                    else:
                        decorate_y = False

                    key = (system, ion, energy)
                    ax = ax_dict[key]
                    ap.run(file, f"{system}_{ion}_{energy}eV", ax, color,
                           decorate_x=decorate_x,
                           decorate_y=decorate_y)
        self.save(fig)

    def save(self, fig):
        fig.tight_layout()
        fig.subplots_adjust(left=0.05, bottom=0.05)
        fig.text(0.02, 0.5, "Etch yield (Si/ion)", va='center',
                 rotation='vertical', fontsize=14)
        fig.text(0.5, 0.02, r"Ion dose ($ \times 10^{16} \mathrm{cm}^{-2}$)",
                 ha='center', fontsize=14)
        fig.savefig('result.png')
        fig.savefig('result.pdf')

def main():
    if len(sys.argv) != 2:
        print("Usage: python etchyield.py <path.yaml>")
        sys.exit(1)
    path_yaml = sys.argv[1]
    with open(path_yaml, 'r') as f:
        paths = yaml.safe_load(f)
    eypt = EtchYieldPlotterTotal()
    eypt.run(paths)


if __name__ == "__main__":
    main()
