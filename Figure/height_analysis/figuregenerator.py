import numpy as np
import matplotlib.pyplot as plt

class FigureGenerator:
    def __init__(self):
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 10,
        })
        self.figsize = (7.1, 7.1 * 0.9)  # Default figure size
        self.fsc = FigureSizeCalculator()
        self.adb = AxisDictBuilder()

    def run(self, data, squeeze=False):
        '''
        Generate the figure and axes for the plot.
        '''
        n_row, n_col = self.fsc.run(data, squeeze=squeeze)
        fig, axes = plt.subplots(n_row, n_col, figsize=self.figsize)
        ax_dict = self.adb.run(data, fig, axes, squeeze=squeeze)
        return fig, ax_dict

class IonEnergySetProvider:
    def run(self, data, return_as_set=False):
        result = {}
        for key in data.keys():
            ion, energy = key.split('_')
            if ion not in result:
                result[ion] = []
            result[ion].append(energy)

        if return_as_set:
            set_ion, set_energy = set(), set()
            for ion, energies in result.items():
                set_ion.add(ion)
                for energy in energies:
                    set_energy.add(energy)
            set_ion = sorted(list(set_ion))
            set_energy = sorted(list(set_energy), key=lambda x: int(x))
            result = (set_ion, set_energy)
            return result

        return result

class FigureSizeCalculator:
    def run(self, data, squeeze=False):
        iesp = IonEnergySetProvider()
        if squeeze:
            count_dict = iesp.run(data)
            n_ion = len(count_dict)
            n_energy = max(len(energies) for energies in count_dict.values())
        else:
            set_ion, set_energy = iesp.run(data, return_as_set=True)
            n_ion = len(set_ion)
            n_energy = len(set_energy)
        return n_ion, n_energy

class FigureSizeCalculatorSelected:
    def __init__(self, axis_config):
        self.axis_config = axis_config

    def run(self, data, squeeze=False):
        n_row, n_col = len(self.axis_config), len(data)
        return n_row, n_col

class AxisDictBuilder:
    def run(self, data, fig, axes, squeeze=False):
        iesp = IonEnergySetProvider()
        if squeeze:
            count_dict = iesp.run(data)
            ax_dict = {}
            for idx, (ion, energies) in enumerate(count_dict.items()):
                for jdx, energy in enumerate(energies):
                    system = f'{ion}_{energy}'
                    ax_dict[system] = axes[idx, jdx]

                    if jdx != 0:
                        axes[idx, jdx].set_yticklabels([])

                    if idx != len(count_dict) - 1:
                        axes[idx, jdx].set_xticklabels([])
                    axes[idx, jdx].margins(x=0.05)
        else:
            set_ion, set_energy = iesp.run(data, return_as_set=True)
            ax_dict = {}
            for system in data.keys():
                ion, energy = system.split('_')
                idx, jdx = set_ion.index(ion), set_energy.index(energy)
                ax_dict[system] = axes[idx, jdx]

            for ion in set_ion:
                for energy in set_energy:
                    key = f'{ion}_{energy}'
                    idx, jdx = set_ion.index(ion), set_energy.index(energy)
                    if key not in ax_dict:
                        fig.delaxes(axes[idx, jdx])

        return ax_dict

class AxisDictBuilderSelected:
    def run(self, data, fig, axes, squeeze=False):
        ax_dict = {}
        for idx, system in enumerate(data.keys()):
            if axes.ndim == 1:
                ax_dict[system] = axes
            else:
                ax_dict[system] = axes[:, idx]

        return ax_dict

class FigureGeneratorSelected(FigureGenerator):
    def __init__(self, data, axis_config):
        super().__init__()
        self.axis_config = axis_config
        self.fsc = FigureSizeCalculatorSelected(axis_config)
        self.adb = AxisDictBuilderSelected()
        self.figsize = (3.5, 3.5)  # Default figure size
