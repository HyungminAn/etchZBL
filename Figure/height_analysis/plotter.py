import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from params import PARAMS
from axisprocessor import BatchAxisProcessor
from axisprocessor import CombinedAxisProcessor
from axisprocessor import AxisProcessorHeight
from axisprocessor import AxisProcessorCarbonThickness
from axisprocessor import AxisProcessorMixedFilmStacked
# from axisprocessor import AxisProcessorDensity
from axisprocessor import AxisProcessorFCRatioMixed
from axisprocessor import AxisProcessorSPXRatio
# from axisprocessor import AxisProcessorNeighbor
# from axisprocessor import AxisProcessorSiCcount
from axisprocessor import AxisProcessorAtomCountRatio
from axisprocessor import AxisProcessorAtomCountNumberDensity

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

    def run(self, data, squeeze=False):
        '''
        Generate the figure and axes for the plot.
        '''

        n_row, n_col = self.get_figure_size(data, squeeze=squeeze)
        fig, axes = plt.subplots(n_row, n_col, figsize=(7.1, 7.1 * 0.9),)

        ax_dict = self.build_ax_dict(data, fig, axes, squeeze=squeeze)

        return fig, ax_dict

    def get_figure_size(self, data, squeeze=False):
        if squeeze:
            count_dict = self.split_ion_energy(data)
            n_ion = len(count_dict)
            n_energy = max(len(energies) for energies in count_dict.values())
            return n_ion, n_energy
        else:
            set_ion, set_energy = self.get_ion_energy_set(data)
            n_ion = len(set_ion)
            n_energy = len(set_energy)
            return n_ion, n_energy

    def split_ion_energy(self, data):
        result = {}
        for key in data.keys():
            ion, energy = key.split('_')
            if ion not in result:
                result[ion] = []
            result[ion].append(energy)
        return result

    def get_ion_energy_set(self, data):
        set_ion, set_energy = set(), set()
        for system in data.keys():
            ion, energy = system.split('_')
            set_ion.add(ion)
            set_energy.add(energy)
        set_ion = sorted(list(set_ion))
        set_energy = sorted(list(set_energy), key=lambda x: int(x))
        return set_ion, set_energy

    def build_ax_dict(self, data, fig, axes, squeeze=False):
        if squeeze:
            count_dict = self.split_ion_energy(data)
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
            set_ion, set_energy = self.get_ion_energy_set(data)
            ax_dict = {}
            for system in data.keys():
                ion, energy = system.split('_')
                idx, jdx = set_ion.index(ion), set_energy.index(energy)
                ax_dict[system] = axes[idx, jdx]

            for ion in set_ion:
                for energy in set_energy:
                    key = f'{ion}_{energy}'
                    if key not in ax_dict:
                        fig.delaxes(axes[set_ion.index(ion), set_energy.index(energy)])

        return ax_dict

class DataPlotter:
    def run(self, data, system, ylim):
        fig, ax_dict = FigureGenerator().run(data, squeeze=True)
        data = self.reconfigure_data(data)
        batch_processor = BatchAxisProcessor(data, ax_dict,
                                             CombinedAxisProcessor, ylim=ylim)
        batch_processor.run()
        self.decorate(fig)
        self.save_figure(fig, system)

    def decorate(self, fig):

        # def xticks_formatter(x, _):
        #     return str(int(x)) if x == int(x) else str(x)
        # for ax in axes:
        #     ax.xaxis.set_major_formatter(FuncFormatter(xticks_formatter))

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.1, left=0.1, right=0.95, wspace=0.15)
        fig.text(0.5, 0.02, r'Ion dose ($\times$ 10$^{17}$ cm$^{-2}$)', ha='center')
        fig.text(0.02, 0.5, r'Surface height change (nm)', rotation='vertical', va='center')
        fig.text(0.98, 0.5, r'Carbon film thickness (nm)', rotation='vertical', va='center')

    def save_figure(self, fig, system):
        '''
        Save the figure in different formats.
        '''
        name = f'3_2_1_height_total_{system}'
        fig.savefig(f'{name}.png')
        fig.savefig(f'{name}.pdf')
        fig.savefig(f'{name}.eps')

    def reconfigure_data(self, data):
        result = {}
        for system, data_system in data.items():
            result[system] = (
                (data_system['height_change'], AxisProcessorHeight, False),
                (data_system['z_film'], AxisProcessorCarbonThickness, True),
                )
        return result

class FigureGeneratorSelected(FigureGenerator):
    def __init__(self, axis_config):
        super().__init__()
        self.axis_config = axis_config

    def run(self, data, squeeze=False):
        n_row, n_col = self.get_figure_size(data)
        n_multi = 1
        figsize = (3.5 * n_multi, 3.5 * n_multi * n_row / n_col)
        fig, axes = plt.subplots(n_row, n_col, figsize=figsize,)
        ax_dict = self.build_ax_dict(data, fig, axes)
        return fig, ax_dict

    def get_figure_size(self, data, squeeze=False):
        n_row, n_col = len(self.axis_config), len(data)
        return n_row, n_col

    def build_ax_dict(self, data, fig, axes, squeeze=False):
        ax_dict = {}
        for idx, system in enumerate(data.keys()):
            ax_dict[system] = axes[:, idx]
        return ax_dict

class DataPlotterSelected(DataPlotter):
    def __init__(self):
        self.axis_config = [
            # ('fc_ratio_mixed', AxisProcessorFCRatioMixed),
            # ('spx_ratio_mixed', AxisProcessorSPXRatio),
            ('stacked', AxisProcessorMixedFilmStacked),
            ('atomcount_mixed', AxisProcessorAtomCountNumberDensity),
            # ('atomcount_mixed_norm', AxisProcessorAtomCountRatio),
            # ('atomcount_film', AxisProcessorAtomCount),
            # ('atomcount_film_norm', AxisProcessorAtomCount),
            # ('h_effect_mixed', AxisProcessorSiCcount),
            # ('neighbor_classification', AxisProcessorNeighbor),

            # ('Density', AxisProcessorDensity),
            # ('spx ratio (film)', AxisProcessorSPXRatioFilmLayer),
        ]

    def run(self, data):
        fig, ax_dict = FigureGeneratorSelected(self.axis_config).run(data)
        batch_processor_dict = self.reconfigure_data(data, ax_dict)
        for bp in batch_processor_dict.values():
            bp.run()
        self.decorate(fig, ax_dict)
        self.save_figure(fig)

    def reconfigure_data(self, data, ax_dict):
        result = {}
        for idx, (key, processorClass) in enumerate(self.axis_config):
            data_selected = {system: {} for system in data.keys()}
            for system in data.keys():
                data_selected[system] = data[system][key]
            print(f'Plotting {key} for {len(data_selected)} systems')

            if key == 'fc_ratio_mixed':
                ylim = (1, 4)
            elif key == 'spx_ratio_mixed':
                ylim = (0, 1)
            else:
                ylim = None

            ax_dict_selected = {system: ax_dict[system][idx] for system in data.keys()}
            result[key] = BatchAxisProcessor(data_selected,
                                             ax_dict_selected,
                                             processorClass,
                                             ylim=ylim)
        return result

    def decorate(self, fig, ax_dict):

        def xticks_formatter(x, _):
            return str(int(x)) if x == int(x) else str(x)

        for idx, (system, axes) in enumerate(ax_dict.items()):
            # turn off y-tick labels for all but the first column
            if idx > 0:
                for ax in axes:
                    ax.set_yticklabels([])
                    ax.yaxis.label.set_visible(False)

            # turn off x-tick labels for all but the last row
            for ax_idx, ax in enumerate(axes):
                if ax_idx < len(axes) - 1:
                    ax.set_xticklabels([])
                else:
                    ax.xaxis.set_major_formatter(FuncFormatter(xticks_formatter))
                ax.xaxis.label.set_visible(False)

                if ax_idx == 0:
                    ion, energy = system.split('_')
                    title = f'{PARAMS.CONVERT.ION_CONVERT_DICT[ion]}, {energy} eV'
                    ax.set_title(title, fontsize=10)

        axes = [ax for ax in ax_dict.values()]
        axes = np.array(axes, dtype=object).T

        n_row, n_col = axes.shape
        handles, labels = {}, {}
        for row in range(n_row):
            handles[row], labels[row] = [], []
            for col in range(n_col):
                ax = axes[row, col]
                handle, label = ax.get_legend_handles_labels()
                for h, l in zip(handle, label):
                    if l not in labels[row]:
                        handles[row].append(h)
                        labels[row].append(l)

        pos_dict = {
            0: (0.6, 0.53),
            1: (0.6, 0.0),
            # 2: (0.5, 0.05),
            # 2: None,
        }

        for row in range(n_row):
            h, l = handles[row], labels[row]
            ncol = len(l) if len(l) < 3 else 3
            if pos_dict[row] is None:
                continue
            fig.legend(h, l,
                       loc='lower center',
                       ncol=ncol,
                       bbox_to_anchor=pos_dict[row],
                       frameon=False,
                       )

        fig.text(0.6, 0.16, r'Ion dose ($\times$ 10$^{17}$ cm$^{-2}$)', ha='center')

    def save_figure(self, fig):
        name = '3_2_1_CF_CH2F_compare_plot'
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.3, wspace=0.2, hspace=0.3)
        fig.savefig(f'{name}.png', dpi=200)
        fig.savefig(f'{name}.pdf')
        fig.savefig(f'{name}.eps')
