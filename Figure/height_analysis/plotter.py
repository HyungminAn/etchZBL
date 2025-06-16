import numpy as np
from matplotlib.ticker import FuncFormatter

from params import PARAMS
from figuregenerator import FigureGenerator
from figuregenerator import FigureGeneratorSelected
from axisprocessor import BatchAxisProcessor
from axisprocessor import CombinedAxisProcessor
from axisprocessor import AxisProcessorMixedFilmStacked
from axisprocessor import AxisProcessorAtomCountNumberDensity
from axisprocessor import AxisProcessorEtchedAmount

class DataReconfigurer:
    def run(self, data):
        result = {}
        for system, data_system in data.items():
            # result[system] = (
            #     (data_system['height_change'], AxisProcessorHeight, False),
            #     (data_system['z_film'], AxisProcessorCarbonThickness, True),
            #     )
            result[system] = (
                (data_system['etchedamount'], AxisProcessorEtchedAmount, False),
                )
        return result

class DataReconfigurerSelected:
    def run(self, data, ax_dict, axis_config):
        result = {}
        for idx, (key, processorClass) in enumerate(axis_config):
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

class DataPlotter:
    def __init__(self, system, ylim):
        self.system = system
        self.ylim = ylim

    def run(self, data):
        fig, ax_dict = FigureGenerator().run(data, squeeze=True)
        drc = DataReconfigurer()
        data = drc.run(data)
        batch_processor = BatchAxisProcessor(data,
                                             ax_dict,
                                             CombinedAxisProcessor,
                                             ylim=self.ylim)
        batch_processor.run()
        self.decorate(fig, ax_dict)
        save_name = f'3_2_1_height_total_{self.system}'
        self.save_figure(fig, save_name)

    def decorate(self, fig, ax_dict):
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.1, left=0.1, right=0.95, wspace=0.15)
        fig.text(0.5, 0.02, r'Ion dose ($\times$ 10$^{17}$ cm$^{-2}$)', ha='center')
        fig.text(0.02, 0.5, r'Surface height change (nm)', rotation='vertical', va='center')
        fig.text(0.98, 0.5, r'Carbon film thickness (nm)', rotation='vertical', va='center')

    def save_figure(self, fig, name):
        '''
        Save the figure in different formats.
        '''
        fig.savefig(f'{name}.png')
        fig.savefig(f'{name}.pdf')
        fig.savefig(f'{name}.eps')

class DataPlotterSelected(DataPlotter):
    def __init__(self, system, ylim):
        super().__init__(system, ylim)
        self.axis_config = [
            ('stacked', AxisProcessorMixedFilmStacked),
            ('atomcount_mixed', AxisProcessorAtomCountNumberDensity),
        ]

    def run(self, data):
        fig, ax_dict = FigureGeneratorSelected(data, self.axis_config).run(data)
        drc = DataReconfigurerSelected()
        batch_processor_dict = drc.run(data, ax_dict, self.axis_config)
        for bp in batch_processor_dict.values():
            bp.run()
        self.decorate(fig, ax_dict)
        save_name = f'3_2_1_{self.system}_selected'
        self.save_figure(fig, save_name)

    def decorate(self, fig, ax_dict):
        self.adjust_xyticks(ax_dict)
        self.add_legend(fig, ax_dict)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.3, wspace=0.2, hspace=0.3)

    def adjust_xyticks(self, ax_dict):
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

    def add_legend(self, fig, ax_dict):
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
