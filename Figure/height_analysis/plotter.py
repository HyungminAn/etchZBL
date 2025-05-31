from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from params import PARAMS

class FigureGenerator:
    def run(self, data):
        '''
        Generate the figure and axes for the plot.
        '''
        plt.rcParams.update({'font.family': 'Arial'})

        n_row, n_col = self.get_figure_size(data)
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 3, n_row * 3),)
        ax_dict = self.build_ax_dict(data, fig, axes)

        return fig, ax_dict

    def get_figure_size(self, data):
        set_ion, set_energy = self.get_ion_energy_set(data)
        n_ion = len(set_ion)
        n_energy = len(set_energy)
        return n_ion, n_energy

    def get_ion_energy_set(self, data):
        set_ion, set_energy = set(), set()
        for system in data.keys():
            ion, energy = system.split('_')
            set_ion.add(ion)
            set_energy.add(energy)
        set_ion = sorted(list(set_ion))
        set_energy = sorted(list(set_energy), key=lambda x: int(x))
        return set_ion, set_energy

    def build_ax_dict(self, data, fig, axes):
        set_ion, set_energy = self.get_ion_energy_set(data)
        ax_dict = {}
        for system in data.keys():
            ion, energy = system.split('_')
            ax_dict[system] = axes[set_ion.index(ion), set_energy.index(energy)]

        for ion in set_ion:
            for energy in set_energy:
                key = f'{ion}_{energy}'
                if key not in ax_dict:
                    fig.delaxes(axes[set_ion.index(ion), set_energy.index(energy)])

        return ax_dict

class BaseAxisProcessor(ABC):
    def __init__(self, system, data, ax):
        self.system = system
        self.data = data
        self.ax = ax

    def run(self):
        self.normalize()
        self.plot()
        self.decorate()

    def normalize(self):
        self.normalize_x()

    def plot(self):
        pass

    def decorate(self):
        pass

    def normalize_x(self):
        x, _, _ = self.data
        x = x.copy()
        x *= PARAMS.CONVERT.CONV_FACTOR_TO_CM2
        self.data = (x, *self.data[1:])
        self.ax.set_xlim(0, 1.0)
        self.ax.set_xlabel('Ion dose (10$^{17}$ cm$^{-2}$)')

class AxisProcessorHeight(BaseAxisProcessor):
    def normalize(self):
        self.normalize_x()
        self.normalize_y()

    def normalize_y(self):
        _, y, _ = self.data
        y = y.copy()
        y *= PARAMS.CONVERT.ANGST_TO_NM
        y -= y[0]
        self.data = (self.data[0], y, self.data[2])

    def plot(self):
        '''
        Plot the height change.
        '''
        x, y, _ = self.data
        color = PARAMS.PLOT.COLORS.COLORS.get(self.system, PARAMS.PLOT.COLORS.COLOR_LIST['default'])
        self.ax.plot(x, y, 'o-', markersize=2, color=color, alpha=0.5)
        print(f'{self.system}: Etched thickness {np.min(y)}')

    def decorate(self):
        ax = self.ax

        ion, energy = self.system.split('_')
        title = f'{PARAMS.CONVERT.ION_CONVERT_DICT[ion]}, {energy} eV'
        ax.set_title(title)

        ax_color = PARAMS.PLOT.COLORS.COLORS.get(self.system, PARAMS.PLOT.COLORS.COLOR_LIST['default'])
        self.ax.set_ylabel('Height change (nm)', color=ax_color)

        ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)

        ax.tick_params(axis='y', colors=ax_color)

class AxisProcessorCarbon(BaseAxisProcessor):
    def normalize(self):
        self.normalize_x()
        self.normalize_y()

    def normalize_y(self):
        x, y, labels = self.data
        y = y.copy()
        y *= PARAMS.CONVERT.ANGST_TO_NM
        self.data = (x, y, labels)
        self.ax.set_ylabel('Carbon film thickness (nm)')

    def decorate(self):
        self.ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)

    def plot(self):
        '''
        Plot the carbon film thickness.
        '''
        x, y, _ = self.data
        x_front = x[x < PARAMS.PLOT.HEIGHT.TRUNCATE_INITIAL_REGION]
        y_front = y[x < PARAMS.PLOT.HEIGHT.TRUNCATE_INITIAL_REGION]
        x_back = x[x >= PARAMS.PLOT.HEIGHT.TRUNCATE_INITIAL_REGION]
        y_back = y[x >= PARAMS.PLOT.HEIGHT.TRUNCATE_INITIAL_REGION]

        self.ax.plot(x_front, y_front,
                       '--', markersize=2, color='black', alpha=0.2)
        self.ax.plot(x_back, y_back,
                       '-', markersize=2, color='black', alpha=0.7)

class AxisProcessorHeightAndCarbonCombined(BaseAxisProcessor):
    def __init__(self, system, data, ax):
        self.system = system
        self.data = data
        self.ax = ax, ax.twinx()

    def run(self):
        ax_change, ax_carbon = self.ax
        p_height = AxisProcessorHeight(self.system, self.data['height_change'], ax_change)
        p_height.run()
        p_carbon = AxisProcessorCarbon(self.system, self.data['carbon_thickness'], ax_carbon)
        p_carbon.run()

class BatchAxisProcessor():
    def __init__(self, data, ax_dict, processorClass, ylim=None):
        self.data = data
        self.ax_dict = ax_dict
        self.processorClass = processorClass
        self.ylim = ylim

    def run(self):
        processors = {}
        for system, ax in self.ax_dict.items():
            plot_data = self.data[system]
            ps = self.processorClass(system, plot_data, ax)
            processors[system] = ps
            ps.run()
        self.set_ylim(processors)

    def set_ylim(self, processors):
        '''
        Set the y-limits of the axes to be the same for all systems.
        '''
        if self.ylim is not None:
            y_min, y_max = self.ylim
        else:
            y_min, y_max = self.get_ylim(processors)

        for processor in processors.values():
            if not isinstance(processor.ax, tuple):
                processor.ax.set_ylim(y_min, y_max)
            else:
                for ax in processor.ax:
                    ax.set_ylim(y_min, y_max)

    def get_ylim(self, processors):
        y_mins, y_maxs = [], []
        for processor in processors.values():
            axes = processor.ax
            if isinstance(axes, tuple):
                for ax in axes:
                    y_min, y_max = ax.get_ylim()
                    y_mins.append(y_min)
                    y_maxs.append(y_max)
            else:
                y_min, y_max = axes.get_ylim()
                y_mins.append(y_min)
                y_maxs.append(y_max)

        y_min, y_max = np.min(y_mins), np.max(y_maxs)
        return y_min, y_max

class DataPlotter:
    def run(self, data):
        fig, ax_dict = FigureGenerator().run(data)
        batch_processor = BatchAxisProcessor(data, ax_dict, AxisProcessorHeightAndCarbonCombined)
        batch_processor.run()
        self.save_figure(fig)

    def save_figure(self, fig):
        '''
        Save the figure in different formats.
        '''
        fig.tight_layout()
        fig.savefig('result.png', dpi=200)
        fig.savefig('result.pdf')
        fig.savefig('result.eps')

class AxisProcessorMixed(BaseAxisProcessor):
    def plot(self):
        '''
        Plot the mixed layer thickness.
        '''
        x, y, _ = self.data
        y_mixed, y_film = y[:, 0], y[:, 1]
        colors = [PARAMS.PLOT.COLORS.COLOR_LIST['layer']['mixed'],
                 PARAMS.PLOT.COLORS.COLOR_LIST['layer']['film']]
        self.ax.stackplot(x, y_mixed, y_film, colors=colors,)

    def decorate(self):
        ax = self.ax
        ax.set_ylabel('Mixed layer thickness (nm)')
        ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)

class AxisProcessorDensity(BaseAxisProcessor):
    def plot(self):
        x, y, _ = self.data
        y_mixed, y_film = y[:, 2], y[:, 3]
        self.ax.plot(x, y_mixed,
                color= PARAMS.PLOT.COLORS.COLOR_LIST['density']['mixed'],
                label='Mixed layer')
        self.ax.plot(x, y_film,
                color= PARAMS.PLOT.COLORS.COLOR_LIST['density']['film'],
                label='Film layer')

    def decorate(self):
        ax = self.ax
        ax.set_ylabel('Density (g/cm$^3$)')
        ax.set_xlabel('Ion dose (10$^{17}$ cm$^{-2}$)')
        ax.legend(loc='lower right')

class AxisProcessorFCRatio(BaseAxisProcessor):
    def plot(self):
        x, y, _ = self.data
        y_mixed, y_film = y[:, 4], y[:, 5]
        self.ax.plot(x, y_mixed,
                color=PARAMS.PLOT.COLORS.COLOR_LIST['fc_ratio']['mixed'],
                label='Mixed layer')
        self.ax.plot(x, y_film,
                color=PARAMS.PLOT.COLORS.COLOR_LIST['fc_ratio']['film'],
                label='Film layer')

    def decorate(self):
        ax = self.ax
        ax.set_ylabel('FC ratio')
        ax.legend(loc='upper left')

class AxisProcessorSPXRatioMixedLayer(BaseAxisProcessor):
    def plot(self):
        x, y, _ = self.data
        y_sp3, y_sp2, y_sp, y_others = y[:, 6:10].T
        ax = self.ax
        color_dict = PARAMS.PLOT.COLORS.COLOR_LIST['spx_ratio']

        ax.plot(x, y_sp3,
                color=color_dict['sp3'], label='sp3 (mixed)')
        ax.plot(x, y_sp2,
                color=color_dict['sp2'], label='sp2 (mixed)')
        ax.plot(x, y_sp,
                color=color_dict['sp'], label='sp (mixed)')
        ax.plot(x, y_others,
                color=color_dict['others'], label='others (mixed)')

    def normalize(self):
        self.normalize_x()
        self.normalize_y()

    def normalize_y(self):
        x, y, labels = self.data
        y = y.copy()

        y_sp3, y_sp2, y_sp, y_others = y[:, 6:10].T
        y_total = y_sp3 + y_sp2 + y_sp + y_others
        y_sp3 /= y_total
        y_sp2 /= y_total
        y_sp /= y_total
        y_others /= y_total
        y[:, 6:10] = np.column_stack((y_sp3, y_sp2, y_sp, y_others))

        self.data = (x, y, labels)

    def decorate(self):
        ax = self.ax
        ax.set_ylabel('Ratio (mixed layer)')
        # ax.legend(loc='upper left')

class AxisProcessorSPXRatioFilmLayer(BaseAxisProcessor):
    def plot(self):
        x, y, _ = self.data
        y_sp3, y_sp2, y_sp, y_others = y[:, 10:14].T
        ax = self.ax
        color_dict = PARAMS.PLOT.COLORS.COLOR_LIST['spx_ratio']

        ax.plot(x, y_sp3, linestyle='--',
                color=color_dict['sp3'], label='sp3 (film)')
        ax.plot(x, y_sp2, linestyle='--',
                color=color_dict['sp2'], label='sp2 (film)')
        ax.plot(x, y_sp, linestyle='--',
                color=color_dict['sp'], label='sp (film)')
        ax.plot(x, y_others, linestyle='--',
                color=color_dict['others'], label='others (film)')

    def normalize(self):
        self.normalize_x()
        self.normalize_y()

    def normalize_y(self):
        x, y, labels = self.data
        y = y.copy()

        y_sp3, y_sp2, y_sp, y_others = y[:, 10:14].T
        y_total = y_sp3 + y_sp2 + y_sp + y_others
        y_sp3 /= y_total
        y_sp2 /= y_total
        y_sp /= y_total
        y_others /= y_total
        y[:, 10:14] = np.column_stack((y_sp3, y_sp2, y_sp, y_others))

        self.data = (x, y, labels)

    def decorate(self):
        ax = self.ax
        ax.set_ylabel('Ratio (film layer)')
        # ax.legend(loc='upper left')

class AxisProcessorNeighbor(BaseAxisProcessor):
    def plot(self):
        x, y, labels = self.data
        for y_label, label in zip(y.T, labels[1:]):
            self.ax.plot(x, y_label, label=label, alpha=0.5,
                    color=PARAMS.PLOT.COLORS.COLOR_LIST['neighbor'].get(label, 'grey'))

    def decorate(self):
        ax = self.ax
        ax.set_ylabel('count')
        ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend(loc='upper left')

class FigureGeneratorSelected(FigureGenerator):
    def __init__(self, axis_config):
        self.axis_config = axis_config

    def run(self, data):
        plt.rcParams.update({'font.family': 'Arial'})

        n_row, n_col = self.get_figure_size(data)
        fig, axes = plt.subplots(n_row, n_col, figsize=(3.5*n_col, 2.5*n_row),)
        ax_dict = self.build_ax_dict(data, fig, axes)
        return fig, ax_dict

    def get_figure_size(self, data):
        n_row, n_col = len(self.axis_config), len(data)
        return n_row, n_col

    def build_ax_dict(self, data, fig, axes):
        ax_dict = {}
        for idx, system in enumerate(data.keys()):
            ax_dict[system] = axes[:, idx]
        return ax_dict

class DataPlotterSelected(DataPlotter):
    def __init__(self):
        self.axis_config = [
            ('HeightAndCarbonCombined', AxisProcessorHeightAndCarbonCombined),
            ('Mixed layer thickness', AxisProcessorMixed),
            ('Density', AxisProcessorDensity),
            ('FC ratio', AxisProcessorFCRatio),
            ('spx ratio (mixed)', AxisProcessorSPXRatioMixedLayer),
            ('spx ratio (film)', AxisProcessorSPXRatioFilmLayer),
            ('Carbon neighbor classification', AxisProcessorNeighbor),
        ]

    def reconfigure_data(self, data, ax_dict):
        result = {}
        for idx, (key, processorClass) in enumerate(self.axis_config):
            data_selected = {system: {} for system in data.keys()}
            if key == 'HeightAndCarbonCombined':
                for system in data.keys():
                    data_selected[system]['height_change'] = data[system]['height_change']
                    data_selected[system]['carbon_thickness'] = data[system]['carbon_thickness']
            elif key == 'Carbon neighbor classification':
                for system in data.keys():
                    data_selected[system] = data[system]['neighbor_classification']
            else:
                for system in data.keys():
                    data_selected[system] = data[system]['film_data']
            print(f'Plotting {key} for {len(data_selected)} systems')

            ylim = None
            if key == 'FC ratio':
                ylim = (0, 2)
            elif key == 'spx ratio (mixed)' or key == 'spx ratio (film)':
                ylim = (0, 1)

            ax_dict_selected = {system: ax_dict[system][idx] for system in data.keys()}
            result[key] = BatchAxisProcessor(data_selected, ax_dict_selected,
                                             processorClass, ylim=ylim)
        return result

    def run(self, data):
        fig, ax_dict = FigureGeneratorSelected(self.axis_config).run(data)
        batch_processor_dict = self.reconfigure_data(data, ax_dict)
        for bp in batch_processor_dict.values():
            bp.run()
        self.save_figure(fig)
