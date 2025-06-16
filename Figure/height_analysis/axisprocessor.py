from abc import ABC, abstractmethod
import numpy as np
from params import PARAMS

_processor_registry = {}

def register_processor(key):
    def decorator(cls):
        _processor_registry[key] = cls
        return cls
    return decorator

def get_processor(key):
    return _processor_registry.get(key)

class BaseAxisProcessor(ABC):
    def __init__(self, system, data, ax, skip_decorate=False):
        self.system = system
        self.data = data
        self.ax = ax
        self.skip_decorate = skip_decorate

    def run(self):
        self.normalize()
        self.plot()
        if not self.skip_decorate:
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
        if not self.skip_decorate:
            self.ax.set_xlabel(r'Ion dose ($\times$ 10$^{17}$ cm$^{-2}$)')

    def smooth_array(self, arr, window_size=9):
        if window_size < 1:
            raise ValueError("Window size must be at least 1.")
        padded = np.pad(arr, pad_width=window_size, mode='edge')
        kernel = np.ones(2 * window_size + 1) / (2 * window_size + 1)
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed

@register_processor('height')
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
        # color = PARAMS.PLOT.COLORS.COLORS.get(self.system, PARAMS.PLOT.COLORS.COLOR_LIST['default'])
        color = PARAMS.PLOT.COLORS.COLORS_Si3N4.get(self.system, PARAMS.PLOT.COLORS.COLOR_LIST['default'])
        self.ax.plot(x, y, 'o-', markersize=2, color=color, alpha=0.5)
        self.ax.axhline(0, color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
        print(f'{self.system}: Etched thickness {np.min(y):.2f} nm')

    def decorate(self):
        ax = self.ax

        ion, energy = self.system.split('_')
        title = f'{PARAMS.CONVERT.ION_CONVERT_DICT[ion]}, {energy} eV'
        ax.set_title(title)

        ax_color = PARAMS.PLOT.COLORS.COLORS.get(self.system, PARAMS.PLOT.COLORS.COLOR_LIST['default'])
        self.ax.set_ylabel('Height change (nm)', color=ax_color)
        ax.tick_params(axis='y', colors=ax_color)

@register_processor('carbon_thickness')
class AxisProcessorCarbonThickness(BaseAxisProcessor):
    def normalize(self):
        self.normalize_x()
        self.normalize_y()

    def plot(self):
        x, y, _ = self.data
        y_film = y[:, 2]
        # color = PARAMS.PLOT.COLORS.COLOR_LIST['layer']['film']
        color = 'black'
        # self.ax.plot(x, y_film, color=color)
        self.ax.fill_between(x, 0, y_film, color=color, alpha=0.5)
        self.ax.set_yticklabels([])

    def normalize_y(self):
        x, y, labels = self.data
        y = y.copy()
        y[:, 2] *= PARAMS.CONVERT.ANGST_TO_NM
        self.data = (x, y, labels)

    def decorate(self):
        ax = self.ax
        ax.set_ylabel('Carbon film thickness (nm)')
        ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)

class AxisProcessorEtchedAmount(BaseAxisProcessor):
    def normalize(self):
        self.normalize_x()
        self.normalize_y()

    def plot(self):
        x, y, _ = self.data
        color = 'black'
        self.ax.plot(x, y, color=color)

    def normalize_y(self):
        x, y, labels = self.data
        y = y.copy()
        y *= PARAMS.CONVERT.ANGST_TO_NM
        self.data = (x, y, labels)

    def decorate(self):
        ax = self.ax
        ax.set_ylabel('Etched amount (nm)')
        ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)

@register_processor('mixedfilmstacked')
class AxisProcessorMixedFilmStacked(BaseAxisProcessor):
    def plot(self):
        '''
        Plot the mixed layer thickness.
        '''
        x, y, _ = self.data
        y_mixed, y_film = y[:, 0], y[:, 1]
        y_mixed = self.smooth_array(y_mixed, window_size=3)
        y_film = self.smooth_array(y_film, window_size=3)
        colors = [PARAMS.PLOT.COLORS.COLOR_LIST['layer']['mixed'],
                 PARAMS.PLOT.COLORS.COLOR_LIST['layer']['film']]
        labels = ['Mixed layer', 'Film layer']
        self.ax.stackplot(x, y_mixed, y_film, colors=colors, labels=labels)

    def normalize(self):
        self.normalize_x()
        self.normalize_y()

    def normalize_y(self):
        x, y, labels = self.data
        y = y.copy()
        y[:, 0] *= PARAMS.CONVERT.ANGST_TO_NM
        y[:, 1] *= PARAMS.CONVERT.ANGST_TO_NM
        self.data = (x, y, labels)

    def decorate(self):
        ax = self.ax
        ax.set_ylabel('Thickness\n(nm)')
        ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.3)

@register_processor('density')
class AxisProcessorDensity(BaseAxisProcessor):
    def plot(self):
        x, y, _ = self.data
        self.ax.plot(x, y)

    def decorate(self):
        ax = self.ax
        ax.set_ylabel('Density (g/cm$^3$)')

class AxisProcessorDensityMixed(AxisProcessorDensity):
    def plot(self):
        x, y, _ = self.data
        label = 'Mixed layer'
        color = PARAMS.PLOT.COLORS.COLOR_LIST['layer']['mixed']
        self.ax.plot(x, y, label=label, color=color)

class AxisProcessorDensityFilm(AxisProcessorDensity):
    def plot(self):
        x, y, _ = self.data
        label = 'Film layer'
        color = PARAMS.PLOT.COLORS.COLOR_LIST['layer']['film']
        self.ax.plot(x, y, label=label, color=color)

@register_processor('fc_ratio')
class AxisProcessorFCRatio(BaseAxisProcessor):
    def plot(self):
        x, y, _ = self.data
        self.ax.plot(x, y)

    def decorate(self):
        ax = self.ax
        ax.set_ylabel('FC ratio')

class AxisProcessorFCRatioMixed(AxisProcessorFCRatio):
    def plot(self):
        x, y, _ = self.data
        label = 'Mixed layer'
        color = PARAMS.PLOT.COLORS.COLOR_LIST['layer']['mixed']
        self.ax.plot(x, y, label=label, color=color)

class AxisProcessorFCRatioFilm(AxisProcessorFCRatio):
    def plot(self):
        x, y, _ = self.data
        label = 'Film layer'
        color = PARAMS.PLOT.COLORS.COLOR_LIST['layer']['film']
        self.ax.plot(x, y, label=label, color=color)

class AxisProcessorSiCcount(BaseAxisProcessor):
    def plot(self):
        x, y, _ = self.data
        label = 'Mixed layer'
        color = 'purple'
        self.ax.plot(x, y, label=label, color=color)

    def decorate(self):
        ax = self.ax
        ax.set_ylabel('Si + C')
        ax.set_ylim(2, 4)

class AxisProcessorAtomCount(BaseAxisProcessor):
    def plot(self):
        x, y, label = self.data
        label = label[1:]
        for y_label, label in zip(y.T, label):
            y_label = self.smooth_array(y_label)
            color = PARAMS.PLOT.COLORS.COLOR_LIST['atom_count'].get(label, 'grey')
            self.ax.plot(x, y_label, label=label, color=color, linewidth=1)

class AxisProcessorAtomCountRatio(AxisProcessorAtomCount):
    def decorate(self):
        ax = self.ax
        ax.set_ylabel('Ratio')
        ax.set_ylim(0, 0.7)

class AxisProcessorAtomCountNumberDensity(AxisProcessorAtomCount):
    def decorate(self):
        ax = self.ax
        ax.set_ylabel('Number density\n(nm $^{-3}$)')
        ax.set_ylim(0, None)

@register_processor('spx_ratio')
class AxisProcessorSPXRatio(BaseAxisProcessor):
    def plot(self):
        x, y, _ = self.data
        y_sp3, y_sp2, y_sp, _ = y.T
        ax = self.ax
        color_dict = PARAMS.PLOT.COLORS.COLOR_LIST['spx_ratio']

        ax.plot(x, y_sp3, color=color_dict['sp3'], label='sp$^3$')
        ax.plot(x, y_sp2, color=color_dict['sp2'], label='sp$^2$')
        ax.plot(x, y_sp, color=color_dict['sp'], label='sp')

    def normalize(self):
        self.normalize_x()
        self.normalize_y()

    def normalize_y(self):
        x, y, labels = self.data
        y = y.copy()

        y_sp3, y_sp2, y_sp, y_others = y.T
        y_total = y_sp3 + y_sp2 + y_sp + y_others
        y_sp3 /= y_total
        y_sp2 /= y_total
        y_sp /= y_total
        y_others /= y_total
        y_new = np.column_stack((y_sp3, y_sp2, y_sp, y_others))

        self.data = (x, y_new, labels)

    def decorate(self):
        ax = self.ax
        ax.set_ylabel('Ratio')

@register_processor('neighbor')
class AxisProcessorNeighbor(BaseAxisProcessor):
    def plot(self):
        x, y, labels = self.data
        label_convert_dict = {
                'Fluorocarbon': 'FC',
                'SiC_cluster': 'SiC',
                }
        for y_label, label in zip(y.T, labels[1:]):
            self.ax.plot(x, y_label, label=label_convert_dict.get(label, label), alpha=0.5,
                    color=PARAMS.PLOT.COLORS.COLOR_LIST['neighbor'].get(label, 'grey'))

    def decorate(self):
        ax = self.ax
        ax.set_ylabel('count')
        ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)

class BatchAxisProcessor:
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

class CombinedAxisProcessor(BaseAxisProcessor):
    '''
    A processor that combines two axes, typically for height and carbon data.
    Usage:
        processor = CombinedAxisProcessor(ax,
            (data1, AxisProcessorClass_1, False), --> goes to left
            (data2, AxisProcessorClass_2, True), --> goes to right
            ...
            )
    '''
    def __init__(self, system, data_class_pair, ax):
        self.system = system
        self.data_class_pair = data_class_pair
        self.ax = (ax, ax.twinx())

    def run(self):
        ax_left, ax_right = self.ax
        for (data, processorClass, twin) in self.data_class_pair:
            if twin:
                ax = ax_right
            else:
                ax = ax_left
            processor = processorClass(self.system, data, ax, skip_decorate=True)
            processor.run()

        ion, energy = self.system.split('_')
        title = f'{PARAMS.CONVERT.ION_CONVERT_DICT[ion]}, {energy} eV'
        ax_left.set_title(title)
