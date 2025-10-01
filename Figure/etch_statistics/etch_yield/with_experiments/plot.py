from dataclasses import dataclass
from itertools import chain

import yaml
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class PlotInfo:
    color_list = {
        'SiO2': 'red',
        'Si3N4': 'red',
    }
    marker_list = {
        'CF3': '^',
        'CF2': '^',
        'CF': '^',
        'CHF2': '^',
        'CH2F': '^',
    }
    species_list = [
        'CF3_',
        'CF2_',
        'CF_',
        'CHF2_',
        'CH2F_',
    ]
    legend_props = {
        'loc': 'lower center',
        'bbox_to_anchor': (0.5, 0),
        'frameon': False,
    }
    scatter_props = {
        's': 30,
        # 'edgecolor': 'k',
        'edgecolor': 'red',
        'alpha': 1,
    }
    scatter_props_ref = {
        's': 30,
        'edgecolor': 'k',
        'color': 'w',
    }
    fit_props = {
        'linestyle': '--',
        'alpha': 1,
    }
    label_map = {
        'SiO2': {
            'sim': {
                'CF3': 'This study',
                'CF2': 'This study',
                'CF': 'This study',
                'CHF2': 'This study',
                # 'CHF2': 'This study (CHF${}_{2}^{+}$)',
                # 'CH2F': 'This study (CH$_2$F${}^{+}$)',
                },
            'exp': {
                'CF3 (ref.1)': 'Karahashi et al.',
                'CF3 (ref.2)': 'Toyoda et al.',
                'CF3 (ref.3)': 'Shibano et al.',
                'CF3 (ref.4)': 'Yamaguchi et al.',

                'CF2 (ref.1)': 'Karahashi et al.',
                'CF2 (ref.3)': 'Shibano et al.',

                'CF (ref.1)': 'Karahashi et al.',
                'CF (ref.3)': 'Shibano et al.',

                'CHF2': 'Ito et al.',
                },
            },
        'Si3N4': {
            'sim': {
                'CF3': 'This study',
                'CF2': 'This study',
                'CF': 'This study',
                'CHF2': 'This study',
                'CH2F': 'This study',
                },
            'exp': {
                'CF2': 'Ito et al.',
                'CHF2': 'Ito et al.',
                },
            },
        }

    scatter_prop_map = {
        'Karahashi et al.': ('black', 'o'),
        'Ito et al.': ('black', 's'),
        'Toyoda et al.': ('black', 'D'),
        'Shibano et al.': ('black', 'x'),
        'Yamaguchi et al.': ('black', '+'),
    }

    key_convert_dict = {
        'CF3_': 'CF$_3$',
        'CF2_': 'CF$_2$',
        'CF_': 'CF',
        'CHF2_': 'CHF$_2$/CH$_2$F',
        'CH2F_': 'CHF$_2$/CH$_2$F',
    }

class DataPlotter:
    def run(self, data_total):
        fig, ax_dict = self.generate_figure()

        for system_type, value_dict in data_total.items():
            for species in PlotInfo.species_list:
                ax = ax_dict[system_type].get(PlotInfo.key_convert_dict[species])
                if ax is None:
                    continue
                data_exp = {k.replace('_', ''): v for k, v in value_dict['exp'].items() if species in k}
                data_sim = {k.replace('_', ''): v for k, v in value_dict['sim'].items() if species in k}

                data_dict = {'exp': data_exp, 'sim': data_sim, 'system': system_type}

                self.plot_ref(ax, data_dict)
                self.plot_points(ax, data_dict)
                self.decorate(ax)

        self.add_subtitles(ax_dict)
        self.set_global_legend(fig, ax_dict)
        self.save(fig)

    def add_subtitles(self, ax_dict):
        shift_x, shift_y = 0.15, 0.2
        text_opts = {
                'ha': 'left',
                'va': 'top',
                }
        texts = [
                ('SiO2',  'CF$_3$',          -shift_x, 1 + shift_y, '(a) SiO$_2$, CF${}_{3}^{+}$'),
                ('SiO2',  'CF$_2$',          -shift_x, 1 + shift_y, '(b) SiO$_2$, CF${}_{2}^{+}$'),
                ('Si3N4', 'CF$_2$',          -shift_x, 1 + shift_y, '(e) Si$_3$N$_4$, CF${}_{2}^{+}$'),
                ('SiO2',  'CF',              -shift_x, 1 + shift_y, '(c) SiO$_2$, CF$^{+}$'),
                ('SiO2',  'CHF$_2$/CH$_2$F', -shift_x, 1 + shift_y, '(d) SiO$_2$, CHF${}_{2}^{+}$'),
                ('Si3N4', 'CHF$_2$/CH$_2$F', -shift_x, 1 + shift_y, '(f) Si$_3$N$_4$, CHF${}_{2}^{+}$'),
                ]
        for (system, ion, x, y, text) in texts:
            ax = ax_dict[system].get(ion)
            ax.text(x, y, text, transform=ax.transAxes, **text_opts)

    def generate_figure(self):
        plt.rcParams.update({
            'font.family': 'arial',
            'font.size': 10,
            })
        n_row, n_col = 2, 3
        multiplier = 7.1 / n_col
        figsize = (n_col * multiplier, n_row * multiplier)
        fig, axes = plt.subplots(n_row, n_col, figsize=figsize)

        ax_dict = {
                'SiO2': {
                    'CF$_3$': axes[0, 0],
                    'CF$_2$': axes[1, 0],
                    'CF': axes[0, 1],
                    'CHF$_2$/CH$_2$F': axes[1, 1],
                },
                'Si3N4': {
                    'CF$_2$': axes[0, 2],
                    'CHF$_2$/CH$_2$F': axes[1, 2],
                },
            }
        for _, axes in ax_dict.items():
            for key, ax in axes.items():
                ax.set_xlabel(r'$\sqrt{E}$')
                ax.set_ylabel('Etch yield (Si/ion)')
        return fig, ax_dict

    def plot_points(self, ax, data):
        key = 'sim'
        for ion_type in data[key].keys():
            x = np.array([i for i in data[key][ion_type].keys()])
            y = np.array([i for i in data[key][ion_type].values()])

            x = np.sqrt(x)
            plot_marker = PlotInfo.marker_list[ion_type]
            plot_color = PlotInfo.color_list[data['system']]
            label = PlotInfo.label_map[data['system']][key][ion_type]
            ax.scatter(x, y,
                       color=plot_color,
                       marker=plot_marker,
                       label=label,
                       zorder=1,
                       **PlotInfo.scatter_props)

            idx_select = np.where(np.abs(y) > 0.01)
            x = x[idx_select]
            y = y[idx_select]

            z = np.polyfit(x, y, 1)
            y_hat = np.poly1d(z)(x)
            ax.plot(x, y_hat, color=plot_color, zorder=0, **PlotInfo.fit_props)

    def plot_ref(self, ax, data):
        key = 'exp'
        system = data['system']
        for ion_type in data[key].keys():
            x = np.array([i for i in data[key][ion_type].keys()])
            y = np.array([i for i in data[key][ion_type].values()])
            color, marker = PlotInfo.scatter_prop_map.get(
                PlotInfo.label_map[system][key].get(ion_type)
            )
            label = PlotInfo.label_map[data['system']][key].get(ion_type)
            if len(y) > 1:
                ax.plot(x, y, color='k', linestyle='--', zorder=0)

            ax.scatter(x, y,
                       label=label,
                       marker=marker,
                       facecolor=color,
                       zorder=0,
                       **PlotInfo.scatter_props_ref)

    def decorate(self, ax):
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 2.0)
        ax.set_xlabel(r'$\sqrt{E_{\text{ion}}/\text{eV}}$')
        ax.set_ylabel('Etch yield (Si/ion)')

    def set_global_legend(self, fig, ax_dict):
        ax_list = []
        for _, axes in ax_dict.items():
            for key, ax in axes.items():
                ax_list.append(ax)

        handles, labels = [], []
        for ax in ax_list:
            handle, label = ax.get_legend_handles_labels()
            handles.extend(handle)
            labels.extend(label)

        # Remove duplicates
        unique_labels = []
        unique_handles = []
        for h, l in zip(handles, labels):
            if l not in unique_labels:
                unique_labels.append(l)
                unique_handles.append(h)

        pop_idx = unique_labels.index('This study')
        label_this_study = unique_labels.pop(pop_idx)
        handle_this_study = unique_handles.pop(pop_idx)
        unique_labels.insert(0, label_this_study)
        unique_handles.insert(0, handle_this_study)
        handles = unique_handles
        labels = unique_labels

        fig.legend(handles, labels, ncol=len(labels)/2, **PlotInfo.legend_props)

    @staticmethod
    def flip(items, ncol):
        return chain(*[items[i::ncol] for i in range(ncol)])

    def save(self, fig):
        # line = plt.Line2D([2/3, 2/3], [0.02, 0.98], color='grey', linestyle='--')
        # fig.add_artist(line)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.23)
        name = '3_1_2_valid_etchyield'
        fig.savefig(f'{name}.png')
        fig.savefig(f'{name}.pdf')
        fig.savefig(f'{name}.eps')

def main():
    pl = DataPlotter()
    with open('dat.yaml', 'r') as f:
        data_total = yaml.safe_load(f)
    pl.run(data_total)


if __name__ == '__main__':
    main()
