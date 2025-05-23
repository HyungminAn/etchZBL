import yaml

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class PlotInfo:
    color_list = {
        'SiO2': 'red',
        'Si3N4': 'blue',
    }
    marker_list = {
        'CF3': 'o',
        'CF2': 's',
        'CF': '^',
        'CH2F': 'd',
        'CHF2': 'p',
    }
    species_list = [
        'CF3_',
        'CF2_',
        'CF_',
        'CHF2_',
        'CH2F_',
    ]
    legend_props = {
        'fontsize': 14,
        'loc': 'upper left',
        'bbox_to_anchor': (0.02, 0.98),
    }
    scatter_props = {
        's': 70,
        'edgecolor': 'k',
        'alpha': 0.7,
    }
    scatter_props_ref = {
        's': 70,
        'edgecolor': 'k',
        'color': 'w',
    }
    fit_props = {
        'linestyle': '--',
        'alpha': 0.3,
    }
    label_map = {
        'CF3': 'CF${}_{3}^{+}$',
        'CF2': 'CF${}_{2}^{+}$',
        'CF': 'CF$^{+}$',
        'CHF2': 'CHF${}_{2}^{+}$',
        'CH2F': 'CH$_2$F$^{+}$',
    }

class DataPlotter:
    def get_plot_fig(self):
        plt.rcParams.update({
            'font.size': 16,
            'font.family': 'arial',
            })
        n_row, n_col = 2, 3
        fig, axes = plt.subplots(n_row, n_col, figsize=(6*n_col, 6*n_row))

        ax_dict = {
                'SiO2': {
                    'CF$_3$': axes[0, 0],
                    'CF$_2$': axes[0, 1],
                    'CF': axes[1, 0],
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
                # ax.set_title(key)
        return fig, ax_dict

    def plot_separate(self, data_total):
        fig, ax_dict = self.get_plot_fig()
        key_convert_dict = {
                'CF3_': 'CF$_3$',
                'CF2_': 'CF$_2$',
                'CF_': 'CF',
                'CHF2_': 'CHF$_2$/CH$_2$F',
                'CH2F_': 'CHF$_2$/CH$_2$F',
                }

        for system_type, value_dict in data_total.items():
            for species in PlotInfo.species_list:
                ax = ax_dict[system_type].get(key_convert_dict[species])
                if ax is None:
                    continue
                data_exp = {k.replace('_', ''): v for k, v in value_dict['exp'].items() if species in k}
                data_sim = {k.replace('_', ''): v for k, v in value_dict['sim'].items() if species in k}

                data_dict = {'exp': data_exp, 'sim': data_sim, 'system': system_type}

                self.plot_ref(ax, data_dict)
                self.plot_points(ax, data_dict)
                self.decorate(ax)

        text_opts = {
                'fontsize': 24,
                'ha': 'left',
                'va': 'top',
                }
        stride_x, stride_y, shift_x, shift_y = 1/3, 0.5, 0.15, 0.2

        texts = [
                ('SiO2',  'CF$_3$',          -shift_x, 1 + shift_y, '(a) SiO$_2$, CF${}_{3}^{+}$'),
                ('SiO2',  'CF$_2$',          -shift_x, 1 + shift_y, '(b) SiO$_2$, CF${}_{2}^{+}$'),
                ('Si3N4', 'CF$_2$',          -shift_x, 1 + shift_y, '(e) Si$_3$N$_4$, CF${}_{2}^{+}$'),
                ('SiO2',  'CF',              -shift_x, 1 + shift_y, '(c) SiO$_2$, CF$^{+}$'),
                ('SiO2',  'CHF$_2$/CH$_2$F', -shift_x, 1 + shift_y, '(d) SiO$_2$, CHF${}_{2}^{+}$/CH$_2$F$^{+}$'),
                ('Si3N4', 'CHF$_2$/CH$_2$F', -shift_x, 1 + shift_y, '(f) Si$_3$N$_4$, CHF${}_{2}^{+}$'),
                ]
        for (system, ion, x, y, text) in texts:
            ax = ax_dict[system].get(ion)
            ax.text(x, y, text, transform=ax.transAxes, **text_opts)

        line = plt.Line2D([2/3, 2/3], [0.02, 0.98], color='grey', linestyle='--')
        fig.add_artist(line)
        fig.tight_layout(pad=2.0)
        name = 'result'
        fig.savefig(f'{name}.png')
        fig.savefig(f'{name}.pdf')
        fig.savefig(f'{name}.eps')


    def plot_points(self, ax, data):
        key = 'sim'
        for ion_type in data[key].keys():
            x = np.array([i for i in data[key][ion_type].keys()])
            y = np.array([i for i in data[key][ion_type].values()])

            x = np.sqrt(x)
            plot_marker = PlotInfo.marker_list[ion_type]
            plot_color = PlotInfo.color_list[data['system']]
            label = f'{PlotInfo.label_map[ion_type]}'
            ax.scatter(x, y,
                       color=plot_color,
                       marker=plot_marker,
                       label=label,
                       **PlotInfo.scatter_props)

            idx_select = np.where(np.abs(y) > 0.01)
            x = x[idx_select]
            y = y[idx_select]

            z = np.polyfit(x, y, 1)
            y_hat = np.poly1d(z)(x)
            ax.plot(x, y_hat, color=plot_color, **PlotInfo.fit_props)

            # cutoff_E = np.square(z[1] / z[0])
            # textbox = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n"
            # textbox += f"$R^2$ ={r2_score(y, y_hat):0.3f}\n"
            # textbox += f"cutoff_E = {cutoff_E:.2f} eV"

            # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # ax.text(
            #     0.05, 0.95, textbox,
            #     horizontalalignment='left',
            #     verticalalignment='top',
            #     transform=ax.transAxes, bbox=props)

    def plot_ref(self, ax, data):
        key = 'exp'
        if len([i for i in data[key].keys()]) > 2:
            raise ValueError('Too many data in the reference data')

        count = 0
        for ion_type in data[key].keys():
            x = np.array([i for i in data[key][ion_type].keys()])
            y = np.array([i for i in data[key][ion_type].values()])
            marker = PlotInfo.marker_list[ion_type.split()[0]]

            my_ion, label = ion_type.split()[0], ""
            if my_ion in PlotInfo.label_map.keys():
                label = ion_type.replace(my_ion, PlotInfo.label_map[my_ion])
            if len(y) > 1:
                ax.plot(x, y, color='k', linestyle='--')
            if count == 0:
                ax.scatter(x, y,
                           label=f"{label} (Exp)",
                           marker=marker,
                           facecolor='none',
                           **PlotInfo.scatter_props_ref)
                count += 1
            else:
                ax.scatter(x, y,
                           label=f"{label} (Exp)",
                           marker=marker,
                           facecolor='black',
                           **PlotInfo.scatter_props_ref)

    def decorate(self, ax):
        ax.legend(**PlotInfo.legend_props)

        ax.set_xlim(0, 50)
        ax.set_ylim(0, 2.0)
        ax.set_xlabel(r'$\sqrt{E_{\text{ion}}/\text{eV}}$')
        ax.set_ylabel('Etch yield (Si/ion)')

    @staticmethod
    def flip(items, ncol):
        from itertools import chain
        return chain(*[items[i::ncol] for i in range(ncol)])


def main():
    plotter = DataPlotter()
    with open('dat.yaml', 'r') as f:
        data_total = yaml.safe_load(f)
    plotter.plot_separate(data_total)


if __name__ == '__main__':
    main()
