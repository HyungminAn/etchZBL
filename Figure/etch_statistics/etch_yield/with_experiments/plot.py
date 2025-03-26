import yaml

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PlotInfo:
    color_marker_list = {
        'CF3': ('red', 'o'),
        'CF2': ('red', 's'),
        'CF': ('red', '^'),
        'CH2F': ('red', 'd'),
        'CHF2': ('red', 'p'),
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
        'loc': 'upper center',
        'bbox_to_anchor': (0.5, -0.2),
        'ncol': 3,
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

class DataPlotter:
    def get_plot_fig(self):
        plt.rcParams.update({
            'font.size': 16,
            'font.family': 'arial',
            })
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlabel(r'$\sqrt{E}$')
        ax.set_ylabel('Etch yield (Si/ion)')
        return fig, ax

    def plot_all_in_one(self, data_total):
        fig, ax = self.get_plot_fig()
        selected_species = ['CF3_', 'CF2_', 'CF_']

        for species in selected_species:
            data_exp = {k.replace('_', ''): v for k, v in data_total['exp'].items() if species in k}
            data_sim = {k.replace('_', ''): v for k, v in data_total['sim'].items() if species in k}
            data = {'exp': data_exp, 'sim': data_sim}

            self.plot_ref(ax, data)
            self.plot_points(ax, data)

        self.decorate(ax)
        fig.tight_layout()
        fig.savefig(f'etch_yield_total.png')

    def plot_separate(self, data_total):
        fig, ax = self.get_plot_fig()
        for species in PlotInfo.species_list:
            ax.clear()
            data_exp = {k.replace('_', ''): v for k, v in data_total['exp'].items() if species in k}
            data_sim = {k.replace('_', ''): v for k, v in data_total['sim'].items() if species in k}
            data = {'exp': data_exp, 'sim': data_sim}

            self.plot_ref(ax, data)
            self.plot_points(ax, data)
            self.decorate(ax)

            fig.tight_layout()
            fig.savefig(f'etch_yield_{species}.png')


    def plot_points(self, ax, data):
        key = 'sim'
        for ion_type in data[key].keys():
            x = np.array([i for i in data[key][ion_type].keys()])
            y = np.array([i for i in data[key][ion_type].values()])
            plot_color, plot_marker = PlotInfo.color_marker_list[ion_type]
            ax.scatter(x, y,
                       color=plot_color,
                       marker=plot_marker,
                       label=f"{ion_type}",
                       **PlotInfo.scatter_props)

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
            _, marker = PlotInfo.color_marker_list[ion_type.split()[0]]
            if len(y) > 1:
                ax.plot(x, y, color='k', linestyle='--')
            if count == 0:
                ax.scatter(x, y,
                           label=f"{ion_type} (Exp)",
                           marker=marker,
                           facecolor='none',
                           **PlotInfo.scatter_props_ref)
                count += 1
            else:
                ax.scatter(x, y,
                           label=f"{ion_type} (Exp)",
                           marker=marker,
                           facecolor='black',
                           **PlotInfo.scatter_props_ref)

    def decorate(self, ax):
        # Re-order legends
        handles, labels = plt.gca().get_legend_handles_labels()
        order = np.argsort(np.array(labels))
        handles_sorted = [handles[i] for i in order]
        labels_sorted = [labels[i] for i in order]

        import matplotlib.patches as mpatches
        empty_patch = mpatches.Patch(color=(0,0,0,0), label="")
        handles_padded = handles_sorted[:2] + [empty_patch] + handles_sorted[2:4] + [empty_patch] + handles_sorted[4:7]
        labels_padded = labels_sorted[:2] + [""] + labels_sorted[2:4] + [""] + labels_sorted[4:7]

        ax.legend(handles_padded, labels_padded, **PlotInfo.legend_props)
        # ax.legend(DataPlotter.flip(handles_sorted, PlotInfo.legend_props['ncol']),
        #           DataPlotter.flip(labels_sorted, PlotInfo.legend_props['ncol']),
        #           **PlotInfo.legend_props)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.set_xlabel(r'$\sqrt{\text{(ion energy)/eV}}$')
        ax.set_ylabel('Etch yield (Si/ion)')

    @staticmethod
    def flip(items, ncol):
        from itertools import chain
        return chain(*[items[i::ncol] for i in range(ncol)])


def main():
    plotter = DataPlotter()
    with open('dat.yaml', 'r') as f:
        data_total = yaml.safe_load(f)
    # plotter.plot_separate(data_total)
    plotter.plot_all_in_one(data_total)


if __name__ == '__main__':
    main()
