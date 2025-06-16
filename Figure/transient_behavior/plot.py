from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import yaml


@dataclass
class PARAMS:
    path_data_exp = 'CF_300_exp.dat'
    path_data = 'data_list.yaml'

    plot_prop_exp = {
        'label': 'Ishikawa et al.',
        'color': 'black',
        's': 10,
    }
    plot_prop_sim = {
        'linestyle': '-',
        'markersize': 1,
        'alpha': 0.5,
    }
    label_convert_dict = {
        'CF_300': 'This study',
        'CF_300 (removal height 2nm)': r'h$_{\mathrm{removal}}$ = 2 nm',
    }


class DataLoader:
    def run(self):
        data_exp = np.loadtxt(PARAMS.path_data_exp, skiprows=1)
        with open(PARAMS.path_data, 'r') as f:
            data_list = yaml.safe_load(f)
        result = {}
        for label, data_dict in data_list.items():
            data = np.loadtxt(data_dict['path'], skiprows=1)
            x_sim, y_sim = data[:, 0], data[:, 1]
            x_sim *= 1 / 9000
            y_sim -= y_sim[0]
            y_sim *= 0.1
            result[label] = { 'x': x_sim, 'y': y_sim, 'color': data_dict['color'] }
        return data_exp, result

class Plotter:
    def run(self, data_exp, data_list):
        fig, ax = self.generate_figure()
        x, y = data_exp[:, 0], data_exp[:, 1]
        ax.scatter(x, y, **PARAMS.plot_prop_exp)

        for key, (data_dict) in data_list.items():
            x_sim, y_sim, color = data_dict['x'], data_dict['y'], data_dict['color']
            label = PARAMS.label_convert_dict.get(key)
            ax.plot(x_sim, y_sim, color=color, label=label, **PARAMS.plot_prop_sim)

        self.decorate(ax)
        self.save(fig)

    def generate_figure(self):
        plt.rcParams.update({'font.family': 'Arial', 'font.size': 10})
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        return fig, ax

    def decorate(self, ax):
        ax.set_title('CF$^{+}$, 300 eV on SiO$_2$', fontsize=10)
        ax.set_xlabel('Ion dose (' + r'$\times$ ' + r'$10^{17}$ cm$^{-2}$)')
        ax.set_ylabel('Surface height change (nm)')
        ax.set_xlim(-0.05, 3.2)
        ax.set_ylim(-15, 45)
        ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), frameon=False)
        ax.axhline(0, color='grey', linestyle='--', lw=1, alpha=0.5)

    def save(self, fig):
        fig.tight_layout()
        name = '3_1_3_valid_transient_SiO2'
        fig.savefig(f'{name}.png', dpi=200)
        fig.savefig(f'{name}.pdf')
        fig.savefig(f'{name}.eps')

def main():
    dl = DataLoader()
    data_exp, data_list = dl.run()
    pl = Plotter()
    pl.run(data_exp, data_list)

if __name__ == "__main__":
    main()
