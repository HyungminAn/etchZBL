from dataclasses import dataclass
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


@dataclass
class PARAMS:
    KEYS = {
            (1, 0, 0, 0, 2): 'SiF2',
            (0, 1, 1, 0, 0): 'CO',
            (0, 0, 1, 0, 2): 'CF2',
            }
    CUT_ITER = 4500
    CONVERT_DICT = {
            'CF2': 'CF$_2$',
            'CO': 'CO',
            'SiF2': 'SiF$_2$',
            }
    COLOR_DICT = {
            'CF2': 'blue',
            'CO': 'red',
            'SiF2': 'green',
            }
    MARKER_DICT = {
            'CF2': 'o',
            'CO': 's',
            'SiF2': '^',
            }


class DataLoader:
    def run(self):
        self.data = self.get_data()
        self.data_exp = self.get_data_exp()
        return self.data, self.data_exp

    def get_data(self):
        path_dict = {
            50: "./src/CF3_50eV_mol_dict.pkl",
            100: "./src/CF3_100eV_mol_dict.pkl",
            200: "./src/CF3_200eV_mol_dict.pkl",
            250: "./src/CF3_250eV_mol_dict.pkl",
            500: "./src/CF3_500eV_mol_dict.pkl",
        }
        result = {}
        for energy, path in path_dict.items():
            with open(path, 'rb') as f:
                data = pickle.load(f)
            for key, name in PARAMS.KEYS.items():
                x = np.array(data[key])
                if name not in result:
                    result[name] = []
                result[name].append([energy, len(x[x < PARAMS.CUT_ITER])])

        for key, value in result.items():
            result[key] = np.array(value)
        return result

    def get_data_exp(self):
        path_dict = {
                'CF2': "./CF2.csv",
                'CO': "./CO.csv",
                'SiF2': "./SiF2.csv",
                }
        result = {}
        for key, path in path_dict.items():
            data = np.loadtxt(path, delimiter=' ')
            result[key] = data
        return result


class DataPlotter:
    def run(self, data_sim, data_exp):
        # fig, axes = self.generate_figure()
        # ax_exp, ax_bar = axes

        fig, axes = self.generate_figure()
        self.plot(data_sim, data_exp, fig, axes)
        # self.plot_bar(data_sim, data_exp, ax_bar)
        self.save(fig)

    def generate_figure(self):
        plt.rcParams.update({'font.size': 10, 'font.family': 'arial'})
        # fig, axes = plt.subplots(2, 1, figsize=(3.5, 2 * 3.5), constrained_layout=True)
        fig, axes = plt.subplots(3, 1, figsize=(3.5, 2 * 3.5),
                                 constrained_layout=True)
        return fig, axes

    def plot(self, data_sim, data_exp, fig, axes):
        keys = [key for key in data_exp.keys()]

        alphabets = ['(a)', '(b)', '(c)']
        legend_locs = ['upper right',
                       'upper left',
                       'upper left']

        for key, ax, count_ax, legend_loc in zip(keys, axes, alphabets, legend_locs):
            x1, y1 = data_exp[key][:, 0], data_exp[key][:, 1]
            x2, y2 = data_sim[key][:, 0], data_sim[key][:, 1]

            if key == 'CF2':
                y1 = y1 / y1[1]
            else:
                y1 = y1 / y1[2]
            y2 = y2 / y2[2]
            label = PARAMS.CONVERT_DICT[key]
            color = PARAMS.COLOR_DICT[key]

            ax.plot(x1, y1,
                    label='Toyoda et al.',
                    linewidth=0.5,
                    linestyle='--',
                    color='grey',
                    marker=PARAMS.MARKER_DICT[key],
                    markerfacecolor='white',
                    markeredgecolor='black')
            ax.plot(x2, y2,
                    label='This study',
                    linestyle='-',
                    marker=PARAMS.MARKER_DICT[key],
                    markerfacecolor='white',
                    markeredgecolor=color,
                    color=color)

            ax.set_xlim(0, 500)
            ax.set_ylim(0, None)
            text = f"{count_ax} {label}"
            ax.text(-0.15, 1.15, text, transform=ax.transAxes, ha='left', va='top')
            ax.set_xlabel('Incident energy (eV)')
            ax.legend(loc=legend_loc, frameon=False, fontsize=8)

        fig.supylabel('Desorption intensity (a.u.)')

        # plotLines = []
        # group_label = Line2D([], [], color='none', label='Toyoda et al.', linewidth=0)
        # plotLines.append(group_label)
        # for key in keys:
        #     data = data_exp[key]
        #     x, y = data[:, 0], data[:, 1]
        #     label = PARAMS.CONVERT_DICT[key]
        #     color = PARAMS.COLOR_DICT[key]
        #     line, = ax_exp.plot(x, y,
        #                 # alpha=0.5,
        #                 linewidth=0.5,
        #                 linestyle='--',
        #                 color='grey',
        #                 marker=PARAMS.MARKER_DICT[key],
        #                 label=label,
        #                 markerfacecolor='white',
        #                 markeredgecolor='black',
        #                )
        #     plotLines.append(line)

        # group_label = Line2D([], [], color='none', label='This study', linewidth=0)
        # plotLines.append(group_label)
        # for key in keys:
        #     data = data_sim[key]
        #     x, y = data[:, 0], data[:, 1]
        #     label = PARAMS.CONVERT_DICT[key]
        #     color = PARAMS.COLOR_DICT[key]
        #     line, = ax_sim.plot(x, y,
        #                 linestyle='-',
        #                 marker=PARAMS.MARKER_DICT[key],
        #                 label=label,
        #                 color=color)
        #     plotLines.append(line)

        # for ax in [ax_exp]:
        #     ax.set_xlabel('Incident energy (eV)')
        #     ax.set_ylabel('Desorption intensity in experiment (a.u.)')
        #     ax.set_xlim(0, 500)

        # for ax in [ax_sim]:
        #     ax.set_xlabel('Incident energy (eV)')
        #     ax.set_ylabel('Number of removed molecules')
        #     ax.set_xlim(0, 500)

        # self.set_global_legend(ax_exp, plotLines)
        # text = '(a)'
        # ax_exp.text(-0.2, 1.2,
        #         text, transform=ax_exp.transAxes, ha='left', va='top')

    # def set_global_legend(self, ax, lines):
    #     labels = [line.get_label() for line in lines]
    #     ax.legend(lines, labels, loc='upper center', frameon=False, bbox_to_anchor=(0.5, -0.2), ncol=2)

    def save(self, fig):
        # fig.tight_layout()
        # fig.subplots_adjust(bottom=0.2, top=0.1)
        name = '3_1_5_valid_byproduct_ratio'
        fig.savefig(f'{name}.png', dpi=200)
        fig.savefig(f'{name}.pdf')
        fig.savefig(f'{name}.eps')

    # def plot_bar(self, data_sim, data_exp, ax_bar):
    #     target = 200  # eV
    #     result_exp = {k: v[v[:, 0] == target, 1] for k, v in data_exp.items()}
    #     result_sim = {k: v[v[:, 0] == target, 1] for k, v in data_sim.items()}

    #     normalize_species = 'CO'
    #     result_exp = {k: v[0]/result_exp[normalize_species][0] for k, v in result_exp.items()}
    #     result_sim = {k: v[0]/result_sim[normalize_species][0] for k, v in result_sim.items()}

    #     print("Exp: ", result_exp)
    #     print("Sim: ", result_sim)

    #     keys = list(result_exp.keys())

    #     y_exp = np.array([result_exp[k] for k in keys])
    #     y_sim = np.array([result_sim[k] for k in keys])
    #     x = np.arange(len(keys))

    #     ax_bar.bar(x - 0.2, y_exp, width=0.4, label='Toyoda et al.',
    #                facecolor='white', edgecolor='black')
    #     ax_bar.bar(x + 0.2, y_sim, width=0.4, label='This study',
    #                facecolor='black', edgecolor='black')

    #     ax_bar.set_xticks(x)
    #     ax_bar.set_xticklabels([PARAMS.CONVERT_DICT[k] for k in keys])
    #     ax_bar.set_ylabel('Relative desorption intensity')

    #     ax_bar.set_title('CF${}_{3}^{+}$, 200 eV on SiO$_2$', fontsize=10)
    #     ax_bar.legend(loc='lower center',
    #                   frameon=False,
    #                   bbox_to_anchor=(0.5, -0.3),
    #                   ncol=2)

    #     text = '(b)'
    #     ax_bar.text(-0.2, 1.2,
    #             text, transform=ax_bar.transAxes, ha='left', va='top')

def main():
    loader = DataLoader()
    data, data_exp = loader.run()
    plotter = DataPlotter()
    plotter.run(data, data_exp)


if __name__ == "__main__":
    main()
