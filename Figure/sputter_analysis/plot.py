from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class PARAMS:
    colors = {
        'Si': '#F0C8A0',
        'N': '#3050F8',
        'C': '#909090',
        'F': '#90E050',
    }
    linestyles = {
        'rm_in_MD': '-',
        'rm_in_byp': '-',
    }
    text_dict = {
        'rm_in_MD': '(a) Sputtered during MD',
        'rm_in_byp': '(b) Removed as byproducts',
    }

class DataLoader:
    def run(self, path):
        print('Loading data from:', path)
        col_idx_dict = {
            'rm_in_MD': {
                'Si': 2,
                'N': 3,
                'C': 4,
                'F': 5,
                },
            'rm_in_byp': {
                'Si': 6,
                'N': 7,
                'C': 8,
                'F': 9,
                },
        }
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        x = data[:, 0] / 9000
        result = {}
        for key, col_idx in col_idx_dict.items():
            result[key] = {}
            for element, idx in col_idx.items():
                y = np.cumsum(data[:, idx])
                result[key][element] = (x, y)
        print('Data loaded successfully.')
        return result

class FigureGenerator:
    def run(self):
        plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.5))
        return fig, axes

class DataPlotter:
    def run(self, data):
        print('Plotting data...')
        fg = FigureGenerator()
        fig, axes = fg.run()
        self.plot(data, axes)
        self.decorate(data, axes)
        self.set_global_legend(fig, axes)
        self.save(fig)
        print('Plotting completed successfully.')

    def plot(self, data, axes):
        ax_rm, ax_byp = axes

        for dat_type, dat in data.items():
            if dat_type == 'rm_in_MD':
                ax = ax_rm
            elif dat_type == 'rm_in_byp':
                ax = ax_byp
            else:
                raise ValueError(f"Unknown data type: {dat_type}")

            for element, (x, y) in dat.items():
                ax.plot(x, y,
                        label=element,
                        color=PARAMS.colors[element],
                        linestyle=PARAMS.linestyles[dat_type])

    def decorate(self, data, axes):
        # ylim = 0
        # for ax in axes:
        #     ylim = max(ylim, ax.get_ylim()[1])

        for dat_type, ax in zip(data.keys(), axes):
            ax.set_xlabel(r'Ion dose ($\times$ 10$^{17}$ cm$^{-2}$)')
            ax.set_xlim(0, 1.5)
            ax.set_ylim(0, None)
            ax.set_ylabel('Count')

            text = PARAMS.text_dict[dat_type]
            ax.text(-0.1, 1.1, text, transform=ax.transAxes,
                    fontsize=10, va='top', ha='left')

    def set_global_legend(self, fig, axes):
        handles, labels = [], []
        for ax in axes:
            hs, ls = ax.get_legend_handles_labels()
            for h, l in zip(hs, ls):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        fig.legend(handles, labels, loc='lower center', ncol=len(labels),
                   fontsize=10, bbox_to_anchor=(0.5, 0.02), frameon=False)

    def save(self, fig):
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)
        fig.savefig('result.png')

def main():
    dl = DataLoader()
    data = dl.run('rm_CF3_1000eV.csv')

    dp = DataPlotter()
    dp.run(data)


if __name__ == "__main__":
    main()
