import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'

convert_dict = {
        'CF_300': 'CF$^+$, 300 eV',
        }


class EtchYieldPlotter():
    @staticmethod
    def run(path_yield, dst, display_Si=False):
        data = np.loadtxt(path_yield, skiprows=2)
        norm_factor = 1 / 9000
        x_yield = data[:, 0] * norm_factor
        n_Si_etched = data[:, 1]
        etch_yield = data[:, 2]

        plt.rcParams.update({'font.size': 10})
        if display_Si:
            fig, (ax_Si, ax_yield) = plt.subplots(2, 1, figsize=(7.1, 3.5))
            ax_Si.plot(x_yield, n_Si_etched, color='black')
            ax_Si.set_xlabel(r"Ion dose ($\times$ 10$^{17}$ $\mathrm{cm}^{-2}$)")
            ax_Si.set_ylabel("# Si")
        else:
            fig, ax_yield = plt.subplots(1, 1, figsize=(3.5, 3.5))

        ax_yield.plot(x_yield, etch_yield, color='red')
        ax_yield.set_xlabel(r"Ion dose ($\times$ 10$^{17}$ $\mathrm{cm}^{-2}$)")
        ax_yield.set_ylabel("Etch yield (Si/ion)")

        yield_avg = etch_yield[-1]
        textbox = f"yield = {yield_avg:.3f}"
        if yield_avg > 0.5:
            x_text, y_text, ha, va = 0.95, 0.05, 'right', 'bottom'
        else:
            x_text, y_text, ha, va = 0.95, 0.95, 'right', 'top'

        ax_yield.text(x_text, y_text, textbox,
                      # bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                      horizontalalignment=ha,
                      verticalalignment=va,
                      transform=ax_yield.transAxes)

        ax_yield.set_xlim(0, None)
        ax_yield.set_ylim(0, 1.5)

        ax_yield.axvline(0.5, color='grey', linestyle='--', alpha=0.5)
        ax_yield.set_title(convert_dict.get(dst, dst))

        # fig.suptitle(dst)
        fig.tight_layout()
        fig.savefig(f'{dst}.png', dpi=200)


def main():
    if len(sys.argv) != 3:
        print("Usage: python etchyield.py <path_yield> <dst>")
        sys.exit(1)

    path_yield = sys.argv[1]
    dst = sys.argv[2]
    EtchYieldPlotter.run(path_yield, dst)


if __name__ == "__main__":
    main()
