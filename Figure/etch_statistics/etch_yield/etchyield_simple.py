import sys
import numpy as np
import matplotlib.pyplot as plt


class EtchYieldPlotter():
    @staticmethod
    def run(path_yield, dst):
        data = np.loadtxt(path_yield, skiprows=2)
        norm_factor = 10 / 9000
        x_yield = data[:, 0] * norm_factor
        n_Si_etched = data[:, 1]
        etch_yield = data[:, 2]

        plt.rcParams.update({'font.size': 16})
        fig, (ax_Si, ax_yield) = plt.subplots(2, 1, figsize=(8, 6))

        ax_Si.plot(x_yield, n_Si_etched, color='black')
        ax_Si.set_xlabel(r"ion dose ($ \times 10^{16} \mathrm{cm}^{-2}$)")
        ax_Si.set_ylabel("number of etched Si")

        ax_yield.plot(x_yield, etch_yield, color='orange')
        ax_yield.set_xlabel(r"ion dose ($ \times 10^{16} \mathrm{cm}^{-2}$)")
        ax_yield.set_ylabel("Etch yield (Si/ion)")

        # yield_avg = np.mean(y_yield[-self.interval:])
        yield_avg = etch_yield[-1]
        textbox = f"yield = {yield_avg:.3f}"
        ax_yield.text(0.95, 0.05, textbox,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                      horizontalalignment='right',
                      verticalalignment='bottom',
                      transform=ax_yield.transAxes)

        fig.tight_layout()
        fig.savefig(f'{dst}.png')


def main():
    if len(sys.argv) != 3:
        print("Usage: python etchyield.py <path_yield> <dst>")
        sys.exit(1)

    path_yield = sys.argv[1]
    dst = sys.argv[2]
    EtchYieldPlotter.run(path_yield, dst)


if __name__ == "__main__":
    main()
