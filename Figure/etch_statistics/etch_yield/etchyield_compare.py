import sys

import matplotlib.pyplot as plt
import numpy as np
import pickle

from etchyield import EYCalculatorFromStructure


class EtchYieldComparator:
    def __init__(self, plotter1, plotter2, interval=10):
        self.plotter1 = plotter1
        self.plotter2 = plotter2
        self.interval = interval

    def compare(self):
        plt.rcParams.update({'font.size': 18})
        fig, (ax_Si, ax_yield) = plt.subplots(2, 1, figsize=(12, 10))

        x1 = np.arange(len(self.plotter1.n_Si_etched)) * self.plotter1.norm_factor
        x2 = np.arange(len(self.plotter2.n_Si_etched)) * self.plotter2.norm_factor

        ax_Si.plot(x1, self.plotter1.n_Si_etched, color='black', label=self.plotter1.__class__.__name__)
        ax_Si.plot(x2, self.plotter2.n_Si_etched, color='red', label=self.plotter2.__class__.__name__)
        ax_Si.set_xlabel(r"ion dose ($ \times 10^{16} \mathrm{cm}^{-2}$)")
        ax_Si.set_ylabel("number of etched Si")
        ax_Si.set_title("Etch Yield Comparison")
        ax_Si.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

        y_yield1 = [y for y in self.plotter1.etch_yield if y >= 0]
        y_yield2 = [y for y in self.plotter2.etch_yield if y >= 0]
        x_yield1 = np.arange(len(y_yield1)) * self.plotter1.norm_factor
        x_yield2 = np.arange(len(y_yield2)) * self.plotter2.norm_factor

        ax_yield.plot(x_yield1, y_yield1, color='orange', label=self.plotter1.__class__.__name__)
        ax_yield.plot(x_yield2, y_yield2, color='green', label=self.plotter2.__class__.__name__)
        ax_yield.set_xlabel(r"ion dose ($ \times 10^{16} \mathrm{cm}^{-2}$)")
        ax_yield.set_ylabel("Etch yield (Si/ion)")
        ax_yield.legend(loc='center left', bbox_to_anchor=(1.1, 0.7))

        ax_yield.set_title(f"Interval: {self.interval}")

        yield_avg1 = np.mean(y_yield1[-self.interval:])
        yield_avg2 = np.mean(y_yield2[-self.interval:])
        textbox = f"{self.plotter1.__class__.__name__} yield = {yield_avg1:.3f}\n{self.plotter2.__class__.__name__} yield = {yield_avg2:.3f}"
        ax_yield.text(1.1, 0.3, textbox,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                      horizontalalignment='left',
                      verticalalignment='center',
                      transform=ax_yield.transAxes)

        fig.tight_layout()
        fig.savefig('etch_yield_comparison.png')

class EYPlotterFromStructureParallel():
    def run(self, x_yield_list, y_yield_list):
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize=(8, 6))
        textbox = []

        labels = [key for key in x_yield_list.keys()]

        for label in labels:
            x = x_yield_list[label]
            y = y_yield_list[label]
            ax.plot(x, y, label=label)
            ax.set_xlabel(r"ion dose ($ \times 10^{16} \mathrm{cm}^{-2}$)")
            ax.set_ylabel("Etch yield (Si/ion)")

            yield_avg = y[-1]
            textbox.append(f"yield = {yield_avg:.3f} ({label})")

        textbox = "\n".join(textbox)
        text_props = {
                'bbox': dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                'horizontalalignment': 'right',
                'verticalalignment': 'bottom',
                'transform': ax.transAxes,
                }
        ax.text(0.95, 0.05, textbox, **text_props)
        ax.legend(loc='center right', bbox_to_anchor=(0.95, 0.5))
        fig.tight_layout()
        fig.savefig(f'compare_result.png')

def run_EYplotterFromStructureParallel():
    if len(sys.argv) % 2 != 1:
        print("Usage: python etchyield.py [src1] [label1] [src2] [label2] ...")
        sys.exit(1)

    src_list = [i for i in sys.argv[1::2]]
    label_list = [i for i in sys.argv[2::2]]

    x_yield_list = {}
    y_tield_list = {}

    for src, label in zip(src_list, label_list):
        with open(src, 'rb') as f:
            data = pickle.load(f)
        x_yield_list[label] = data['x_yield']
        y_tield_list[label] = data['y_yield']

    plotter = EYPlotterFromStructureParallel()
    plotter.run(x_yield_list, y_tield_list)

# def run_EYcomparator():
#     comparator = EtchYieldComparator(plotter1, plotter2, interval=interval)
#     comparator.compare()


if __name__ == "__main__":
    run_EYplotterFromStructureParallel()
