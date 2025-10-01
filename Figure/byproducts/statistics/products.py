import sys
import pickle
from dataclasses import dataclass
from collections import defaultdict

import yaml
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class PARAMS:
    GAS_CONVERT_DICT = {
            'SiF4': 'SiF$_4$',
            'OC': 'CO',
            'O2C': 'CO$_2$',
            'OCF2': 'COF$_2$',
            'CF2': 'CF$_2$',
            'O2': 'O$_2$',
            'SiF2': 'SiF$_2$',
            'CF4': 'CF$_4$',
            'OF2': 'OF$_2$',
            'F2': 'F$_2$',
            'OCH2': 'CH$_2$O',
            'OCHF': 'CHFO',
            'OH2': 'H$_2$O',

            'NCF': 'FCN',
            'NCH': 'HCN',
            'H2': 'H$_2$',
            'N2': 'N$_2$',
            'NH3': 'NH$_3$',
            'CH4': 'CH$_4$',
            'SiHF3': 'SiHF$_3$',
            'SiH2F2': 'SiH$_2$F$_2$',

            'CH2F2': 'CH$_2$F$_2$',
            'CHF3': 'CHF$_3$',
            'CH3F': 'CH$_3$F',
            }

    ION_CONVERT_DICT = {
            'CF': 'CF$^+$',
            'CF2': 'CF${}_{2}^{+}$',
            'CF3': 'CF${}_{3}^{+}$',
            'CH2F': 'CH${}_{2}$F$^+$',
            'CHF2': 'CHF${}_{2}^{+}$',
            }

class FigureGenerator:
    def run(self, paths):
        plt.rcParams.update({
            'font.size': 18,
            'font.family': 'arial',
            'hatch.linewidth': 0.5,
            })
        ion_energy_pair = {}
        for ion in paths.keys():
            ion_energy_pair[ion] = []
            for energy in paths[ion].keys():
                ion_energy_pair[ion].append(energy)
        n_row = len(ion_energy_pair)
        # n_col = max(len(energies) for energies in ion_energy_pair.values())
        n_col = 1

        multiplier = 2.3
        fig_size = (7.1 * multiplier, 7.1 * multiplier)
        fig, axes = plt.subplots(n_row, n_col, figsize=fig_size,)
        ax_dict = {}
        for row_idx, ion in enumerate(ion_energy_pair.keys()):
            ax = axes[row_idx]
            key = ion
            ax_dict[key] = ax
        return fig, ax_dict

class LabelConverter:
    def __init__(self, elem_dict):
        self.elem_dict = elem_dict

    def run(self, mol_dict):
        order = {
            k: idx
            for idx, (k, v) in enumerate(
                sorted(mol_dict.items(), key=lambda item: len(item[1]), reverse=True))
        }
        labels = [(k, self._stoichiometry_to_str(k))
                  for (k, _) in sorted(order.items(), key=lambda x: x[1])]
        # trunc_len = 10
        # if len(labels) > trunc_len:
        #     labels = labels[:trunc_len] + [f'_{i}' for i in labels[trunc_len:]]
        labels = {k: v for k, v in labels}
        return labels

    def _stoichiometry_to_str(self, stoichiometry):
        elem_dict = self.elem_dict
        return ''.join(f"{elem_dict[i]}{count}"
                       if count > 1 else f"{elem_dict[i]}"
                       for i, count in enumerate(stoichiometry)
                       if count > 0)

class SpeciesCounter:
    def run(self, paths, elem_dict, cutoff_inc=4500):
        lbc = LabelConverter(elem_dict)
        result = {}
        for ion in paths.keys():
            for energy, path in paths[ion].items():
                with open(path, 'rb') as f:
                    mol_dict = pickle.load(f)
                label_dict = lbc.run(mol_dict)
                key = (ion, energy)
                if key not in result:
                    result[key] = {}
                for stoichiometry, gen_idx_list in mol_dict.items():
                    count = len([i for i in gen_idx_list if i <= cutoff_inc])
                    if count == 0:
                        continue
                    result[key][label_dict[stoichiometry]] = count
                    print(f"{key} {label_dict[stoichiometry]}: {count}")
        return result

class Plotter:
    """
    Grouped bar plot for byproducts per ion/energy.

    Expects input `result` shaped like:
        {
            (ion, energy): {byproduct_type: count, ...},
            ...
        }

    For each ion (one row / axis):
      - X-axis shows byproduct types, sorted by the total count across all energies (descending).
      - Within each byproduct group, bars for different energies are placed side-by-side.
    """

    def __init__(
        self,
        *,
        figsize_scale_x: float = 0.35,  # Scales figure width by number of byproducts (per row)
        figsize_per_row: float = 1.9,   # Height (inches) per ion row
        max_group_width: float = 0.9,   # Total relative width occupied by bars within a byproduct group
        show_legend_per_row: bool = False,  # Show legend on each row instead of only the first
        sharex: bool = False,            # Share x-axis across rows
        sharey: bool = False,           # Share y-axis across rows
    ):
        self.figsize_scale_x = figsize_scale_x
        self.figsize_per_row = figsize_per_row
        self.max_group_width = max_group_width
        self.show_legend_per_row = show_legend_per_row
        self.sharex = sharex
        self.sharey = sharey

    @staticmethod
    def _sort_energy_key(e):
        """Sort energies numerically when possible, otherwise lexicographically."""
        try:
            return (0, float(e))
        except (TypeError, ValueError):
            return (1, str(e))

    def _prepare(self, result: dict):
        """Aggregate and index data structures needed for plotting."""
        if not result:
            raise ValueError("result dict is empty.")

        # Sets and maps for fast lookup
        ions = sorted({ion for (ion, _e) in result.keys()}, key=lambda x: str(x))
        ion_to_energies = defaultdict(set)
        ion_to_byproducts = defaultdict(set)

        # counts[(ion, energy)][byproduct] = count
        counts = defaultdict(lambda: defaultdict(float))

        for (ion, energy), bp_dict in result.items():
            ion_to_energies[ion].add(energy)
            for bp, c in bp_dict.items():
                ion_to_byproducts[ion].add(bp)
                counts[(ion, energy)][bp] += float(c)

        return ions, ion_to_energies, ion_to_byproducts, counts

    def run(self, result: dict):
        """
        Build the figure and axes for the grouped bar plots.

        Parameters
        ----------
        result : dict
            {(ion, energy): {byproduct_type: count, ...}, ...}

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : list[matplotlib.axes.Axes]
        """
        ions, ion_to_energies, ion_to_byproducts, counts = self._prepare(result)

        # Compute dynamic figure size based on the max number of byproducts across ions
        max_bps = max(len(bps) for bps in ion_to_byproducts.values())
        fig_w = max(6.0, self.figsize_scale_x * max(1, max_bps))
        fig_h = max(2.2, self.figsize_per_row * max(1, len(ions)))

        plt.rcParams.update({'font.size': 10, 'font.family': 'arial'})
        fig, axes = plt.subplots(
            nrows=len(ions),
            ncols=1,
            figsize=(fig_w, fig_h),
            sharex=self.sharex,
            sharey=self.sharey,
            squeeze=False,
        )
        axes = [ax for (ax,) in axes]  # flatten to a simple list

        # Will be reused for the bottom x-ticks (shared x)
        last_byproducts_sorted = None
        first_row_energy_count = None

        for ax, ion in zip(axes, ions):
            energies = sorted(ion_to_energies[ion], key=self._sort_energy_key)
            byproducts = list(ion_to_byproducts[ion])

            # Sort byproducts by total count across energies (descending, then by name)
            totals = {
                bp: sum(counts[(ion, e)].get(bp, 0.0) for e in energies)
                for bp in byproducts
            }
            byproducts_sorted = sorted(byproducts, key=lambda bp: (-totals[bp], str(bp)))
            ticks = np.arange(len(byproducts_sorted))
            labels = [PARAMS.GAS_CONVERT_DICT.get(bp, str(bp)) for bp in byproducts_sorted]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=45, ha="right")

            x = np.arange(len(byproducts_sorted), dtype=float)
            nE = max(1, len(energies))

            # Compute per-energy bar width and offsets within a group
            per_bar_w = self.max_group_width / nE
            start = -self.max_group_width / 2 + per_bar_w / 2
            offsets = [start + i * per_bar_w for i in range(nE)]

            # Plot bars for each energy
            for i, e in enumerate(energies):
                y = np.array([counts[(ion, e)].get(bp, 0.0) for bp in byproducts_sorted], dtype=float)
                ax.bar(x + offsets[i],
                       y,
                       width=per_bar_w,
                       label=str(e))

            # Styling per row
            ax.set_ylabel(PARAMS.ION_CONVERT_DICT.get(ion, str(ion)))
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            ax.axhline(0, linewidth=0.8, alpha=0.8)

            # Legend placement
            if self.show_legend_per_row:
                ax.legend(title="Energy", ncols=min(6, nE), frameon=False)

            # Save last byproducts list for shared x tick labels
            last_byproducts_sorted = byproducts_sorted
            if first_row_energy_count is None:
                first_row_energy_count = nE

        # # Shared x labels and ticks on the bottom axis
        # if last_byproducts_sorted is not None:
        #     ticks = np.arange(len(last_byproducts_sorted))
        #     labels = [str(bp) for bp in last_byproducts_sorted]
        #     for ax in axes:
        #         ax.set_xticks(ticks)
        #         ax.set_xticklabels(labels, rotation=45, ha="right")

        axes[-1].set_xlabel("Byproduct type")
        fig.supylabel("Count")

        # Single legend at the top if not per-row
        if not self.show_legend_per_row and first_row_energy_count is not None:
            axes[0].legend(title="Energy (eV)",
                           # ncols=min(6, first_row_energy_count),
                           ncols=1,
                           frameon=False,
                           loc='upper right',
                           )

        fig.tight_layout()
        fig.savefig("result_byproduct_stats.png", dpi=200)
        return fig, axes


def main():
    if len(sys.argv) != 3:
        print("Usage: python products.py <path_yaml> <SiO2/Si3N4>")
        sys.exit(1)
    path_yaml = sys.argv[1]
    with open(path_yaml, 'r') as f:
        paths = yaml.safe_load(f)
    system = sys.argv[2]
    cutoff_inc = 4500

    if system == 'SiO2':
        elem_dict = {0: 'Si', 1: 'O', 2: 'C', 3: 'H', 4: 'F'}
    elif system == 'Si3N4':
        elem_dict = {0: 'Si', 1: 'N', 2: 'C', 3: 'H', 4: 'F'}
    else:
        print("Invalid system. Choose 'SiO2' or 'Si3N4'.")
        sys.exit(1)

    sc = SpeciesCounter()
    data = sc.run(paths, elem_dict, cutoff_inc=cutoff_inc)

    pl = Plotter()
    pl.run(data)

if __name__ == "__main__":
    main()
