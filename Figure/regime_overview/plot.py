import sys
from dataclasses import dataclass

import yaml
import matplotlib.pyplot as plt

@dataclass
class PARAMS:
    """
    Hyperparameters for the Plotter:
    - regime_styles: dict mapping regime names to style dicts with 'marker' and 'color'
    - ion_order: list specifying order of ions on the x-axis
    """
    regime_styles = {
        'regime_1': {'marker': 'o', 'color': 'red'},
        'regime_2': {'marker': 's', 'color': 'blue'},
        'regime_3': {'marker': '^', 'color': 'green'},
    }
    ion_order = ['CH2F', 'CF', 'CHF2', 'CF2', 'CF3']
    ion_convert_dict = {
            'CF': 'CF$^{+}$',
            'CF2': 'CF${}_{2}^{+}$',
            'CF3': 'CF${}_{3}^{+}$',
            'CH2F': 'CH$_{2}$F$^{+}$',
            'CHF2': 'CHF${}_{2}^{+}$',
            }

    regime_convert_dict = {
            'regime_1': 'Deposition',
            'regime_2': 'Deposition, after initial etching',
            'regime_3': 'Etching',
            }

class Plotter:
    def __init__(self, data):
        """
        Args:
            data (dict): {
                'ion_1': {energy1: 'regime name', energy2: 'regime name', ...},
                'ion_2': {...},
                ...
            }
            params (class): A class containing hyperparameters (PARAMS)
        """
        self.data = data
        self.params = PARAMS
        # Determine x-axis order
        self.ions = (
            self.params.ion_order if self.params.ion_order else sorted(data.keys())
        )
        # Gather all unique energies across ions
        all_energies = set()
        for ion_dict in data.values():
            all_energies.update(ion_dict.keys())
        self.energies = sorted(all_energies)

    def run(self):
        """
        Create the scatter plot of regimes for each (ion, energy).
        Returns:
            fig, ax
        """
        fig, ax = plt.subplots(figsize=(3.5, 4))

        # map each ion to an x position
        x_positions = {ion: idx for idx, ion in enumerate(self.ions)}

        # Track which regimes have been plotted for the legend
        plotted = set()

        # Plot each point
        for ion, energy_map in self.data.items():
            x = x_positions.get(ion)
            for energy, regime in energy_map.items():
                style = self.params.regime_styles.get(regime)
                label = regime if regime not in plotted else None
                ax.scatter(
                    x,
                    energy,
                    marker=style['marker'],
                    color=style['color'],
                    label=PARAMS.regime_convert_dict.get(regime, regime) if label else None,
                )
                plotted.add(regime)

        # Configure axes
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels([PARAMS.ion_convert_dict.get(ion, ion) for ion in self.ions])
        ax.set_xlabel('Ion type')
        ax.set_ylabel('Ion energy (eV)')
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(labels)), key=lambda i: labels[i])
        ax.legend([handles[idx] for idx in order],
                  [labels[idx] for idx in order],
                  bbox_to_anchor=(0.5, -0.2),
                  loc='upper center',
                  # ncol=len(plotted),
                  ncol=1,
                  frameon=False,
                  )
        ax.set_ylim(0, 550)  # SiO2
        # ax.set_ylim(0, 1050)  # Si3N4

        fig.tight_layout()
        name = '3_2_1_regime_overview'
        fig.savefig(f'{name}.png')
        fig.savefig(f'{name}.pdf')
        fig.savefig(f'{name}.eps')



def main():
    if len(sys.argv) < 2:
        print("Usage: python plot.py <input.yaml>")
        return

    path_yaml = sys.argv[1]
    with open(path_yaml, 'r') as file:
        data = yaml.safe_load(file)

    pl = Plotter(data)
    pl.run()


if __name__ == "__main__":
    main()
