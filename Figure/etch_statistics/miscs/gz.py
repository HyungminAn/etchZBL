import sys
from ase.io import read
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import numpy as np
from itertools import repeat
from dataclasses import dataclass


@dataclass
class PlotInfo:
    ELEM_DICT = {
            'Si': 0,
            'O': 1,
            'C': 2,
            'H': 3,
            'F': 4,
            }
    ELEM_GROUP_DICT = {
            'SiO': [0, 1],
            'CHF': [2, 3, 4],
            }
    Z_CUT = 100  # Angstrom
    GRID_SIZE = 0.05  # Angstrom
    SIGMA = 4

    READ_OPTIONS = {
        'format': 'lammps-data',
        'atom_style': 'atomic',
        'sort_by_id': False
    }

class AtomicDensityPlotter:
    def __init__(self, path_image):
        self.path_image = path_image

        self.z_cut = PlotInfo.Z_CUT
        self.grid_size = PlotInfo.GRID_SIZE
        self.sigma = PlotInfo.SIGMA
        self.elem_dict = PlotInfo.ELEM_DICT
        self.elem_group_dict = PlotInfo.ELEM_GROUP_DICT

    def run(self):
        image = read(self.path_image, **PlotInfo.READ_OPTIONS)
        x, species_gz = self._get_gz(image)
        z_max = np.max(image.get_positions()[:, 2])
        self._normalize(x, species_gz, z_max)
        self._plot_gz_group(x, species_gz, z_max)

    @staticmethod
    def _cal_atomic_gz(z, plt_x_axis, sigma):
        return np.exp(-((z - plt_x_axis)/sigma)**2/2.) / np.sqrt(2*np.pi) / sigma

    def _get_gz(self, image):
        pos_z = image.get_positions()[:, 2]
        n_grid = int(self.z_cut / self.grid_size)
        x = np.linspace(0, self.z_cut, n_grid)
        x_list = [x.copy() for _ in range(len(pos_z))]

        pool = Pool(cpu_count())
        atomic_gz = pool.starmap(self._cal_atomic_gz, zip(pos_z, x_list, repeat(self.sigma)))

        species_gz = np.zeros((len(x_list[0]), len(self.elem_dict)))
        atomic_numbers = image.get_array('type')
        for e, i in enumerate(atomic_gz):
            idx_elem = atomic_numbers[e] - 1
            species_gz[:, idx_elem] += i

        return x, species_gz

    @staticmethod
    def _normalize(x, species_gz, z_max):
        mask = x <= z_max
        species_gz[~mask] = 0
        row_sums = np.sum(species_gz[mask], axis=1, keepdims=True)
        species_gz[mask] /= row_sums

    def _plot_gz_group(self, x, species_gz, z_max):
        fig, (ax_species, ax_group) = plt.subplots(1, 2)

        for elem, idx in self.elem_dict.items():
            y = species_gz[:, idx]
            ax_species.plot(y, x, label=elem)

        for elem_group, idx in self.elem_group_dict.items():
            y = np.sum(species_gz[:, idx], axis=1)
            ax_group.plot(y, x, label=elem_group)

        for ax in [ax_species, ax_group]:
            ax.set_xlabel("$g(z)$")
            ax.set_ylabel("Height ($\AA$)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, self.z_cut)
            ax.legend(loc="upper right")
            ax.axhline(z_max, color='grey', linestyle='--')

        fig.tight_layout()
        fig.savefig(f"gz.png")

if __name__ == '__main__':
    path_image = sys.argv[1]
    plotter = AtomicDensityPlotter(path_image)
    plotter.run()
