import os
import sys
from multiprocessing import Pool, cpu_count
from itertools import repeat
from dataclasses import dataclass
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ase.io import read


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

class GZCalculator:
    def __init__(self, path_images):
        self.path_images = path_images

        self.z_cut = PlotInfo.Z_CUT
        self.grid_size = PlotInfo.GRID_SIZE
        self.sigma = PlotInfo.SIGMA
        self.elem_dict = PlotInfo.ELEM_DICT
        self.elem_group_dict = PlotInfo.ELEM_GROUP_DICT

    def run(self):
        path_save = "data.pkl"
        if os.path.exists(path_save):
            with open(path_save, 'rb') as f:
                data = pickle.load(f)
            return data

        data = {}
        for path_image in self.path_images:
            image = read(path_image, **PlotInfo.READ_OPTIONS)
            x, species_gz = self._get_gz(image)
            z_max = np.max(image.get_positions()[:, 2])
            self._normalize(x, species_gz, z_max)
            data[path_image] = (x, species_gz, z_max)
            print(f"Finished {path_image}")

        with open(path_save, 'wb') as f:
            pickle.dump(data, f)
        return data

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
    def _cal_atomic_gz(z, plt_x_axis, sigma):
        return np.exp(-((z - plt_x_axis)/sigma)**2/2.) / np.sqrt(2*np.pi) / sigma

    @staticmethod
    def _normalize(x, species_gz, z_max):
        mask = x <= z_max
        species_gz[~mask] = 0
        row_sums = np.sum(species_gz[mask], axis=1, keepdims=True)
        species_gz[mask] /= row_sums

class GZPlotterAnimated:
    def __init__(self):
        self.elem_dict = PlotInfo.ELEM_DICT
        self.elem_group_dict = PlotInfo.ELEM_GROUP_DICT
        self.z_cut = PlotInfo.Z_CUT

    def run(self, data):
        fig, (ax_species, ax_group) = plt.subplots(1, 2, figsize=(10, 6))

        species_lines = {elem: ax_species.plot([], [], label=elem)[0] for elem in self.elem_dict.keys()}
        group_lines = {group: ax_group.plot([], [], label=group)[0] for group in self.elem_group_dict.keys()}

        for ax in [ax_species, ax_group]:
            ax.set_xlabel("$g(z)$")
            ax.set_ylabel("Height ($\AA$)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, self.z_cut)
            ax.legend(loc="upper right")

        def init():
            for line in species_lines.values():
                line.set_data([], [])
            for line in group_lines.values():
                line.set_data([], [])
            return list(species_lines.values()) + list(group_lines.values())

        def update(frame):
            path_image = list(data.keys())[frame]
            x, species_gz, z_max = data[path_image]

            for elem, idx in self.elem_dict.items():
                species_lines[elem].set_data(species_gz[:, idx], x)

            for group, idx in self.elem_group_dict.items():
                group_lines[group].set_data(np.sum(species_gz[:, idx], axis=1), x)

            for ax in [ax_species, ax_group]:
                ax.axhline(z_max, color='grey', linestyle='--', lw=0.5)

            return list(species_lines.values()) + list(group_lines.values())

        ani = FuncAnimation(
            fig, update, frames=len(data), init_func=init, blit=True, interval=10
        )

        ani.save("gz_animation.gif", writer="pillow", dpi=200)
        plt.show()


class AtomicDensityPlotter:
    @staticmethod
    def run(path_images):
        calculator = GZCalculator(path_images)
        data = calculator.run()

        plotter = GZPlotterAnimated()
        plotter.run(data)


if __name__ == '__main__':
    matplotlib.use('Agg')

    path_root_dir = sys.argv[1]
    path_images = []
    for root, dirs, files in os.walk(path_root_dir):
        for file in files:
            if not file.endswith("_after_mod.coo"):
                continue

            path_image = os.path.join(root, file)
            path_images.append(path_image)
    path_images = sorted(path_images, key=lambda x: int(x.split('/')[-1].split('_')[2]))

    AtomicDensityPlotter.run(path_images)
