import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ase.io import read
import pickle
import matplotlib.pyplot as plt

class SurfaceHeightAnalyzer:
    def __init__(self, src, n_traj=100, max_workers=4):
        self.src = src
        self.n_traj = n_traj
        self.max_workers = max_workers

    def _get_max_z(self, path_dump):
        image = read(path_dump, format='lammps-data', atom_style='atomic')
        max_z = np.max(image.get_positions()[:, 2])
        print(f"{path_dump} Done: {max_z}")
        return max_z

    def analyze_surface_height(self):
        path_dump_list = [
            f'{self.src}/CHF_shoot_{i}_after_removing.coo'
            for i in range(1, self.n_traj+1)
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            self.z_values = list(executor.map(self._get_max_z, path_dump_list))

    def plot(self, filename='max_z_profile.png'):
        x = np.arange(len(self.z_values))

        fig, ax = plt.subplots()
        ax.plot(x, self.z_values)
        ax.axhline(self.z_values[0], color='grey', linestyle='--')
        ax.set_xlabel('Trajectory')
        ax.set_ylabel('Max Z (Å)')
        ax.set_title('Surface Height Profile')

        fig.tight_layout()
        fig.savefig(filename)

    def run(self):
        self.analyze_surface_height()
        self.plot()


if __name__ == "__main__":
    src = sys.argv[1]
    analyzer = SurfaceHeightAnalyzer(src)
    analyzer.run()
