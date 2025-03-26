import os
import sys
import pickle
import matplotlib.pyplot as plt
from typing import List, Tuple

class TimeElapsedPlotter:
    def __init__(self, src: str, n_incidence):
        self.src = src
        self.n_incidence = n_incidence
        self.path_data_save = f"data_{os.path.basename(src)}.pkl"

    def run(self):
        data_n_atoms, data_time = self._get_data()
        self._plot(data_n_atoms, data_time)

    @staticmethod
    def time_to_hours(time_str: str) -> float:
        h, m, s = map(int, time_str.split(':'))
        return h + m / 60 + s / 3600

    def _get_data(self) -> Tuple[List[int], List[float]]:
        print(f"Loading data from {self.src}...", end=' ')
        if os.path.exists(self.path_data_save):
            with open(self.path_data_save, 'rb') as f:
                data = pickle.load(f)
            print("Done.")
            return data

        data_n_atoms, data_time = [], []
        for i in range(1, self.n_incidence+1):
            coo_file = f"{self.src}/CHF_shoot_{i}.coo"
            log_file = f"{self.src}/log_{i}.lammps"

            with open(coo_file, 'r') as f:
                n_atoms = int(f.readlines()[2].split()[0])

            with open(log_file, 'r') as f:
                time = f.readlines()[-1].split()[-1]
                if ':' not in time:
                    continue
                time = self.time_to_hours(time)

            data_n_atoms.append(n_atoms)
            data_time.append(time)
            print(i, n_atoms, time)

        data = (data_n_atoms, data_time)
        with open(self.path_data_save, 'wb') as f:
            pickle.dump(data, f)

        print("Done.")
        return data

    @staticmethod
    def _plot(data_n_atoms: List[int], data_time: List[float]):
        print("Plotting...", end=' ')
        fig, ax = plt.subplots()
        x = range(len(data_n_atoms))

        ax.plot(x, data_n_atoms, label="n_atoms", color='blue', alpha=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("n_atoms")

        ax2 = ax.twinx()
        ax2.bar(x, data_time, label="time", color='red', alpha=0.5)
        ax2.set_ylabel("Time (h)")

        total_time = sum(data_time)
        ax.set_title(f"Total time: {total_time:.2f} hours")
        fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))

        fig.tight_layout()
        fig.savefig("time_spent.png")
        print("Done.")

if __name__ == "__main__":
    src = sys.argv[1]
    n_incidence = int(sys.argv[2])
    analyzer = TimeElapsedPlotter(src, n_incidence)
    analyzer.run()
