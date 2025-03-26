import sys
import numpy as np
import matplotlib.pyplot as plt

class MonitorAtomsPlotter:
    def __init__(self, src, n_traj):
        self.src = src
        self.n_traj = n_traj
        self.time_col = 1
        self.elements = {
            'Si': 6,
            'O': 7,
            'C': 8,
            'H': 9,
            'F': 10
        }

    def run(self):
        data = self._read_data()
        self._plot(data)

    def _read_data(self):
        data = {elem: [] for elem in self.elements}
        data['time'] = []
        time_shift = 0

        for i in range(1, self.n_traj + 1):
            cols = [self.time_col] + list(self.elements.values())
            mat = np.loadtxt(f'{self.src}/thermo_{i}.dat', skiprows=2, usecols=cols)

            time = mat[:, 0] + time_shift
            time_shift += mat[-1, 0]
            print(time_shift, time[-1])
            data['time'].append(time)
            for j, elem in enumerate(self.elements):
                data[elem].append(mat[:, j+1])

            print(f"Iteration {i} finished")

        return {k: np.concatenate(v) for k, v in data.items()}

    def _plot(self, data):
        fig, ax = plt.subplots()
        for elem in self.elements:
            ax.plot(data['time'], data[elem], label=elem)

        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Number of atoms')
        ax.legend(loc='upper right')
        ax.set_title(self.src)

        fig.tight_layout()
        fig.savefig('number_of_atoms.png')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python monitor.py <src> <n_traj>")
        sys.exit(1)
    src = sys.argv[1]
    n_traj = int(sys.argv[2])

    plotter = MonitorAtomsPlotter(src, n_traj)
    plotter.run()
