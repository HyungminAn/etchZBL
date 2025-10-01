import os
import numpy as np
import matplotlib.pyplot as plt

class TemperatureAnalyzer:
    def __init__(self, n_traj=150):
        self.n_traj = n_traj
        self.temperatures = None

    def read_data(self):
        temperatures = []
        for i in range(1, self.n_traj + 1):
            mat = np.loadtxt(f'thermo_{i}.dat', skiprows=2, usecols=(1, 2))
            time, temp = mat[:, 0], mat[:, 1]
            idx_nve_end = np.searchsorted(time, 2.0) - 1
            temperatures.append(temp[idx_nve_end])
            print(f'thermo_{i}.dat Complete')

        self.temperatures = np.array(temperatures)

    def plot(self):
        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots(figsize=(10, 5))

        T_m = 1983
        ax.plot(self.temperatures, label='T_after_NVE (2 ps)', color='k')
        ax.axhline(T_m, linestyle='--', color='blue', label=f'SiO2 T_m ({T_m} K)')
        ax.axhline(T_m * 0.9, linestyle='--', color='grey', label='90% of T_m')

        ax.legend(loc='upper right')
        ax.set_xlim((0, self.n_traj))
        ax.set_xlabel('number of incident CF3')
        ax.set_ylabel('Temperature (K)')

        title = os.path.basename(os.getcwd())
        ax.set_title(title)

        fig.tight_layout()
        fig.savefig('T_after_NVE.png')

    def run(self):
        self.read_data()
        self.plot()

if __name__ == "__main__":
    analyzer = TemperatureAnalyzer()
    analyzer.run()
