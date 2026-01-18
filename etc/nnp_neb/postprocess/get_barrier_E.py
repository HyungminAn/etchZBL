import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
"""
Usage : python -.py log.lammps
"""


def main(path_log):
    with open(path_log, 'r') as f:
        line = f.readlines()[-1]
        line = line.split()[9:]

    x = [float(i) for i in line[0::2]]
    y_abs = [float(i) for i in line[1::2]]
    y = [i-y_abs[0] for i in y_abs]  # relative energy
    Eb = max(y)
    dE = y[-1]

    matplotlib.use('Agg')
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(x, y, marker='o', color='k', linestyle='--')
    ax.set_xlim((0, 1))
    ax.set_xlabel("Reaction Coordinate")
    ax.set_ylabel("Relative energy (eV)")
    ax.set_title(f"$E^{{\ddag}}$: {Eb:.2f} eV, $\Delta E$: {dE:.2f} eV")
    ax.axhline(0, color='grey')

    fig.tight_layout()
    fig.savefig('NEBresult.png')

    mat = np.array([x, y]).transpose()
    np.savetxt('e_nnp.dat', mat)
    mat = np.array([x, y_abs]).transpose()
    np.savetxt('e_nnp_abs.dat', mat)


main(sys.argv[1])
