"""SevenNet checkpoint scans for docs/zbl_study -> sevennet_scan.json.

Needs torch (CPU is fine). Runs pot/SevenNet/pot_7net_chg_tot_l3i3_zbl_a0.104.pt
(TorchScript, LAMMPS deploy format) on non-periodic clusters:
  * dimers: total energy (ML x switch + ZBL) and the raw ML energy
  * r -> 0 limit of the raw ML energy
  * Si17 cluster: one atom pushed onto the centre, forces on the 15 spectators
    for SevenNet (total and raw ML) and for the SIMPLE-NN Si3N4 BPNN
  * the part of the spectator force that comes from the switching weight
Run: python3 sevennet_scan.py   (about 3 min; the BPNN part dominates)
"""
import json
import os
import warnings

import numpy as np
import torch

import simplenn_eval as sn
from make_figures import HERE, zbl_sw
from make_figures2 import POTS, SEVENNET, sevennet_constants

warnings.filterwarnings("ignore")
EXTRA = {"chemical_symbols_to_index": b"", "cutoff": b""}
MODEL = torch.jit.load(SEVENNET, _extra_files=EXTRA, map_location="cpu").eval()
SYMS = EXTRA["chemical_symbols_to_index"].decode().split()
RC = float(EXTRA["cutoff"])
C7 = sevennet_constants()
CM = 5 / (2 * 99 ** 0.23 / 0.4685)          # cutoff_max of the softmin, reduced units


def evaluate(pos, symbols):
    pos = np.asarray(pos, float)
    src, dst = [], []
    for i in range(len(pos)):
        for j in range(len(pos)):
            if i != j and np.linalg.norm(pos[j] - pos[i]) < RC:
                src.append(i)
                dst.append(j)
    types = torch.tensor([SYMS.index(s) for s in symbols], dtype=torch.long)
    out = MODEL({"pos": torch.tensor(pos, dtype=torch.float32), "edge_index": torch.tensor([src, dst], dtype=torch.long),
                 "atom_type": types, "x": types.clone(), "num_atoms": torch.tensor([len(pos)]),
                 "cell_lattice_vectors": torch.eye(3) * 100, "pbc_shift": torch.zeros(len(src), 3),
                 "cell_volume": torch.tensor(1e6)})
    e_ml = C7["energy_scale"] * out["scaled_atomic_energy"].detach().numpy().ravel() + np.array([C7["shift"][s] for s in symbols])
    return float(out["inferred_total_energy"].detach()), out["inferred_force"].detach().numpy(), e_ml


def f_atom(pos, Z, i, alpha=0.104):
    """Switching weight of atom i, reconstructed from the checkpoint code."""
    num = den = 0.0
    for j in range(len(pos)):
        if j == i:
            continue
        rb = np.linalg.norm(pos[j] - pos[i]) / ((Z[i] ** 0.23 + Z[j] ** 0.23) / 0.4685)
        if rb > CM:
            continue
        cv = 1.0 if rb < 0.139 else (CM**2 - rb**2) ** 2 * (CM**2 + 2 * rb**2 - 3 * 0.139**2) / (CM**2 - 0.139**2) ** 3
        w = cv * np.exp(-rb / alpha)
        num += rb * w
        den += w
    sigma = num / den if num else 0.139
    t = min(max((sigma - 0.069) / 0.07, 0), 1)
    return sigma, t**5 * (70 * t**4 - 315 * t**3 + 540 * t**2 - 420 * t + 126)


def dimers():
    r = np.concatenate([np.linspace(0.02, 0.5, 25), np.linspace(0.52, 5.2, 118)])
    out = {"r": [round(float(x), 4) for x in r], "pairs": {}, "limit": {}}
    Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14}
    for a, b in [("Si", "Si"), ("Si", "F"), ("Si", "O"), ("F", "F"), ("C", "F"), ("N", "N"), ("H", "H")]:
        E0, _, m0 = evaluate([[0, 0, 0], [20, 0, 0]], [a, b])
        tot, ml = [], []
        for x in r:
            E, _, m = evaluate([[0, 0, 0], [x, 0, 0]], [a, b])
            tot.append(round(E - E0, 4))
            ml.append(round(float(m.sum() - m0.sum()), 4))
        s = (Z[a] ** 0.23 + Z[b] ** 0.23) / 0.4685
        out["pairs"][f"{a}-{b}"] = {"Z": [Z[a], Z[b]], "E_tot": tot, "E_ml": ml, "window": [round(0.069 * s, 3), round(0.139 * s, 3)]}
        out["limit"][f"{a}-{b}"] = [[x, round(float(evaluate([[0, 0, 0], [x, 0, 0]], [a, b])[2].sum() - m0.sum()), 3)]
                                    for x in (0.3, 0.1, 0.03, 0.01, 1e-3, 3e-4)]
    return out


def si17():
    a = 5.431
    first = np.array([[0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) * a / 4
    second = np.array([(2, 2, 0), (2, -2, 0), (-2, 2, 0), (-2, -2, 0), (2, 0, 2), (2, 0, -2),
                       (-2, 0, 2), (-2, 0, -2), (0, 2, 2), (0, 2, -2), (0, -2, 2), (0, -2, -2)]) * a / 4
    return np.vstack([first, second])


def spill():
    P0 = si17()
    N = len(P0)
    S, Z = ["Si"] * N, [14] * N
    u = P0[1] / np.linalg.norm(P0[1])
    spect = list(range(2, N))
    _, nets = sn.load(POTS["Si3N4"])

    def fd(E, p, h):
        F = np.zeros((len(spect), 3))
        for n, k in enumerate(spect):
            for c in range(3):
                q = p.copy()
                q[k, c] += h
                ep = E(q)
                q[k, c] -= 2 * h
                F[n, c] = -(ep - E(q)) / (2 * h)
        return F

    e_ml = lambda p: float(evaluate(p, S)[2].sum())
    e_bp = lambda p: sum(sn.energy_and_dEdG(nets["Si"], sn.descriptors(nets["Si"], i, p, S))[0] for i in range(N))
    rows, ref = [], None
    for r in [2.35, 1.8, 1.4, 1.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.12]:
        p = P0.copy()
        p[1] = r * u
        E, F, m = evaluate(p, S)
        cur = {"tot": F[spect], "ml": fd(e_ml, p, 2e-3), "bp": fd(e_bp, p, 1e-4)}
        sw = np.zeros((N, 3))
        for i in (0, 1):
            for k in spect:
                for c in range(3):
                    q = p.copy()
                    q[k, c] += 1e-5
                    fp = f_atom(q, Z, i)[1]
                    q[k, c] -= 2e-5
                    sw[k, c] -= m[i] * (fp - f_atom(q, Z, i)[1]) / 2e-5
        ref = ref or cur
        d = lambda key: round(float(np.max(np.linalg.norm(cur[key] - ref[key], axis=1))), 3)
        sig, f0 = f_atom(p, Z, 0)
        rows.append({"r": r, "dF_tot": d("tot"), "dF_ml": d("ml"), "dF_bpnn": d("bp"),
                     "F_switch": round(float(np.max(np.linalg.norm(sw[spect], axis=1))), 3),
                     "f0": round(f0, 4), "sigma0": round(sig, 4), "E_ml0": round(float(m[0]), 3)})
        print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    res = {"dimers": dimers(), "spill": spill()}
    for k, v in res["dimers"]["limit"].items():
        print(k, v)
    with open(os.path.join(HERE, "sevennet_scan.json"), "w") as fh:
        json.dump(res, fh, separators=(",", ":"))
