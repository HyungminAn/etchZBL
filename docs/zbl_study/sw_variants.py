"""Q4: how much of the spectator force from the SevenNet switch survives
(a) a smaller softmin alpha, (b) referencing E_i^ML to the isolated atom,
(c) an ACE-style switch on the hard nearest-neighbour distance.
Uses the checkpoint's own E_i^ML on the Si17 path; writes sw_variants.json."""
import json
import os

import numpy as np

from sevennet_scan import evaluate, f_atom, si17, HERE

P0 = si17()
N = len(P0)
Z = [14] * N
u = P0[1] / np.linalg.norm(P0[1])
E_iso = float(evaluate([[0, 0, 0]], ["Si"])[2][0])


def spect_force(p, Eml, alpha, ref=0.0):
    sw = np.zeros((N, 3))
    for i in (0, 1):
        for k in range(2, N):
            for c in range(3):
                q = p.copy()
                q[k, c] += 1e-5
                fp = f_atom(q, Z, i, alpha)[1]
                q[k, c] -= 2e-5
                sw[k, c] -= (Eml[i] - ref) * (fp - f_atom(q, Z, i, alpha)[1]) / 2e-5
    return float(np.max(np.linalg.norm(sw[2:], axis=1)))


rows = []
for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
    p = P0.copy()
    p[1] = r * u
    _, _, Eml = evaluate(p, ["Si"] * N)
    row = {"r": r, "E_ml0": round(float(Eml[0]), 2)}
    for a in (0.104, 0.05, 0.03, 0.02):
        row[f"a{a}"] = round(spect_force(p, Eml, a), 3)
    row["ref_iso"] = round(spect_force(p, Eml, 0.104, E_iso), 3)
    row["hardmin"] = 0.0          # d f_i / d x_k = 0 for every k that is not the nearest neighbour
    rows.append(row)
    print(row, flush=True)
json.dump({"E_iso": E_iso, "rows": rows}, open(os.path.join(HERE, "sw_variants.json"), "w"))
