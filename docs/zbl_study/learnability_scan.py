"""Q1 experiment: can a model whose radial inputs flatten as r -> 0 learn data
that keeps rising as r -> 0?  -> learnability.json + figures/learnability.svg

Dimer only, so any atom-centred or message-passing model reduces to
E(r) = Phi(features(r)):
  * "G2"     : the 8 Si-Si G2 functions of the SiO2 SIMPLE-NN potential (Rs = 0, rc = 6),
               tanh MLP 8-30-30-1 per atom (BPNN-like)
  * "Bessel" : the 8 trained Bessel functions of the SevenNet checkpoint,
               SiLU MLP 8-64-64-1 (stand-in for any GNN on a dimer)
Targets: "full"  = a DFT-like reference that follows ZBL at short range
         "delta" = full - (ZBL + S) with the repo's Si-Si inner/outer (1.598/2.054)
Chain-rule bound used for the prediction:
  |dE/dr| <= ||dE/dG|| * ||dG/dr||   ->   required L(r) = |F_ref(r)| / ||dG/dr||
Run: python3 learnability_scan.py   (needs torch; ~3 min on CPU)
"""
import json
import os

import numpy as np
import torch

from make_figures import HERE, zbl, zbl_sw
from make_figures2 import POTS, sevennet_constants
import simplenn_eval as sn

torch.set_default_dtype(torch.float64)
_, NETS = sn.load(POTS["SiO2"])
ETA = np.array([float(s[2]) for s in NETS["Si"]["sym"] if s[0] == "2" and s[5] == "Si"])
C7 = sevennet_constants()
CN = np.array(C7["bessel_freq"])


def features(kind, r):
    r = torch.as_tensor(r)
    if kind == "G2":
        fc = 0.5 * (1 + torch.cos(np.pi * r / 6.0))
        return torch.exp(-torch.as_tensor(ETA)[None, :] * r[:, None] ** 2) * fc[:, None]
    return 0.4 * torch.sin(torch.as_tensor(CN)[None, :] * r[:, None]) / r[:, None]


def reference(r):
    """DFT-like Si-Si dimer: 1.05 x ZBL at short range, Morse bond near 2.3 A."""
    w = 1 / (1 + np.exp(-8.0 * (r - 1.9)))
    D, a, re = 3.0, 1.6, 2.35
    x = np.exp(-a * (r - re))
    Em, dEm = D * (x * x - 2 * x), D * (-2 * a * x * x + 2 * a * x)
    Ez, dEz, _ = zbl(r, 14, 14)
    dw = 8.0 * w * (1 - w)
    E = (1 - w) * 1.05 * Ez + w * Em
    dE = (1 - w) * 1.05 * dEz + w * dEm + dw * (Em - 1.05 * Ez)
    return E, -dE


def make_model(kind):
    if kind == "G2":
        return torch.nn.Sequential(torch.nn.Linear(8, 30), torch.nn.Tanh(), torch.nn.Linear(30, 30), torch.nn.Tanh(), torch.nn.Linear(30, 1))
    return torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1))


def fit(kind, target, r_min, steps=3000, seed=0):
    torch.manual_seed(seed)
    r = np.linspace(r_min, 4.0, 300)
    E, F = reference(r)
    if target == "delta":
        Es, dEs = zbl_sw(r, 14, 14, 1.598, 2.054)
        E, F = E - Es, F + dEs
    rt = torch.tensor(r, requires_grad=True)
    G = features(kind, rt)
    mu, sd = G.detach().mean(0), G.detach().std(0) + 1e-12
    model = make_model(kind)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Et, Ft = torch.tensor(E), torch.tensor(F)
    sE, sF = Et.std() + 1e-9, Ft.std() + 1e-9          # scale-free loss so both targets are fitted alike

    def predict(rr):
        g = features(kind, rr)
        e = 2 * model((g - mu) / sd).squeeze(-1)
        f = -torch.autograd.grad(e.sum(), rr, create_graph=True)[0]
        return e, f

    for _ in range(steps):
        opt.zero_grad()
        e, f = predict(rt)
        loss = (((e - Et) / sE) ** 2).mean() + (((f - Ft) / sF) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    e, f = predict(rt)
    # Lipschitz constant of the fitted map in raw-feature units, max over the data
    Gq = features(kind, torch.tensor(r)).detach().requires_grad_(True)
    eq = 2 * model((Gq - mu) / sd).squeeze(-1)
    dEdG = torch.autograd.grad(eq.sum(), Gq)[0].norm(dim=1)
    rr = torch.tensor(r, requires_grad=True)
    gg = features(kind, rr)
    dGdr = torch.stack([torch.autograd.grad(gg[:, k].sum(), rr, retain_graph=True)[0] for k in range(8)], 1).norm(dim=1)
    err_f = (f - Ft).detach().numpy()
    return {"r": r.tolist(), "E_ref": E.tolist(), "E_fit": e.detach().numpy().tolist(),
            "F_ref": F.tolist(), "F_fit": f.detach().numpy().tolist(),
            "rmse_E": float(np.sqrt(np.mean((e.detach().numpy() - E) ** 2))),
            "rmse_F": float(np.sqrt(np.mean(err_f ** 2))),
            "L_fit": float(dEdG.max()), "L_req": float(np.max(np.abs(F) / dGdr.detach().numpy())),
            "dGdr": dGdr.detach().numpy().tolist()}


def required_L(kind, r):
    rr = torch.tensor(r, requires_grad=True)
    gg = features(kind, rr)
    dGdr = torch.stack([torch.autograd.grad(gg[:, k].sum(), rr, retain_graph=True)[0] for k in range(8)], 1).norm(dim=1).detach().numpy()
    E, F = reference(r)
    Es, dEs = zbl_sw(r, 14, 14, 1.598, 2.054)
    return dGdr, np.abs(F) / (2 * dGdr), np.abs(F + dEs) / (2 * dGdr)


RMINS = [1.4, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.12]


def run_part(kind, target):
    torch.set_num_threads(2)
    out = {}
    for rm in RMINS:
        res = fit(kind, target, rm)
        out[f"{kind}|{target}|{rm}"] = res
        print(f"{kind:6} {target:5} r_min={rm:4}: RMSE_E={res['rmse_E']:9.3f} eV  RMSE_F={res['rmse_F']:9.2f} eV/A  "
              f"L_fit={res['L_fit']:9.1f}  L_req={res['L_req']:10.1f}", flush=True)
    with open(os.path.join(HERE, f"_learn_{kind}_{target}.json"), "w") as fh:
        json.dump(out, fh)


def merge():
    rmins = RMINS
    out = {"rmins": rmins, "fits": {}}
    for kind in ["G2", "Bessel"]:
        for target in ["full", "delta"]:
            fn = os.path.join(HERE, f"_learn_{kind}_{target}.json")
            out["fits"].update(json.load(open(fn)))
            os.remove(fn)
    r = np.linspace(0.05, 4.0, 400)
    for kind in ["G2", "Bessel"]:
        dG, Lf, Ld = required_L(kind, r)
        out[f"Lreq_{kind}"] = {"r": r.tolist(), "dGdr": dG.tolist(), "full": Lf.tolist(), "delta": Ld.tolist()}
    with open(os.path.join(HERE, "learnability.json"), "w") as fh:
        json.dump(out, fh, separators=(",", ":"))


if __name__ == "__main__":
    # python3 learnability_scan.py            -> all four parts in parallel, then merge
    # python3 learnability_scan.py G2 full    -> one part
    import subprocess
    import sys
    if len(sys.argv) == 3:
        run_part(sys.argv[1], sys.argv[2])
    else:
        procs = [subprocess.Popen([sys.executable, __file__, k, t]) for k in ("G2", "Bessel") for t in ("full", "delta")]
        for pr in procs:
            pr.wait()
        merge()
