"""Part 2 figures + demo_data.json for docs/zbl_study (BPNN bounds, radial vs
angular force, atom-level switching, GNN radial basis, PES softening toy).

Run: python3 make_figures2.py   (needs numpy, matplotlib)
Reads the SIMPLE-NN potentials in pot/simplenn and the SevenNet deploy file's
constants (without torch). Writes figures/*.svg and demo_data.json.
"""
import io
import json
import os
import pickle
import zipfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import zbl, zbl_sw, COL, FIG, HERE
import simplenn_eval as sn

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
POTS = {"Si3N4": os.path.join(ROOT, "pot/simplenn/Si3N4/pot_SiNCHF_new"),
        "SiO2": os.path.join(ROOT, "pot/simplenn/SiO2/potential_saved_bestmodel")}
SEVENNET = os.path.join(ROOT, "pot/SevenNet/pot_7net_chg_tot_l3i3_zbl_a0.104.pt")
Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14}
# inner/outer used for the residuals (generate_trainset/.../get_residual_new_parallel.py);
# N pairs from etc/nurion/test_case/01_3atoms/lammps_oneshot.in
CUT = {"SiSi": (1.598, 2.054), "SiN": (1.104, 1.42), "SiC": (1.208, 1.553), "SiH": (1.079, 1.388),
       "SiF": (1.145, 1.472), "SiO": (1.070, 1.376), "NN": (0.78, 1.002), "NC": (1.104, 1.42),
       "NH": (0.735, 0.945), "NF": (0.938, 1.206), "CC": (0.92, 1.183), "CH": (0.797, 1.024),
       "CF": (0.908, 1.168), "CO": (0.8, 1.029), "HH": (0.525, 0.675), "HF": (0.657, 0.844),
       "FF": (0.997, 1.282), "OO": (0.863, 1.110), "OF": (0.961, 1.235), "OH": (0.691, 0.888)}


def cut(a, b):
    return CUT.get(a + b) or CUT.get(b + a)


def rnd(x, n=4):
    return [float(f"{v:.{n}g}") for v in np.asarray(x).ravel()]


# --------------------------------------------------------------- BPNN dimers
def dimer_curves():
    r = np.linspace(0.05, 4.0, 800)
    out = {}
    for pot, path in POTS.items():
        elems, nets = sn.load(path)
        d = {"elements": elems, "bound": {}, "pairs": {}}
        for e in elems:
            s, b = sn.bound(nets[e])
            d["bound"][e] = [round(float(s), 3), round(float(b), 3)]
        for i, a in enumerate(elems):
            for b in elems[i:]:
                c = cut(a, b)
                if c is None:
                    continue
                Ea = sn.forward(nets[a], sn.G_dimer(nets[a], r, b))
                Eb = sn.forward(nets[b], sn.G_dimer(nets[b], r, a))
                E = Ea + Eb
                Einf = (sn.forward(nets[a], np.zeros((1, len(nets[a]["sym"])))) +
                        sn.forward(nets[b], np.zeros((1, len(nets[b]["sym"])))))[0]
                d["pairs"][f"{a}-{b}"] = {"Z": [Z[a], Z[b]], "cut": list(c),
                                          "E": rnd(E - Einf)}
        out[pot] = d
    return {"r": rnd(r), "pots": out}


def folded_net(pot="SiO2", el="Si"):
    """Fold scaling + PCA into the first tanh layer so the browser can run it."""
    elems, nets = sn.load(POTS[pot])
    net = nets[el]
    (W0, b0, _), (W1, b1, _), (W2, b2, _), (W3, b3, _) = net["layers"]
    A = W1 @ W0 / net["s1"][None, :]
    c = W1 @ (b0 - W0 @ (net["s0"] / net["s1"])) + b1
    # a typical input: Si in an SiO4 tetrahedron
    t = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    pos = np.vstack([[0, 0, 0], 1.61 * t])
    G0 = sn.descriptors(net, 0, pos, ["Si", "O", "O", "O", "O"])
    rng = np.random.default_rng(3)
    d = rng.normal(size=G0.shape) * net["s1"]
    d /= np.linalg.norm(d / net["s1"])
    return {"A": [rnd(row, 5) for row in A], "c": rnd(c, 6), "W2": [rnd(row, 5) for row in W2],
            "b2": rnd(b2, 6), "W3": rnd(W3[0], 6), "b3": float(b3[0]), "G0": rnd(G0, 6), "dir": rnd(d, 5),
            "s1": rnd(net["s1"], 6), "bound": float(np.abs(W3).sum()),
            "sym": [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])] + x[5:] for x in net["sym"]]}


def G_tetra(f, r):
    """Descriptors of Si with four O at distance r in a perfect tetrahedron."""
    G = np.zeros(len(f["sym"]))
    rjk = r * np.sqrt(8 / 3)
    fcut = lambda d, rc: 0.5 * (1 + np.cos(np.pi * d / rc)) if d < rc else 0.0
    for m, s in enumerate(f["sym"]):
        st, rc, eta = s[0], s[1], s[2]
        if st == 2:
            if s[5] == "O":
                G[m] = 4 * np.exp(-eta * (r - s[3]) ** 2) * fcut(r, rc)
            continue
        if not (s[5] == "O" and s[6] == "O"):
            continue
        zeta, lam = s[3], s[4]
        ang = 2 ** (1 - zeta) * abs(1 - lam / 3) ** zeta
        if st == 4:
            G[m] = 6 * ang * np.exp(-eta * (2 * r * r + rjk * rjk)) * fcut(r, rc) ** 2 * fcut(rjk, rc)
        else:
            G[m] = 6 * ang * np.exp(-eta * 2 * r * r) * fcut(r, rc) ** 2
    return G


def run_folded(f, G, act="tanh"):
    A, c = np.array(f["A"]), np.array(f["c"])
    W2, b2, W3 = np.array(f["W2"]), np.array(f["b2"]), np.array(f["W3"])
    g = (lambda x: np.tanh(x)) if act == "tanh" else (lambda x: x / (1 + np.exp(-x)))
    h = g(G @ A.T + c)
    h = g(h @ W2.T + b2)
    return h @ W3 + f["b3"]


# --------------------------------------------------------------- cluster test
def cluster_SiO4F():
    elems, nets = sn.load(POTS["SiO2"])
    t = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    O = 1.61 * t
    axis = -t[0]
    perp = np.cross(axis, [0, 0, 1])
    perp /= np.linalg.norm(perp)
    u = np.cos(np.radians(20)) * axis + np.sin(np.radians(20)) * perp
    rows = []
    for r in [2.4, 2.0, 1.7, 1.5, 1.3, 1.1, 0.9, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12]:
        pos = np.vstack([[0, 0, 0], O, r * u])
        E, Fr, Fa = sn.decompose(nets, pos, ["Si", "O", "O", "O", "O", "F"])
        Fz = -float(zbl_sw(r, 14, 9, 1.145, 1.472)[1])
        par = lambda v: float(v @ u)
        perpn = lambda v: float(np.linalg.norm(v - (v @ u) * u))
        rows.append({"r": r, "E": round(float(E), 4),
                     "Frad_par": round(par(Fr[5]), 3), "Frad_perp": round(perpn(Fr[5]), 3),
                     "Fang_par": round(par(Fa[5]), 3), "Fang_perp": round(perpn(Fa[5]), 3),
                     "F_NN": round(float(np.linalg.norm(Fr[5] + Fa[5])), 3), "F_zbl": round(Fz, 2),
                     "F_O_max": round(float(max(np.linalg.norm(Fr[k] + Fa[k]) for k in range(1, 5))), 3)})
    return rows


# --------------------------------------------------------------- SevenNet constants
def sevennet_constants():
    z = zipfile.ZipFile(SEVENNET)
    pre = z.namelist()[0].split("/")[0]
    DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64}

    class St:
        def __init__(self, dt, key):
            self.dt, self.key = dt, key

    def rebuild(st, off, size, stride, *a):
        arr = np.frombuffer(z.read(f"{pre}/constants/{st.key}"), dtype=st.dt)
        if not len(size):
            return arr[off]
        return np.lib.stride_tricks.as_strided(arr[off:], shape=size,
                                               strides=[s * arr.itemsize for s in stride]).copy()

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name.startswith("_rebuild_tensor"):
                return rebuild
            if name in DT:
                return name
            if mod.startswith("torch") or mod.startswith("collections"):
                return lambda *a, **k: None
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            _, st, key, loc, n = pid
            return St(DT[st], key)

    c = U(io.BytesIO(z.read(f"{pre}/constants.pkl"))).load()
    syms = z.read(f"{pre}/extra/chemical_symbols_to_index").decode().split()
    shift = {e: round(float(c[65][syms.index(e)]), 3) for e in ["H", "C", "N", "O", "F", "Si"]}
    return {"bessel_freq": rnd(c[0], 5), "rc": 5.0, "prefactor": 0.4, "cutoff_on": 4.5,
            "energy_scale": round(float(c[64][0]), 4), "shift": shift,
            "sw": {"alpha": 0.104, "inner": 0.069, "outer": 0.139}}


# --------------------------------------------------------------- toys
def bessel(r, freq, pref=0.4):
    r = np.asarray(r, float)[:, None]
    return pref * np.sin(freq[None, :] * r) / r


CM7 = 5 / (2 * 99 ** 0.23 / 0.4685)    # softmin cutoff (reduced units) in the checkpoint


def softmin_switch(dists, Zs, alpha=0.104, inner=0.069, outer=0.139):
    """Atom-level switch used in the SevenNet checkpoint (DeePMD-style softmin)."""
    s = np.array([(Zs[0] ** 0.23 + z ** 0.23) / 0.4685 for z in Zs[1:]])
    rb = np.asarray(dists) / s
    cv = np.where(rb < outer, 1.0, (CM7**2 - rb**2) ** 2 * (CM7**2 + 2 * rb**2 - 3 * outer**2) / (CM7**2 - outer**2) ** 3)
    cv = np.where(rb > CM7, 0.0, cv)
    w = cv * np.exp(-rb / alpha)
    sigma = (rb * w).sum() / w.sum()
    tau = (sigma - inner) / (outer - inner)
    t = np.clip(tau, 0, 1)
    f = t**5 * (70 * t**4 - 315 * t**3 + 540 * t**2 - 420 * t + 126)
    return sigma, f


def softening_toy(frac_hi=0.05, lam=1e-2, n=200, seed=0):
    """Fit a smooth model to Morse data biased to near-equilibrium; report curvature ratio."""
    rng = np.random.default_rng(seed)
    D, a, re = 3.0, 1.6, 2.35
    V = lambda r: D * (np.exp(-2 * a * (r - re)) - 2 * np.exp(-a * (r - re))) + D
    n_hi = int(frac_hi * n)
    r_eq = re + rng.normal(0, 0.06, n - n_hi)
    r_hi = rng.uniform(1.75, 3.2, n_hi)
    x = np.concatenate([r_eq, r_hi])
    y = V(x) + rng.normal(0, 0.02, len(x))
    centers = np.linspace(1.6, 3.4, 16)
    Phi = lambda r: np.exp(-((r[:, None] - centers[None, :]) / 0.25) ** 2)
    P = Phi(x)
    w = np.linalg.solve(P.T @ P + lam * len(x) * np.eye(len(centers)), P.T @ y)
    rr = np.linspace(1.6, 3.4, 300)
    return rr, V(rr), Phi(rr) @ w, x, y


# --------------------------------------------------------------- figures
def hidden_saturation(f, G):
    A, c = np.array(f["A"]), np.array(f["c"])
    h1 = np.tanh(G @ A.T + c)
    h2 = np.tanh(h1 @ np.array(f["W2"]).T + np.array(f["b2"]))
    return (np.abs(h1) > 0.99).mean(axis=1), (np.abs(h2) > 0.99).mean(axis=1)


def fig_tanh_bound(f):
    r = np.linspace(0.05, 4.0, 400)
    G = np.array([G_tetra(f, x) for x in r])
    E_t = run_folded(f, G, "tanh")
    s1, s2 = hidden_saturation(f, G)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for a_ in ax:
        a_.axvspan(1.55, 1.70, color="#e7f5ff", zorder=0)
    ax[0].axhspan(f["b3"] - f["bound"], f["b3"] + f["bound"], color="#f1f3f5")
    ax[0].plot(r, E_t, color=COL["sw"])
    ax[0].axhline(f["b3"], color="#868e96", lw=0.8, ls=":")
    ax[0].text(3.0, f["b3"] + f["bound"] - 1.6, "b + Σ|w|", fontsize=8, color="#495057")
    ax[0].text(3.0, f["b3"] - f["bound"] + 0.6, "b − Σ|w|", fontsize=8, color="#495057")
    ax[0].set_ylim(f["b3"] - 1.4 * f["bound"], f["b3"] + 1.4 * f["bound"])
    ax[0].set_xlabel("Si–O distance in an SiO4 tetrahedron (Å)")
    ax[0].set_ylabel("atomic energy E_Si (eV)")
    ax[0].set_title("SiO2 potential, Si network (blue band: equilibrium)")
    ax[1].plot(r, s1 * 100, color=COL["mult"], label="hidden layer 1")
    ax[1].plot(r, s2 * 100, color=COL["fermi"], ls="--", label="hidden layer 2")
    ax[1].set_ylim(0, 100); ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_xlabel("Si–O distance (Å)"); ax[1].set_ylabel("units with |tanh| > 0.99 (%)")
    ax[1].set_title("compressed far below equilibrium: units saturate")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "tanh_bound.svg"))
    plt.close(fig)


def fig_g2():
    r = np.linspace(0, 3, 400)
    rc, eta = 6.0, 0.7
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for Rs, ls in [(0.0, "-"), (1.0, "--")]:
        e = np.exp(-eta * (r - Rs) ** 2)
        f = 0.5 * (1 + np.cos(np.pi * r / rc))
        df = -0.5 * np.pi / rc * np.sin(np.pi * r / rc)
        t1 = -2 * eta * (r - Rs) * e * f
        t2 = e * df
        ax[0].plot(r, e * f, ls=ls, color=COL["zbl"], label=f"G2, Rs={Rs}")
        ax[1].plot(r, t1, ls=ls, color=COL["sw"], label=f"−2η(r−Rs)·e·f_c, Rs={Rs}")
        ax[1].plot(r, t2, ls=ls, color=COL["mult"], label=f"e·f_c′, Rs={Rs}")
    ax[0].set_xlabel("r (Å)"); ax[0].set_title("G2 = exp(−η(r−Rs)²)·f_c(r),  η=0.7, rc=6")
    ax[1].axhline(0, color="#adb5bd", lw=0.8)
    ax[1].set_xlabel("r (Å)"); ax[1].set_title("dG2/dr = (term 1) + (term 2)")
    ax[0].legend(frameon=False, fontsize=8); ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "g2_derivative.svg"))
    plt.close(fig)


def angular_grad(rij, theta_deg=70, rik=2.0, eta=0.03, zeta=4, lam=1.0, rc=4.5):
    """|dG4/dx_j| split into the component along r_ij and perpendicular to it."""
    out = []
    th = np.radians(theta_deg)
    xk = np.array([rik, 0, 0])
    for r in rij:
        def G(xj):
            a, b = np.linalg.norm(xj), rik
            c = np.linalg.norm(xj - xk)
            cos = xj @ xk / a / b
            fcut = lambda d: 0.5 * (1 + np.cos(np.pi * d / rc)) if d < rc else 0
            return 2 ** (1 - zeta) * (1 + lam * cos) ** zeta * np.exp(-eta * (a * a + b * b + c * c)) * fcut(a) * fcut(b) * fcut(c)
        xj = r * np.array([np.cos(th), np.sin(th), 0])
        g = np.array([(G(xj + h) - G(xj - h)) / 2e-7 for h in np.eye(3) * 1e-7])
        u = xj / r
        out.append((abs(g @ u), np.linalg.norm(g - (g @ u) * u)))
    return np.array(out)


def fig_angular():
    r = np.logspace(-2, np.log10(2.0), 120)
    g = angular_grad(r)
    eta, rc = 0.7, 6.0
    g2 = np.abs(-2 * eta * r * np.exp(-eta * r * r) * 0.5 * (1 + np.cos(np.pi * r / rc))
                + np.exp(-eta * r * r) * (-0.5 * np.pi / rc) * np.sin(np.pi * r / rc))
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.loglog(r, g2, color=COL["zbl"], label="|dG2/dr|  (radial)  ∝ r")
    ax.loglog(r, g[:, 0], color=COL["mult"], label="G4: component along r_ij → 0")
    ax.loglog(r, g[:, 1], color=COL["sw"], label="G4: component ⊥ r_ij  ∝ 1/r")
    ax.loglog(r, 0.02 / r**2, color="#868e96", ls=":", label="reference slope 1/r² (ZBL force)")
    ax.set_xlabel("r_ij (Å)"); ax.set_ylabel("|∂G/∂x_j|  (1/Å)")
    ax.set_title("descriptor gradients as a neighbour approaches")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "angular_gradient.svg"))
    plt.close(fig)


def fig_cluster(rows):
    r = np.array([x["r"] for x in rows])
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.loglog(r, [x["F_zbl"] if x["F_zbl"] > 0 else np.nan for x in rows], color=COL["zbl"], marker="o", ms=3, label="ZBL+S on F (Si–F pair)")
    ax.loglog(r, [x["F_NN"] for x in rows], color=COL["sw"], marker="o", ms=3, label="|F_NN| on F (total)")
    ax.loglog(r, [abs(x["Frad_par"]) + 1e-3 for x in rows], color=COL["mult"], ls="--", label="|F_NN radial|, along approach")
    ax.loglog(r, [x["Fang_perp"] for x in rows], color=COL["fermi"], ls="--", label="|F_NN angular|, ⊥ approach")
    ax.loglog(r, [x["F_O_max"] for x in rows], color=COL["bad"], ls=":", label="max |F_NN| on spectator O")
    ax.set_xlabel("Si–F distance (Å)"); ax.set_ylabel("force (eV/Å)")
    ax.set_title("SiO2 BPNN: F approaching an SiO4 unit")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cluster_forces.svg"))
    plt.close(fig)


def fig_dimers(data):
    r = np.array(data["r"])
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (pot, pair) in zip(axs, [("SiO2", "Si-Si"), ("Si3N4", "N-N"), ("SiO2", "Si-F")]):
        p = data["pots"][pot]["pairs"][pair]
        E = np.array(p["E"])
        Es, _ = zbl_sw(r, *p["Z"], *p["cut"])
        ax.axvspan(*p["cut"], color="#f1f3f5")
        ax.plot(r, E, color=COL["sw"], label="E_NN (dimer)")
        ax.plot(r, Es, color=COL["zbl"], label="ZBL+S")
        ax.plot(r, E + Es, color=COL["ref"], ls="--", label="total")
        ax.set_ylim(-15, 40); ax.axhline(0, color="#adb5bd", lw=0.8)
        ax.set_title(f"{pot}: {pair}"); ax.set_xlabel("r (Å)")
    axs[0].set_ylabel("E (eV, vs separated atoms)")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bpnn_dimers.svg"))
    plt.close(fig)


def fig_softmin(sw):
    d = np.linspace(0.2, 2.4, 300)
    Zs = [14, 9, 8, 8, 8]          # Si with one F (moving) and three O at 1.61 Å
    sig, f = zip(*[softmin_switch([x, 1.61, 1.61, 1.61], Zs) for x in d])
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    s_SiF = (14**0.23 + 9**0.23) / 0.4685
    ax[0].plot(d, np.array(sig), color=COL["sw"], label="softmin σ_i (reduced)")
    ax[0].plot(d, np.minimum(d / s_SiF, 1.61 / ((14**0.23 + 8**0.23) / 0.4685)), color="#868e96", ls=":", label="hard min of r/s_ij")
    ax[0].axhspan(sw["inner"], sw["outer"], color="#f1f3f5")
    ax[0].set_xlabel("Si–F distance (Å)"); ax[0].set_ylabel("σ_i"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("atom-level switching variable (Si with 3 O + 1 F)")
    ax[1].plot(d, f, color=COL["mult"])
    ax[1].set_xlabel("Si–F distance (Å)"); ax[1].set_ylabel("f(τ_i): weight on E_i^ML")
    ax[1].set_title("E_i^ML → f(τ_i)·E_i^ML   (C⁴ smoothstep)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "softmin_switch.svg"))
    plt.close(fig)


def fig_bessel(c7):
    freq = np.array(c7["bessel_freq"])
    r = np.linspace(0.01, 5, 500)
    B = bessel(r, freq)
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.axvspan(1.5, 3.5, color="#f1f3f5", label="typical bond/neighbour range")
    for n in range(len(freq)):
        ax.plot(r, B[:, n], lw=1, color=plt.cm.viridis(n / len(freq)))
    ax.scatter(np.zeros(len(freq)), 0.4 * freq, color="#c2255c", s=12, zorder=5, label="limit 0.4·c_n at r→0")
    ax.set_xlabel("r (Å)"); ax.set_ylabel("B_n(r) = 0.4·sin(c_n r)/r")
    ax.set_title("SevenNet checkpoint radial basis (8 trainable frequencies)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bessel_basis.svg"))
    plt.close(fig)


def fig_sevennet(scan, data):
    d = scan["dimers"]
    r = np.array(d["r"])
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.5))
    for a_, pair in zip(ax, ["Si-Si", "Si-F"]):
        p = d["pairs"][pair]
        Ez = zbl(np.maximum(r, 1e-3), *p["Z"])[0]
        a_.semilogy(r, Ez, color=COL["zbl"], label="bare ZBL")
        a_.semilogy(r, np.where(np.array(p["E_ml"]) > 0, p["E_ml"], np.nan), color=COL["bad"], label="SevenNet raw ML (no switch)")
        a_.semilogy(r, np.where(np.array(p["E_tot"]) > 0, p["E_tot"], np.nan), color=COL["ref"], ls="--", label="SevenNet total (switch + ZBL)")
        a_.axvspan(*p["window"], color="#f1f3f5")
        a_.set_xlim(0, 2.5); a_.set_ylim(0.1, 2e5); a_.set_xlabel("r (Å)"); a_.set_title(pair)
    ax[0].set_ylabel("E (eV, vs separated atoms)")
    ax[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sevennet_dimer.svg"))
    plt.close(fig)


def fig_spill(scan):
    rows = scan["spill"]
    r = np.array([x["r"] for x in rows])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(r, [x["dF_tot"] for x in rows], color=COL["ref"], marker="o", ms=3, label="SevenNet total (as used in MD)")
    ax.plot(r, [x["dF_ml"] for x in rows], color=COL["bad"], marker="o", ms=3, label="SevenNet raw ML part")
    ax.plot(r, [x["F_switch"] for x in rows], color=COL["fermi"], ls="--", label="−E_i^ML ∇f_i  (switch-gradient term)")
    ax.plot(r, [x["dF_bpnn"] for x in rows], color=COL["sw"], marker="o", ms=3, label="BPNN Si3N4 (no switch)")
    ax.invert_xaxis(); ax.set_xlabel("distance of the pushed atom to the centre atom (Å)")
    ax.set_ylabel("max change of spectator force (eV/Å)")
    ax.set_title("Si17 cluster: forces on the 15 atoms that do not move")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "spill.svg"))
    plt.close(fig)


def fig_learn(L):
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.6))
    for k, t, c, ls in [("G2", "full", COL["sw"], "-"), ("Bessel", "full", COL["bad"], "-"),
                        ("G2", "delta", COL["mult"], "--"), ("Bessel", "delta", COL["fermi"], "--")]:
        ax[0].semilogy(L["rmins"], [L["fits"][f"{k}|{t}|{x}"]["rmse_F"] for x in L["rmins"]], color=c, ls=ls, marker="o", ms=3, label=f"{k}, {t}")
    ax[0].invert_xaxis(); ax[0].set_xlabel("training r_min (Å)"); ax[0].set_ylabel("force RMSE on training set (eV/Å)")
    ax[0].set_title("same model, data extended toward r → 0"); ax[0].legend(frameon=False, fontsize=8)
    for k, c in [("G2", COL["sw"]), ("Bessel", COL["bad"])]:
        d = L[f"Lreq_{k}"]
        ax[1].loglog(d["r"], d["full"], color=c, label=f"{k}, full")
        ax[1].loglog(d["r"], d["delta"], color=c, ls="--", label=f"{k}, delta")
    rr = np.array(L["Lreq_G2"]["r"]); m = rr < 0.4
    ax[1].loglog(rr[m], 3e1 * rr[m] ** -3, color="#868e96", ls=":", label="∝ r⁻³")
    ax[1].set_xlabel("r (Å)"); ax[1].set_ylabel("required Lipschitz constant |F|/(2‖dG/dr‖)")
    ax[1].set_title("a priori: what the weights would have to do"); ax[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "learnability.svg"))
    plt.close(fig)


def fig_softening():
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for frac, c in [(0.0, COL["bad"]), (0.05, COL["sw"]), (0.3, COL["mult"])]:
        rr, Vt, Vf, x, y = softening_toy(frac)
        ax[0].plot(rr, Vf, color=c, label=f"fit, off-eq. data {int(frac*100)}%")
    ax[0].plot(rr, Vt, color="#212529", lw=1.2, label="true PES (Morse)")
    rr, Vt, Vf, x, y = softening_toy(0.05)
    ax[0].scatter(x, y, s=4, color="#868e96", zorder=0)
    ax[0].set_ylim(-0.5, 8); ax[0].set_xlabel("r (Å)"); ax[0].set_ylabel("E (eV)")
    ax[0].legend(frameon=False, fontsize=7.5); ax[0].set_title("same smooth model, different data")
    fr = np.linspace(0, 0.4, 21)
    ratio = []
    for q in fr:
        rr, Vt, Vf, *_ = softening_toy(q)
        m = (rr > 1.8) & (rr < 2.2)
        ratio.append(np.polyfit(rr[m], Vf[m], 1)[0] / np.polyfit(rr[m], Vt[m], 1)[0])
    ax[1].plot(fr * 100, ratio, color=COL["sw"], marker="o", ms=3)
    ax[1].axhline(1, color="#adb5bd", lw=0.8)
    ax[1].set_xlabel("share of off-equilibrium data (%)"); ax[1].set_ylabel("fitted / true slope at 1.8–2.2 Å")
    ax[1].set_title("softening factor")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "softening_toy.svg"))
    plt.close(fig)


if __name__ == "__main__":
    data = dimer_curves()
    folded = folded_net()
    rows = cluster_SiO4F()
    c7 = sevennet_constants()
    print("bounds:", {k: v["bound"] for k, v in data["pots"].items()})
    print("SevenNet:", c7)
    for x in rows:
        print(x)
    fig_tanh_bound(folded); fig_g2(); fig_angular(); fig_cluster(rows); fig_dimers(data)
    fig_softmin(c7["sw"]); fig_bessel(c7); fig_softening()
    extra = {}
    p7 = os.path.join(HERE, "sevennet_scan.json")
    if os.path.exists(p7):          # produced by sevennet_scan.py (needs torch)
        extra = json.load(open(p7))
        fig_sevennet(extra, data); fig_spill(extra)
    learn, swv = {}, {}
    pl = os.path.join(HERE, "learnability.json")      # produced by learnability_scan.py (needs torch)
    if os.path.exists(pl):
        learn = json.load(open(pl))
        for f in learn["fits"].values():
            for k in ("r", "E_ref", "E_fit", "F_ref", "F_fit", "dGdr"):
                f[k] = rnd(f[k][::2])
        for k in ("Lreq_G2", "Lreq_Bessel"):
            learn[k] = {kk: rnd(v[::2]) for kk, v in learn[k].items()}
        fig_learn(learn)
    ps = os.path.join(HERE, "sw_variants.json")       # produced by sw_variants.py (needs torch)
    if os.path.exists(ps):
        swv = json.load(open(ps))
    with open(os.path.join(HERE, "demo_data.json"), "w") as fh:
        json.dump({"dimers": data, "folded": folded, "cluster": rows, "sevennet": c7, "sevennet_scan": extra,
                   "learn": learn, "sw_variants": swv}, fh, separators=(",", ":"))
    print("demo_data.json", os.path.getsize(os.path.join(HERE, "demo_data.json")) // 1024, "KB")
