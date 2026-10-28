"""Figures and numbers for docs/zbl_study (ZBL + switching function notes).

Units: eV, Angstrom, amu. ZBL constants follow lammps/pair_zbl_const.h.
Run: python3 make_figures.py  (needs numpy, matplotlib)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

KE2 = 14.399645          # e^2/(4 pi eps0) [eV A]
C_ = [0.02817, 0.28022, 0.50986, 0.18175]
D_ = [0.20162, 0.40290, 0.94229, 3.19980]
TU = 10.1805             # sqrt(amu A^2 / eV) in fs
MASS = {1: 1.008, 6: 12.011, 7: 14.007, 9: 18.998, 14: 28.0855}


def zbl(r, zi, zj):
    """E, dE/dr, d2E/dr2 of the bare ZBL potential."""
    ai = (zi**0.23 + zj**0.23) / 0.46850
    k = zi * zj * KE2
    e = sum(c * np.exp(-d * ai * r) for c, d in zip(C_, D_))
    ep = sum(-c * d * ai * np.exp(-d * ai * r) for c, d in zip(C_, D_))
    epp = sum(c * (d * ai)**2 * np.exp(-d * ai * r) for c, d in zip(C_, D_))
    E = k * e / r
    dE = k * (ep - e / r) / r
    d2E = k * (epp - 2 * ep / r + 2 * e / r**2) / r
    return E, dE, d2E


def gromacs_coeff(zi, zj, r1, rc):
    """A, B, C of the LAMMPS lj/gromacs-style switching S(r)."""
    tc = rc - r1
    fc, fcp, fcpp = zbl(rc, zi, zj)
    A = (-3 * fcp + tc * fcpp) / tc**2
    B = (2 * fcp - tc * fcpp) / tc**3
    C = -fc + tc / 2 * fcp - tc**2 / 12 * fcpp
    return A, B, C


def zbl_sw(r, zi, zj, r1, rc):
    """E, dE/dr of ZBL + S(r) (what pair zbl and zbl/pair compute)."""
    r = np.asarray(r, dtype=float)
    A, B, C = gromacs_coeff(zi, zj, r1, rc)
    E, dE, _ = zbl(r, zi, zj)
    t = np.clip(r - r1, 0, None)
    E = E + C + A / 3 * t**3 + B / 4 * t**4
    dE = dE + A * t**2 + B * t**3
    out = r >= rc
    return np.where(out, 0.0, E), np.where(out, 0.0, dE)


def poly_env(r, r1, rc):
    """Multiplicative C2 envelope: 1 for r<r1, 0 for r>rc (quintic smoothstep)."""
    x = np.clip((r - r1) / (rc - r1), 0, 1)
    f = 1 - (10 * x**3 - 15 * x**4 + 6 * x**5)
    df = -(30 * x**2 - 60 * x**3 + 30 * x**4) / (rc - r1)
    return f, df


def zbl_mult(r, zi, zj, r1, rc):
    E, dE, _ = zbl(r, zi, zj)
    f, df = poly_env(r, r1, rc)
    return f * E, f * dE + df * E


def morse(r, D=3.0, re=2.35, a=1.6):
    x = np.exp(-a * (r - re))
    return D * (x**2 - 2 * x), D * (-2 * a * x**2 + 2 * a * x)


def fermi(r, AF=14.0, rC=1.6):
    f = 1 / (1 + np.exp(-AF * (r - rC)))
    return f, AF * f * (1 - f)


# ---------------------------------------------------------------- tables
def table_shift():
    pairs = [("Si-Si", 14, 14, 1.598, 2.054), ("Si-F", 14, 9, 1.145, 1.472),
             ("Si-C", 14, 6, 1.208, 1.553), ("Si-H", 14, 1, 1.079, 1.388),
             ("C-F", 6, 9, 0.908, 1.168), ("F-F", 9, 9, 0.997, 1.282),
             ("H-H", 1, 1, 0.525, 0.675)]
    rows = []
    for n, zi, zj, r1, rc in pairs:
        A, B, C = gromacs_coeff(zi, zj, r1, rc)
        rows.append((n, r1, rc, zbl(r1, zi, zj)[0], C, -zbl(rc, zi, zj)[1]))
    return rows


def r_min(V, E, lo=0.05, hi=3.0):
    """Distance of closest approach in a head-on collision: V(r)=E."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if V(mid) > E:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def table_rmin():
    rows = []
    for n, zi, zj, r1, rc in [("Si-Si", 14, 14, 1.598, 2.054),
                              ("Si-F", 14, 9, 1.145, 1.472),
                              ("C-F", 6, 9, 0.908, 1.168)]:
        for Ecm in [10, 30, 100, 300, 1000]:
            a = r_min(lambda r: zbl(r, zi, zj)[0], Ecm)
            b = r_min(lambda r: float(zbl_sw(r, zi, zj, r1, rc)[0]), Ecm)
            rows.append((n, Ecm, a, b))
    return rows


# ---------------------------------------------------------------- NVE
def nve(force_energy, mu, Ecm, r0=3.0, dt_fs=1.0, tmax_fs=400.0):
    """1D head-on collision of a dimer (relative coordinate), velocity Verlet.

    force_energy(r, v) -> (F, E_reported). Returns arrays t, r, Etot."""
    dt = dt_fs / TU
    r = r0
    v = -np.sqrt(2 * Ecm / mu)
    F, E = force_energy(r, v)
    ts, rs, Es = [0.0], [r], [0.5 * mu * v * v + E]
    n = int(tmax_fs / dt_fs)
    for i in range(n):
        v += 0.5 * dt * F / mu
        r += dt * v
        F, E = force_energy(r, v)
        v += 0.5 * dt * F / mu
        ts.append((i + 1) * dt_fs); rs.append(r); Es.append(0.5 * mu * v * v + E)
    return np.array(ts), np.array(rs), np.array(Es)


def nve_cases():
    zi = zj = 14
    mu = MASS[14] / 2
    r1, rc = 1.598, 2.054
    Ecm = 30.0

    def sw(r, v):
        E, dE = zbl_sw(r, zi, zj, r1, rc)
        return -float(dE), float(E)

    def hard(r, v):
        if r >= rc:
            return 0.0, 0.0
        E, dE, _ = zbl(r, zi, zj)
        return -dE, E

    def blend_ok(r, v):
        f, df = fermi(r)
        Ez, dEz, _ = zbl(r, zi, zj)
        Em, dEm = morse(r)
        E = (1 - f) * Ez + f * Em
        dE = (1 - f) * dEz + f * dEm + df * (Em - Ez)
        return -dE, E

    def blend_bad(r, v):  # forgets the df*(Em - Ez) term
        f, df = fermi(r)
        Ez, dEz, _ = zbl(r, zi, zj)
        Em, dEm = morse(r)
        E = (1 - f) * Ez + f * Em
        dE = (1 - f) * dEz + f * dEm
        return -dE, E

    def kin_switch(r, v):  # ZBL only "for energetic atoms": depends on velocity
        if 0.5 * mu * v * v > 5.0:
            E, dE = zbl_sw(r, zi, zj, r1, rc)
        else:
            E, dE = 0.0, 0.0
        Em, dEm = morse(r)
        return -float(dE) - dEm, float(E) + Em

    def sw_plus_morse(r, v):
        E, dE = zbl_sw(r, zi, zj, r1, rc)
        Em, dEm = morse(r)
        return -float(dE) - dEm, float(E) + Em

    cases = [
        ("(a) ZBL + S(r) [gromacs]", sw),
        ("(b) hard cutoff (no S)", hard),
        ("(c) Fermi blend, full gradient", blend_ok),
        ("(d) Fermi blend, missing f'(r) term", blend_bad),
        ("(e) velocity-dependent ZBL on/off", kin_switch),
    ]
    out = {}
    for name, fe in cases:
        out[name] = nve(fe, mu, Ecm)
    out["_ref_sw_morse"] = nve(sw_plus_morse, mu, Ecm)
    return out


def narrow_window_drift():
    """Final |dE| after one N-N head-on collision: bare ZBL vs wide/narrow window."""
    rows = []
    mu = MASS[7] / 2
    for n, r1, rc in [("bare ZBL", None, None), ("wide 0.78/1.00", 0.78, 1.00),
                      ("narrow 0.95/1.00", 0.95, 1.00)]:
        for Ecm in [10.0, 30.0, 100.0]:
            for dt in [1.0, 0.5, 0.2, 0.1]:
                if r1 is None:
                    def fe(r, v):
                        E, dE, _ = zbl(r, 7, 7)
                        return -dE, E
                else:
                    def fe(r, v, r1=r1, rc=rc):
                        E, dE = zbl_sw(r, 7, 7, r1, rc)
                        return -float(dE), float(E)
                t, r, E = nve(fe, mu, Ecm, r0=1.6, dt_fs=dt, tmax_fs=200.0)
                rows.append((n, Ecm, dt, abs(E[-1] - E[0])))
    return rows


# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110})
COL = {"zbl": "#2a6fdb", "sw": "#d9480f", "mult": "#2b8a3e", "ref": "#555555",
       "bad": "#c2255c", "fermi": "#7048e8"}


def fig_switch():
    zi = zj = 14; r1, rc = 1.598, 2.054
    r = np.linspace(0.8, 2.4, 800)
    Ez, dEz, _ = zbl(r, zi, zj)
    Es, dEs = zbl_sw(r, zi, zj, r1, rc)
    Em, dEm = zbl_mult(r, zi, zj, r1, rc)
    A, B, C = gromacs_coeff(zi, zj, r1, rc)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for a in ax:
        a.axvspan(r1, rc, color="#f1f3f5", zorder=0)
        a.axvline(r1, color="#adb5bd", lw=0.8); a.axvline(rc, color="#adb5bd", lw=0.8)
        a.axhline(0, color="#adb5bd", lw=0.8)
    ax[0].plot(r, Ez, color=COL["zbl"], label="bare ZBL")
    ax[0].plot(r, Es, color=COL["sw"], label="ZBL + S(r)  (additive, pair zbl)")
    ax[0].plot(r, Em, color=COL["mult"], ls="--", label="f_c(r)·ZBL  (multiplicative)")
    ax[0].annotate(f"C = {C:.2f} eV", xy=(1.1, float(zbl(1.1, 14, 14)[0]) + C),
                   xytext=(1.25, 25), arrowprops=dict(arrowstyle="->", color="#495057"))
    ax[0].set_ylim(-5, 40); ax[0].set_xlabel("r (Å)"); ax[0].set_ylabel("pair energy (eV)")
    ax[0].set_title("Si–Si energy, inner 1.598 / outer 2.054 Å")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(r, -dEz, color=COL["zbl"], label="bare ZBL")
    ax[1].plot(r, -dEs, color=COL["sw"], label="ZBL + S(r)")
    ax[1].plot(r, -dEm, color=COL["mult"], ls="--", label="f_c(r)·ZBL")
    ax[1].set_ylim(-2, 60); ax[1].set_xlabel("r (Å)"); ax[1].set_ylabel("force −dE/dr (eV/Å)")
    ax[1].set_title("force: identical for r < inner")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "switching.svg")); plt.close(fig)


def fig_nve(res):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    colors = [COL["sw"], COL["ref"], COL["mult"], COL["bad"], COL["fermi"]]
    for (name, (t, r, E)), c in zip([(k, v) for k, v in res.items() if not k.startswith("_")], colors):
        ax.plot(t, E - E[0], color=c, label=name)
    ax.axhline(0, color="#adb5bd", lw=0.8)
    ax.set_xlabel("time (fs)"); ax.set_ylabel("E_total − E_total(0)  (eV)")
    ax.set_title("Si–Si head-on collision, E_cm = 30 eV, dt = 1 fs (velocity Verlet)")
    ax.set_xlim(0, 60)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "nve.svg")); plt.close(fig)


def fig_reference():
    """Q7: what delta learning reproduces when the reference misbehaves at short r."""
    zi = zj = 14; r1, rc = 1.598, 2.054
    r = np.linspace(0.9, 2.6, 600)
    Ez, _, _ = zbl(r, zi, zj)
    Es, _ = zbl_sw(r, zi, zj, r1, rc)
    Em, _ = morse(r)
    w, _ = fermi(r, AF=8.0, rC=1.9)
    good = (1 - w) * Ez * 1.05 + w * Em                      # DFT-like: -> ZBL (+5 %)
    soft = good - (1 - w) * Ez * 0.6 * np.clip((1.6 - r) / 0.7, 0, 1)  # MLIP-like, too soft
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for a in ax:
        a.axvspan(r1, rc, color="#f1f3f5", zorder=0); a.axhline(0, color="#adb5bd", lw=0.8)
    ax[0].plot(r, Ez, color=COL["zbl"], label="bare ZBL")
    ax[0].plot(r, good, color=COL["ref"], label="reference A (DFT-like)")
    ax[0].plot(r, soft, color=COL["bad"], label="reference B (MLIP, softening)")
    ax[0].set_ylim(-5, 60); ax[0].set_xlabel("r (Å)"); ax[0].set_ylabel("E (eV)")
    ax[0].set_title("reference vs ZBL"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(r, good - Es, color=COL["ref"], label="target A − (ZBL+S)")
    ax[1].plot(r, soft - Es, color=COL["bad"], label="target B − (ZBL+S)")
    ax[1].set_xlabel("r (Å)"); ax[1].set_ylabel("residual the MLIP must learn (eV)")
    ax[1].set_title("residual: ~constant (A) vs growing (B)")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "reference_residual.svg")); plt.close(fig)


if __name__ == "__main__":
    print("== shift C and outer force (training cutoffs) ==")
    for n, r1, rc, Ein, C, Fout in table_shift():
        print(f"{n:6} {r1:6.3f} {rc:6.3f}  E_zbl(r1)={Ein:7.2f}  C={C:7.2f}  F_zbl(rc)={Fout:6.2f}")
    print("== closest approach (head-on), ZBL vs ZBL+S ==")
    for n, E, a, b in table_rmin():
        print(f"{n:6} Ecm={E:5d}  r_min(ZBL)={a:.3f}  r_min(ZBL+S)={b:.3f}")
    res = nve_cases()
    print("== NVE: max |dE| during collision, final dE ==")
    for k, (t, r, E) in res.items():
        print(f"{k:40} max|dE|={np.max(np.abs(E-E[0])):9.4f}  final dE={E[-1]-E[0]:9.4f}  r_end={r[-1]:.2f}")
    print("== window width vs dt (N-N head-on, final |dE| after one collision) ==")
    for row in narrow_window_drift():
        print("%-17s Ecm=%5.0f dt=%.1f fs  final|dE|=%.4f eV" % row)
    fig_switch(); fig_nve(res); fig_reference()
