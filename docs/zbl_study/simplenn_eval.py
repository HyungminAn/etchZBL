"""Minimal Python evaluator for SIMPLE-NN (LAMMPS pair_nn) potential files.

Only what the notes need: G2/G4/G5 symmetry functions without PBC, the
atomic network, dE/dG by backprop, and a radial/angular split of the force.
Conventions follow lammps/pair_nn_original (pair_nn.cpp, symmetry_functions.h).
"""
import numpy as np


def load(path):
    """Parse a SIMPLE-NN potential file -> (element list, {element: net})."""
    L = open(path).read().split('\n')
    elems = L[0].split()[1:]
    nets = {}
    i = 0
    while i < len(L):
        t = L[i].split()
        if not (t and t[0] == 'POT'):
            i += 1
            continue
        el = t[1]
        i += 1
        nsym = int(L[i].split()[1])
        i += 1
        sym = [L[i + k].split() for k in range(nsym)]
        i += nsym
        s0 = np.array(L[i].split()[1:], float)
        s1 = np.array(L[i + 1].split()[1:], float)
        i += 2
        nn = [int(x) for x in L[i].split()[1:]]
        nodes = [nsym] + nn[1:]          # first layer is the (linear) PCA layer
        i += 1
        layers = []
        for l in range(len(nodes) - 1):
            act = L[i].split()[2]
            i += 1
            W = np.zeros((nodes[l + 1], nodes[l]))
            b = np.zeros(nodes[l + 1])
            for o in range(nodes[l + 1]):
                W[o] = np.array(L[i].split()[1:], float)
                b[o] = float(L[i + 1].split()[1])
                i += 2
            layers.append((W, b, act))
        nets[el] = dict(sym=sym, s0=s0, s1=s1, layers=layers)
    return elems, nets


def fc(r, rc):
    return np.where(r < rc, 0.5 * (1 + np.cos(np.pi * r / rc)), 0.0)


def forward(net, G):
    """Atomic energy for descriptor rows G (n x nsym)."""
    x = (G - net['s0']) / net['s1']
    for W, b, act in net['layers']:
        x = x @ W.T + b
        if act == 'tanh':
            x = np.tanh(x)
    return x[..., 0]


def bound(net):
    """|E_i - b_out| <= sum|w_out| because the last hidden layer is tanh."""
    W, b, _ = net['layers'][-1]
    return np.abs(W).sum(), b[0]


def G_dimer(net, r, partner):
    """Descriptors of an atom whose only neighbour (type `partner`) is at r."""
    G = np.zeros((len(r), len(net['sym'])))
    for k, s in enumerate(net['sym']):
        if s[0] == '2' and s[5] == partner:
            rc, eta, rs = float(s[1]), float(s[2]), float(s[3])
            G[:, k] = np.exp(-eta * (r - rs) ** 2) * fc(r, rc)
    return G


def descriptors(net, i, pos, types):
    """Full G2/G4/G5 vector of atom i in a cluster (no PBC)."""
    G = np.zeros(len(net['sym']))
    xi = pos[i]
    others = [j for j in range(len(pos)) if j != i]
    d = {j: np.linalg.norm(pos[j] - xi) for j in others}
    for m, s in enumerate(net['sym']):
        st, rc, eta = int(s[0]), float(s[1]), float(s[2])
        if st == 2:
            rs = float(s[3])
            for j in others:
                if types[j] == s[5] and d[j] < rc:
                    G[m] += np.exp(-eta * (d[j] - rs) ** 2) * fc(d[j], rc)
            continue
        zeta, lam, a, b = float(s[3]), float(s[4]), s[5], s[6]
        for jj, j in enumerate(others):
            for k in others[jj + 1:]:
                if not ((types[j] == a and types[k] == b) or (types[j] == b and types[k] == a)):
                    continue
                rij, rik = d[j], d[k]
                rjk = np.linalg.norm(pos[j] - pos[k])
                if rij > rc or rik > rc:
                    continue
                cos = np.dot(pos[j] - xi, pos[k] - xi) / rij / rik
                ang = 2 ** (1 - zeta) * abs(1 + lam * cos) ** (zeta - 1) * (1 + lam * cos)
                if st == 4:
                    if rjk > rc:
                        continue
                    G[m] += ang * np.exp(-eta * (rij**2 + rik**2 + rjk**2)) * fc(rij, rc) * fc(rik, rc) * fc(rjk, rc)
                else:
                    G[m] += ang * np.exp(-eta * (rij**2 + rik**2)) * fc(rij, rc) * fc(rik, rc)
    return G


def energy_and_dEdG(net, G):
    x = (G - net['s0']) / net['s1']
    outs = []
    for W, b, act in net['layers']:
        x = x @ W.T + b
        if act == 'tanh':
            x = np.tanh(x)
        outs.append(x)
    g = np.ones(1)
    for (W, b, act), out in zip(reversed(net['layers']), reversed(outs)):
        if act == 'tanh':
            g = g * (1 - out**2)
        g = g @ W
    return x[0], g / net['s1']


def decompose(nets, pos, types, h=1e-5):
    """E and the force split F = F_rad + F_ang (chain rule over G2 vs G4/G5)."""
    N = len(pos)
    E = 0.0
    Frad = np.zeros((N, 3))
    Fang = np.zeros((N, 3))
    for i in range(N):
        net = nets[types[i]]
        e, dEdG = energy_and_dEdG(net, descriptors(net, i, pos, types))
        E += e
        rad = np.array([s[0] == '2' for s in net['sym']])
        for k in range(N):
            for c in range(3):
                p = pos.copy()
                p[k, c] += h
                Gp = descriptors(net, i, p, types)
                p[k, c] -= 2 * h
                dG = (Gp - descriptors(net, i, p, types)) / (2 * h)
                Frad[k, c] -= dEdG[rad] @ dG[rad]
                Fang[k, c] -= dEdG[~rad] @ dG[~rad]
    return E, Frad, Fang
