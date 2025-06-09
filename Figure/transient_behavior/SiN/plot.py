import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors

class PARAMS:
    ENERGIES = [500, 750, 1000]
    SIM_FILES = {}
    for ion in ['CF+', 'CH2F+']:
        SIM_FILES[ion] = {}
        for e in ENERGIES:
            SIM_FILES[ion][e] = f"./sim/data_{ion.replace('+','')}_{e}eV.csv"
    EXP_CSV = './exp/thickness_ref1.csv'
    BASE_COLORS = {'CF+': ('blue', 'darkblue'), 'CH2F+': ('red', 'darkred')}
    LINTYPES = ['-', '--', ':']
    EXP_MARKERS = {'CF+': '^', 'CH2F+': 'o'}
    EXP_COLORS = {'CF+': 'blue', 'CH2F+': 'red'}
    SIM_OVERLAY = {
        'CF+/500eV': ('orange', '-'),
        'CF+/1000eV': ('red', '--'),
        'CH2F+/1000eV': ('blue', ':'),
    }
    MEAN_STEP = 10
    DOSE_STEP = 10

    ion_name_dict = {
        'CF+': 'CF$^+$',
        'CH2F+': 'CH$_2$F$^+$',
    }

    label_dict = {
        'CF+/500eV': 'This study (500 eV)',
        'CF+/1000eV': 'This study (1000 eV)',
        'CH2F+/1000eV': 'This study (1000 eV)',
        'CF+ (exp)': 'Ito et al.',
        'CH2F+ (exp)': 'Ito et al.',
    }

class FigureGenerator:
    def run(self, ncols=1, nrows=1):
        figsize = (3.5 * ncols, 3.5 * nrows)
        plt.rcParams.update({'font.family': 'arial', 'font.size': 10})
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        return fig, axs

class SimValuePlotter:
    def __init__(self, files, cols, lts, mean_step=PARAMS.MEAN_STEP,
                 dose_step=PARAMS.DOSE_STEP, max_step=None):
        self.files = files
        self.cols = cols
        self.lts = lts
        self.mean_step = mean_step
        self.dose_step = dose_step
        self.max_step = max_step

    def run(self, ax):
        for label, path in self.files.items():
            if not os.path.exists(path):
                continue
            d = pd.read_csv(path)
            dose, change = d['dose'].values, d['change'].values
            mv = [np.mean(change[max(0, i - self.mean_step):i + self.mean_step])
                  for i in range(len(change))]
            x = dose[::self.dose_step]
            y = np.array(mv)[::self.dose_step]
            if self.max_step:
                x, y = x[:self.max_step], y[:self.max_step]
            if label not in PARAMS.SIM_OVERLAY or '1000eV' not in label:
                cut = int(4500 / self.dose_step)
                x, y = x[:cut], y[:cut]
            kw = {}
            if label in self.cols:
                kw['color'] = self.cols[label]
            if label in self.lts:
                kw['ls'] = self.lts[label]
            ax.plot(x, y, label=PARAMS.label_dict[label], **kw)
        ax.set_ylabel('Thickness change (nm)')
        ax.set_xlabel(r'Ion Dose ($\times$ 10$^{17}$ cm$^{-2}$)')
        ax.set_ylim(-1.5, 6.0)

class ExpValuePlotter:
    def __init__(self, ptype, draw_fit=False, adjust_xlim=False):
        self.ptype = ptype.lower()
        self.draw_fit = draw_fit
        self.adjust_xlim = adjust_xlim

    def run(self, ax):
        df = pd.read_csv(PARAMS.EXP_CSV)
        for ion in PARAMS.EXP_MARKERS:
            if self.ptype != 'all' and ion.lower() != self.ptype:
                continue
            df_i = df[df['ion'] == ion]
            x, y = df_i['dose'].values, df_i['thickness'].values
            if self.draw_fit:
                if ion == 'CF+':
                    func = lambda x, a, b: a * x**2 + b * x
                    xs = np.linspace(0, 1.9, 100)
                else:
                    func = lambda x, a, b, c, e: e * x**4 + a * x**3 + b * x**2 + c * x
                    xs = np.linspace(0, 1.7, 100)
                p, _ = curve_fit(func, x, y)
                ax.plot(xs, func(xs, *p), '-',
                        color=PARAMS.EXP_COLORS[ion], zorder=1)
            ax.scatter(x, y,
                       marker=PARAMS.EXP_MARKERS[ion],
                       color=PARAMS.EXP_COLORS[ion],
                       label=PARAMS.label_dict[f'{ion} (exp)'],
                       s=60, linewidths=1,
                       edgecolors='black', zorder=2)
        if self.adjust_xlim:
            ax.set_xlim(0, 2)

class Plotter:
    def __init__(self):
        self.p = PARAMS

    def run_calc_only(self, fg):
        # single simulation
        fig1, ax1 = fg.run(1, 1)
        ax1.axhline(0, linestyle='--', linewidth=1, color='gray')
        files = {f'{ion}/{e}eV': path
                 for ion, d in self.p.SIM_FILES.items()
                 for e, path in d.items()}
        cols, lts = {}, {}
        for k, (c, lt) in self.p.SIM_OVERLAY.items():
            cols[k] = c
            lts[k] = lt
        line_dict = {}
        SimValuePlotter(files, cols, lts).run(ax1)
        self.save_figure(fig1, '3_1_3_valid_transient_Si3N4_calc_only')
        ax1.legend( loc='upper left', bbox_to_anchor=(0.01, 0.99))

    def run_calc_with_exp(self, fg):
        # multi-panel simulation + experiment
        fig2, axs = fg.run(1, 2)
        axs = axs.flatten()
        alphabet_dict = {0: '(a) CF$^+$', 1: '(b) CH$_2$F$^+$'}
        for i, ion in enumerate(['CF+', 'CH2F+']):
            ax = axs[i]
            ax.axhline(0, linestyle='--', linewidth=1, color='gray')
            colors = self.interp_colors(*self.p.BASE_COLORS[ion], len(self.p.ENERGIES))
            lts = self.p.LINTYPES
            files = {f'{ion}/{e}eV': self.p.SIM_FILES[ion][e] for e in self.p.ENERGIES}
            SimValuePlotter(
                files,
                dict(zip(files, colors)),
                dict(zip(files, lts))
            ).run(ax)
            ExpValuePlotter(ion, draw_fit=False, adjust_xlim=True).run(ax)
            # ax.set_title(f'{PARAMS.ion_name_dict[ion]}')
            ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), frameon=False)
            ax.text(-0.2, 1.1, alphabet_dict[i], transform=ax.transAxes, fontsize=10)
        self.save_figure(fig2, '3_1_3_valid_transient_Si3N4')

    def save_figure(self, fig, filename):
        fig.tight_layout()
        fig.savefig(f'{filename}.png', dpi=200)
        fig.savefig(f'{filename}.pdf')
        fig.savefig(f'{filename}.eps')

    def run(self):
        fg = FigureGenerator()
        self.run_calc_only(fg)
        self.run_calc_with_exp(fg)

    def interp_colors(self, c1, c2, n):
        r1, g1, b1 = mcolors.to_rgb(c1)
        r2, g2, b2 = mcolors.to_rgb(c2)
        return [mcolors.to_hex((
            r1 + (r2 - r1) * i / (n - 1),
            g1 + (g2 - g1) * i / (n - 1),
            b1 + (b2 - b1) * i / (n - 1)
        )) for i in range(n)]

def main():
    Plotter().run()

if __name__ == '__main__':
    main()
