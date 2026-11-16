"""Kasus utama: verifikasi, konvergensi, Tabel 2-5, Gambar 2-3 (input baru dari literatur)."""
import json, sys, numpy as np
from scipy.optimize import brentq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
KOMA = FuncFormatter(lambda v, _: ('%g' % v).replace('.', ',').replace('-', '\u2212'))  # desimal koma (Bahasa Indonesia)
from camo_model import *
from inputs_new import new_params

p = new_params()
DESIGNS = [('Kain polos', None, 0.0), ('Binder saja', None, 0.0), ('Binder + PCM', None, p.w_cap),
           ('Binder + PCM + MXene 2%', 'mxene', p.w_cap), ('Binder + PCM + grafena 2%', 'gnp', p.w_cap)]
def make(name, f, wc, pp=p, **kw):
    comp = None if name == 'Kain polos' else composite_props(pp, filler=f, w_cap=wc, **kw)
    return Column(pp, comp), comp
R = {}

# ---------- Tabel 2
t2 = {}
for name, f, wc in DESIGNS[1:]:
    c = composite_props(p, filler=f, w_cap=wc)
    t2[name] = {k: float(c[k]) for k in ('rho', 'cp', 'k', 'eps', 'alpha', 'L_mass', 'Lv')}
R['table2'] = t2

# ---------- verifikasi (3 uji)
ver = {}
c = composite_props(p, filler='mxene', w_cap=p.w_cap)
col = Column(p, c)
# (i) integral kapasitas panas semu = kalor laten analitik
T = np.linspace(p.Tm-12*p.sigma_m, p.Tm+12*p.sigma_m, 200001)
g = np.exp(-0.5*((T-p.Tm)/p.sigma_m)**2)/(np.sqrt(2*np.pi)*p.sigma_m)
ver['latent_numeric_MJ_m3'] = float(np.trapezoid(c['Lv']*g, T)/1e6)
ver['latent_analytic_MJ_m3'] = float(c['Lv']/1e6)
# (ii) suhu permukaan tunak vs solusi analitik hambatan seri
Ta_, Tsur_, G_ = 30.0, 27.0, 0.0
colv = Column(p, c)
Tt = np.linspace(p.T_skin, Ta_, colv.N); Ts = Tt[-1]
for _ in range(600): Tt, Ts, _n = colv.step(Tt, Ts, 600.0, Ta_, Tsur_, G_)
Rcond = p.d_air/p.air.k + p.d_cot/p.cotton.k + p.d_comp/c['k']
f = lambda Tsx: (p.T_skin-Tsx)/Rcond - (p.h_c*(Tsx-Ta_) + c['eps']*SIG*((Tsx+K0)**4-(Tsur_+K0)**4) - c['alpha']*G_)
Ts_an = brentq(f, 0, 60)
ver['Ts_steady_numeric'] = float(Ts); ver['Ts_steady_analytic'] = float(Ts_an)
ver['Ts_steady_abs_diff'] = float(abs(Ts-Ts_an))
# (iii) galat keseimbangan energi global (siklus harian 3 hari)
o = run(col, 'daily')
dE = float(np.sum(col.dx*(col.H(o['T_end'])-col.H(o['T_start']))))
Wnet = float(np.sum((o['q_skin']-o['q_surf']))*o['dt'])
scale = float(np.sum(np.abs(o['q_skin'])+np.abs(o['q_surf']))*o['dt'])
ver['energy_dE_J_m2'] = dE; ver['energy_flux_J_m2'] = Wnet
ver['energy_rel_error'] = abs(dE-Wnet)/scale
ver['newton_mean'] = float(o['nit'].mean()); ver['newton_max'] = int(o['nit'].max())
R['verification'] = ver

# ---------- konvergensi (MXene 2%, siklus harian, hari ke-3)
conv = []
for label, scale_n, dt in [('20 sel, dt 30 s', 0.5, 30), ('40 sel, dt 30 s', 1, 30), ('80 sel, dt 30 s', 2, 30),
                           ('40 sel, dt 60 s', 1, 60), ('40 sel, dt 15 s', 1, 15), ('80 sel, dt 15 s', 2, 15)]:
    pp = new_params(n_air=int(4*scale_n), n_cot=int(6*scale_n), n_comp=int(30*scale_n))
    cc = composite_props(pp, filler='mxene', w_cap=pp.w_cap)
    oo = run(Column(pp, cc), 'daily', dt=dt); m = metrics(oo, 48, 72)
    conv.append(dict(case=label, mean=float(m['mean']), rms=float(m['rms']), max=float(m['max']), Ts_max=float(oo['Ts'].max())))
R['convergence'] = conv

# ---------- Tabel 3 dan 4 + gambar
runs = {'daily': {}, 'shade': {}}
t34 = {'daily': {}, 'shade': {}}
for scen, win in [('daily', (48, 72)), ('shade', (None, None))]:
    for name, f_, wc in DESIGNS:
        col_, _c = make(name, f_, wc)
        o_ = run(col_, scen); runs[scen][name] = o_
        m = metrics(o_, *win)
        t34[scen][name] = {k: float(v) for k, v in m.items()}
        t34[scen][name]['Ts_max'] = float(o_['Ts'][(o_['t'] > 48) if scen == 'daily' else slice(None)].max())
        if scen == 'shade': t34[scen][name]['Ts_end'] = float(o_['Ts'][-1])
R['table3'] = t34['daily']; R['table4'] = t34['shade']
# fraksi cair PCM (kasus MXene)
R['liquid_daily_day3'] = {n: [float(runs['daily'][n]['phi'][runs['daily'][n]['t'] > 48].min()),
                              float(runs['daily'][n]['phi'][runs['daily'][n]['t'] > 48].max())]
                          for n in ('Binder + PCM', 'Binder + PCM + MXene 2%', 'Binder + PCM + grafena 2%')}
R['liquid_shade_before_sun'] = float(runs['shade']['Binder + PCM + MXene 2%']['phi'][runs['shade']['Binder + PCM + MXene 2%']['t'] <= 0.5].min())

colors = {'Kain polos': 'k', 'Binder saja': 'tab:blue', 'Binder + PCM': 'tab:orange',
          'Binder + PCM + MXene 2%': 'tab:green', 'Binder + PCM + grafena 2%': 'tab:red'}
styles = {'Binder saja': '-', 'Binder + PCM': '--'}
def figure(scen, fname, title, xlab, tf):
    o0 = runs[scen]['Binder saja']; m = (o0['t'] > 48) if scen == 'daily' else (o0['t'] >= 0)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    a = ax[0, 0]
    for n, o in runs[scen].items(): a.plot(tf(o['t'])[m], o['Ts'][m], styles.get(n, '-'), color=colors[n], label=n, lw=1.4)
    a.plot(tf(o0['t'])[m], o0['Tbg'][m], ':', color='gray', label='Latar belakang'); a.plot(tf(o0['t'])[m], o0['Ta'][m], ':', color='tab:purple', label='Udara')
    a.set_ylabel('Suhu permukaan [°C]'); a.legend(fontsize=6.5)
    a = ax[0, 1]
    for n, o in runs[scen].items(): a.plot(tf(o['t'])[m], o['dT'][m], styles.get(n, '-'), color=colors[n], lw=1.4)
    a.axhspan(-2, 2, color='tab:green', alpha=0.15); a.axhline(0, color='k', lw=0.5); a.set_ylabel(r'Kontras suhu tampak $\Delta T_\mathrm{app}$ [K]')
    a = ax[1, 0]
    for n, o in runs[scen].items():
        if n in ('Kain polos', 'Binder saja'): continue
        a.plot(tf(o['t'])[m], o['phi'][m], styles.get(n, '-'), color=colors[n], lw=1.4)
    a.set_ylabel('Fraksi cair PCM'); a.set_ylim(-0.05, 1.05)
    a = ax[1, 1]
    for n, o in runs[scen].items(): a.plot(tf(o['t'])[m], o['q_skin'][m], styles.get(n, '-'), color=colors[n], lw=1.4)
    a.axhline(0, color='k', lw=0.5); a.set_ylabel('Fluks dari kulit [W/m²]')
    for a in ax.ravel():
        a.set_xlabel(xlab); a.grid(alpha=0.3)
        a.xaxis.set_major_formatter(KOMA); a.yaxis.set_major_formatter(KOMA)
    if scen == 'daily': [a.set_xlim(0, 24) for a in ax.ravel()]
    fig.suptitle(title, fontsize=10); fig.savefig(fname, dpi=160); plt.close(fig)
figure('daily', 'Gambar2_siklus_harian.png', 'Siklus harian (hari ke-3)', 'Waktu [jam]', lambda t: (t-48.0))
figure('shade', 'Gambar3_teduh_matahari.png', 'Transien teduh–matahari (matahari muncul pada t = 0,5 jam)', 'Waktu [jam]', lambda t: t)
# ---------- Tabel 5
t5 = []
for Tm in (28.0, 42.0, 48.0):
    for d in (1e-3, 3e-3):
        pp = new_params(Tm=Tm)
        cw = composite_props(pp, filler=None, w_cap=0.5, latent=True)
        cn = composite_props(pp, filler=None, w_cap=0.5, latent=False)
        ow = run(Column(pp, cw, d_comp=d), 'shade'); on = run(Column(pp, cn, d_comp=d), 'shade')
        mw = metrics(ow, 0.5, None); mn = metrics(on, 0.5, None)
        t5.append(dict(Tm=Tm, d_mm=d*1e3, dT_without=float(mn['mean']), dT_with=float(mw['mean']), phi_max=float(ow['phi'].max()), Ts_max=float(ow['Ts'].max())))
R['table5'] = t5
json.dump(R, open('results_main.json', 'w'), indent=1, ensure_ascii=False)
print('done')
