"""
Model 1D transien perpindahan panas pada kain komposit berlapis (PCM + grafena/MXene)
untuk kamuflase inframerah pita 8-14 um.

Mengimplementasikan Persamaan (1)-(8) pada paper:
  (1) dH/dt = d/dx( k dT/dx )                         -> volume hingga cell-centered
  (2) H(T) = rho*cp*T + Lv*phi(T), phi = 0.5[1+erf((T-Tm)/(sqrt2*sigma_m))]
  (3) kondisi batas permukaan luar: konduksi = konveksi + radiasi GP - surya terserap
  (4) Hamilton-Crosser dua langkah (kapsul bola, lalu filler lembaran)
  (5) model tutupan filler untuk emisivitas dan absorptivitas
  (6)-(8) radiansi tampak pita 8-14 um, suhu tampak, kontras terhadap latar belakang
Integrasi waktu: backward Euler, Newton-Raphson dengan Jacobi tridiagonal.
"""
import numpy as np
from dataclasses import dataclass, field, replace
from scipy.linalg import solve_banded
from scipy.special import erf

SIG = 5.670374419e-8
H_PL, C_L, K_B = 6.62607015e-34, 2.99792458e8, 1.380649e-23
K0 = 273.15

# ------------------------------------------------------------------ radiansi pita
_trap = getattr(np, "trapezoid", None) or np.trapz
_lam = np.linspace(8e-6, 14e-6, 601)
def band_radiance(T_c):
    """Persamaan (7): integral Planck 8-14 um, T dalam degC (skalar atau array)."""
    T = np.atleast_1d(np.asarray(T_c, float)) + K0
    x = H_PL*C_L/(_lam[None, :]*K_B*T[:, None])
    B = 2*H_PL*C_L**2/_lam[None, :]**5 / np.expm1(x)
    return _trap(B, _lam, axis=1)

_Ttab = np.linspace(-20, 120, 1401)
_Ltab = band_radiance(_Ttab)
def apparent_temperature(L):
    """Invers L_b(T) dengan interpolasi tabel (suhu benda hitam ekuivalen, degC)."""
    return np.interp(L, _Ltab, _Ttab)

# ------------------------------------------------------------------ material
@dataclass
class Mat:
    rho: float; cp: float; k: float; eps: float = np.nan; alpha: float = np.nan

@dataclass
class Params:
    air: Mat = field(default_factory=lambda: Mat(1.17, 1005, 0.026))
    cotton: Mat = field(default_factory=lambda: Mat(400, 1300, 0.06, 0.90, 0.50))
    binder: Mat = field(default_factory=lambda: Mat(1100, 1800, 0.25, 0.90, 0.60))
    capsule: Mat = field(default_factory=lambda: Mat(900, 2000, 0.20))
    gnp: Mat = field(default_factory=lambda: Mat(2200, 700, 1000.0, 0.50, 0.90))
    mxene: Mat = field(default_factory=lambda: Mat(3800, 500, 50.0, 0.25, 0.80))
    # PCM
    Tm: float = 28.0; sigma_m: float = 1.0; L_core: float = 200e3; f_core: float = 0.8
    w_cap: float = 0.30; w_fill: float = 0.02; w_theta: float = 0.03
    psi_fill: float = 0.15
    # geometri (m) dan diskretisasi
    d_air: float = 1.0e-3; d_cot: float = 0.5e-3; d_comp: float = 1.0e-3
    n_air: int = 4; n_cot: int = 6; n_comp: int = 30
    # batas
    T_skin: float = 34.0; h_c: float = 12.0; eps_bg: float = 0.95
    dT_sur: float = 3.0   # T_sur = T_a - dT_sur

def hamilton_crosser(km, kd, phi, psi):
    n = 3.0/psi
    return km*(kd + (n-1)*km - (n-1)*phi*(km-kd))/(kd + (n-1)*km + phi*(km-kd))

def composite_props(p: Params, filler=None, w_cap=None, w_fill=None, latent=True):
    """Sifat efektif lapisan komposit: Persamaan (4) dan (5) + pencampuran massa."""
    w_cap = p.w_cap if w_cap is None else w_cap
    w_fill = (p.w_fill if filler else 0.0) if w_fill is None else w_fill
    fm = {'gnp': p.gnp, 'mxene': p.mxene, None: None}[filler]
    w_b = 1.0 - w_cap - w_fill
    comp = [(w_b, p.binder), (w_cap, p.capsule)] + ([(w_fill, fm)] if fm and w_fill > 0 else [])
    inv_rho = sum(w/m.rho for w, m in comp)
    rho = 1.0/inv_rho
    cp = sum(w*m.cp for w, m in comp)
    vol = {id(m): (w/m.rho)/inv_rho for w, m in comp}
    # langkah 1: binder + kapsul (bola, psi = 1)
    v_b, v_c = vol[id(p.binder)], vol[id(p.capsule)]
    phi_c = v_c/(v_b + v_c)
    k1 = hamilton_crosser(p.binder.k, p.capsule.k, phi_c, 1.0)
    # langkah 2: campuran + filler lembaran
    if fm and w_fill > 0:
        phi_f = vol[id(fm)]
        k = hamilton_crosser(k1, fm.k, phi_f, p.psi_fill)
        theta = 1.0 - np.exp(-w_fill/p.w_theta)
        eps = p.binder.eps*(1-theta) + fm.eps*theta
        alpha = p.binder.alpha*(1-theta) + fm.alpha*theta
    else:
        k, eps, alpha = k1, p.binder.eps, p.binder.alpha
    Lv = rho*w_cap*p.f_core*p.L_core if latent else 0.0
    return dict(rho=rho, cp=cp, k=k, eps=eps, alpha=alpha, Lv=Lv,
                L_mass=w_cap*p.f_core*p.L_core if latent else 0.0)

# ------------------------------------------------------------------ solver
class Column:
    """Susunan 1D: [kulit] | udara | kain | (komposit) | [permukaan luar]."""
    def __init__(self, p: Params, comp: dict | None, d_comp=None):
        self.p = p
        layers = [(p.d_air, p.n_air, dict(rho=p.air.rho, cp=p.air.cp, k=p.air.k, Lv=0.0)),
                  (p.d_cot, p.n_cot, dict(rho=p.cotton.rho, cp=p.cotton.cp, k=p.cotton.k, Lv=0.0))]
        if comp is not None:
            d = p.d_comp if d_comp is None else d_comp
            layers.append((d, p.n_comp, comp))
            self.eps, self.alpha = comp['eps'], comp['alpha']
        else:
            self.eps, self.alpha = p.cotton.eps, p.cotton.alpha
        dx, rho, cp, k, Lv = [], [], [], [], []
        for d, n, m in layers:
            dx += [d/n]*n; rho += [m['rho']]*n; cp += [m['cp']]*n
            k += [m['k']]*n; Lv += [m['Lv']]*n
        self.dx, self.rho, self.cp = map(np.array, (dx, rho, cp))
        self.k, self.Lv = np.array(k), np.array(Lv)
        self.N = len(dx)
        # konduktansi antar sel (resistansi seri setengah-sel) dan ke kulit / permukaan
        R = self.dx[:-1]/(2*self.k[:-1]) + self.dx[1:]/(2*self.k[1:])
        self.G = 1.0/R
        self.G_skin = self.k[0]/(self.dx[0]/2)
        self.G_surf = self.k[-1]/(self.dx[-1]/2)
        self.Tm, self.sm = p.Tm, p.sigma_m

    def H(self, T):
        phi = 0.5*(1+erf((T-self.Tm)/(np.sqrt(2)*self.sm)))
        return self.rho*self.cp*T + self.Lv*phi
    def dH(self, T):
        g = np.exp(-0.5*((T-self.Tm)/self.sm)**2)/(np.sqrt(2*np.pi)*self.sm)
        return self.rho*self.cp + self.Lv*g
    def phi(self, T):
        return 0.5*(1+erf((T-self.Tm)/(np.sqrt(2)*self.sm)))

    def step(self, Tn, Ts_guess, dt, Ta, Tsur, G, tol=1e-7, maxit=50, limit=5.0):
        """Satu langkah backward Euler. Tak diketahui: T_1..T_N dan T_s (suhu permukaan)."""
        N, p = self.N, self.p
        Hn = self.H(Tn)
        x = np.concatenate([Tn, [Ts_guess]])
        nit = 0
        for it in range(maxit):
            T, Ts = x[:N], x[N]
            F = np.zeros(N+1)
            # sel: dx*(H-Hn)/dt = flux masuk - flux keluar
            flux = self.G*(T[:-1]-T[1:])           # dari sel i ke i+1 (positif ke arah permukaan)
            qin = np.zeros(N); qout = np.zeros(N)
            qin[0] = self.G_skin*(p.T_skin - T[0])
            qin[1:] = flux
            qout[:-1] = flux
            qout[-1] = self.G_surf*(T[-1]-Ts)
            F[:N] = self.dx*(self.H(T)-Hn)/dt - (qin - qout)
            # permukaan: konduksi = konveksi + radiasi - surya
            qs = p.h_c*(Ts-Ta) + self.eps*SIG*((Ts+K0)**4-(Tsur+K0)**4) - self.alpha*G
            F[N] = self.G_surf*(T[-1]-Ts) - qs
            # Jacobi tridiagonal (N+1)
            dHt = self.dH(T)
            diag = np.zeros(N+1); up = np.zeros(N); lo = np.zeros(N)
            d_main = self.dx*dHt/dt
            d_main[0] += self.G_skin
            d_main[:-1] += self.G
            d_main[1:] += self.G
            d_main[-1] += self.G_surf
            diag[:N] = d_main
            up[:N-1] = -self.G           # dF_i/dT_{i+1}
            lo[:N-1] = -self.G           # dF_{i+1}/dT_i
            up[N-1] = -self.G_surf       # dF_N/dTs
            lo[N-1] = self.G_surf        # dFs/dT_N
            diag[N] = -self.G_surf - p.h_c - 4*self.eps*SIG*(Ts+K0)**3
            ab = np.zeros((3, N+1)); ab[0, 1:] = up; ab[1] = diag; ab[2, :-1] = lo
            dx_ = solve_banded((1, 1), ab, -F)
            m = np.max(np.abs(dx_))
            if m > limit: dx_ *= limit/m
            x = x + dx_
            nit += 1
            if np.max(np.abs(dx_)) < tol: break
        return x[:N], x[N], nit

# ------------------------------------------------------------------ skenario
def daily_forcing(t, dT_sur=3.0):  # t dalam jam sejak 00:00 hari pertama
    h = t % 24.0
    Ta = 27.5 + 4.5*np.cos(2*np.pi*(h-14.0)/24.0)
    Tbg = 29.0 + 9.0*np.cos(2*np.pi*(h-13.5)/24.0)
    G = 800.0*np.sin(np.pi*(h-6.0)/12.0) if 6.0 <= h <= 18.0 else 0.0
    return Ta, Ta-dT_sur, G, Tbg

def shade_sun_forcing(t, dT_sur=3.0):  # t dalam jam, matahari muncul pada t = 0,5 jam
    if t < 0.5:
        Ta, G, Tbg = 30.0, 0.0, 28.0
    else:
        Ta, G = 30.0, 700.0
        Tbg = 36.0 - 8.0*np.exp(-(t-0.5)/0.5)
    return Ta, Ta-dT_sur, G, Tbg

def run(col: Column, scenario='daily', dt=None, T0=None):
    p = col.p
    if scenario == 'daily':
        dt = 30.0 if dt is None else dt
        t_end, forcing = 72.0, daily_forcing
    else:
        dt = 5.0 if dt is None else dt
        t_end, forcing = 2.5, shade_sun_forcing
    n = int(round(t_end*3600/dt))
    N = col.N
    Ta, Tsur, G, Tbg = forcing(0.0, p.dT_sur)
    # kondisi awal: profil tunak linear sederhana, lalu relaksasi
    T = np.linspace(p.T_skin, Ta, N) if T0 is None else T0.copy()
    Ts = T[-1]
    out = dict(t=[], Ts=[], Tbg=[], Ta=[], Tsur=[], G=[], phi=[], q_skin=[], q_surf=[], nit=[])
    T_start = T.copy()
    if scenario == 'shade':
        # awal tunak pada kondisi teduh (iterasi hingga konvergen)
        for _ in range(400):
            T, Ts, _n = col.step(T, Ts, 600.0, Ta, Tsur, G)
    for i in range(1, n+1):
        t = i*dt/3600.0
        Ta, Tsur, G, Tbg = forcing(t, p.dT_sur)
        T, Ts, nit = col.step(T, Ts, dt, Ta, Tsur, G)
        out['t'].append(t); out['Ts'].append(Ts); out['Tbg'].append(Tbg)
        out['Ta'].append(Ta); out['Tsur'].append(Tsur); out['G'].append(G)
        out['phi'].append(col.phi(T)[col.N-p.n_comp:].mean() if col.Lv.max() > 0 else 0.0)
        out['q_skin'].append(col.G_skin*(p.T_skin-T[0])); out['q_surf'].append(col.G_surf*(T[-1]-Ts)); out['nit'].append(nit)
    out = {k: np.array(v) for k, v in out.items()}
    out['dt'] = dt; out['T_start'] = T_start; out['T_end'] = T.copy()
    # Persamaan (6)-(8)
    L_s = col.eps*band_radiance(out['Ts']) + (1-col.eps)*band_radiance(out['Tsur'])
    L_bg = p.eps_bg*band_radiance(out['Tbg']) + (1-p.eps_bg)*band_radiance(out['Tsur'])
    out['Tapp_s'] = apparent_temperature(L_s)
    out['Tapp_bg'] = apparent_temperature(L_bg)
    out['dT'] = out['Tapp_s'] - out['Tapp_bg']
    return out

def metrics(out, t_from=None, t_to=None, thr=2.0):
    t = out['t']; m = np.ones_like(t, bool)
    if t_from is not None: m &= t > t_from
    if t_to is not None: m &= t <= t_to
    d = np.abs(out['dT'][m])
    return dict(mean=d.mean(), rms=np.sqrt((out['dT'][m]**2).mean()), max=d.max(),
                hidden=100.0*(d < thr).mean())
