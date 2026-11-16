"""Nilai awal sifat material dari literatur (Tabel 1 baru) dan parameter PCM baru."""
from camo_model import Params, Mat
def new_params(**over):
    p = Params(
        air=Mat(1.16, 1007, 0.026),
        cotton=Mat(400, 1300, 0.06, 0.88, 0.61),   # alpha kain katun gelap (Watanabe et al., 2008)
        binder=Mat(1100, 1800, 0.25, 0.90, 0.60),
        capsule=Mat(935, 2000, 0.20),
        gnp=Mat(2200, 700, 3.2, 0.50, 0.90),     # k tegak lurus bidang terukur 1-3,5 (Cabrera et al., 2021)
        mxene=Mat(3800, 500, 0.4, 0.25, 0.80),
        Tm=28.0, sigma_m=1.0, L_core=245e3, f_core=0.63, w_cap=0.27, w_fill=0.02)
    for k, v in over.items(): setattr(p, k, v)
    return p
