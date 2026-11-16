"""Uji sensitivitas: ubah satu asumsi per kasus, hitung ulang metrik utama."""
import json, copy
from camo_model import *
from inputs_new import new_params

def mod(fn):
    p = new_params(); fn(p); return p
VARIANTS = [
 ('Acuan (Tabel 1 baru)', lambda p: None),
 ('k filler dalam bidang (GNP 1000; MXene 50)', lambda p: (setattr(p.gnp,'k',1000.0), setattr(p.mxene,'k',50.0))),
 ('k MXene 6,2 (nanosheet)', lambda p: setattr(p.mxene,'k',6.2)),
 ('k binder 0,13', lambda p: setattr(p.binder,'k',0.13)),
 ('k mikrokapsul 0,1', lambda p: setattr(p.capsule,'k',0.10)),
 ('k mikrokapsul 0,5', lambda p: setattr(p.capsule,'k',0.50)),
 ('k katun 0,04', lambda p: setattr(p.cotton,'k',0.04)),
 ('k katun 0,07', lambda p: setattr(p.cotton,'k',0.07)),
 ('c_p binder 1500', lambda p: setattr(p.binder,'cp',1500.0)),
 ('c_p binder 2000', lambda p: setattr(p.binder,'cp',2000.0)),
 ('rho katun 300', lambda p: setattr(p.cotton,'rho',300.0)),
 ('rho katun 500', lambda p: setattr(p.cotton,'rho',500.0)),
 ('eps binder 0,80', lambda p: setattr(p.binder,'eps',0.80)),
 ('eps binder 0,95', lambda p: setattr(p.binder,'eps',0.95)),
 ('alpha binder 0,50', lambda p: setattr(p.binder,'alpha',0.50)),
 ('alpha binder 0,70', lambda p: setattr(p.binder,'alpha',0.70)),
 ('eps MXene 0,29', lambda p: setattr(p.mxene,'eps',0.29)),
 ('eps MXene 0,15', lambda p: setattr(p.mxene,'eps',0.15)),
 ('alpha MXene 0,70', lambda p: setattr(p.mxene,'alpha',0.70)),
 ('alpha MXene 0,90', lambda p: setattr(p.mxene,'alpha',0.90)),
 ('eps grafena 0,40', lambda p: setattr(p.gnp,'eps',0.40)),
 ('eps grafena 0,60', lambda p: setattr(p.gnp,'eps',0.60)),
 ('alpha grafena 0,80', lambda p: setattr(p.gnp,'alpha',0.80)),
 ('w_theta 0,02', lambda p: setattr(p,'w_theta',0.02)),
 ('w_theta 0,05', lambda p: setattr(p,'w_theta',0.05)),
 ('alpha katun 0,36 (abu-abu)', lambda p: setattr(p.cotton,'alpha',0.36)),
 ('alpha katun 0,17 (putih)', lambda p: setattr(p.cotton,'alpha',0.17)),
 ('k grafena 5', lambda p: setattr(p.gnp,'k',5.0)),
 ('T_sur = T_a', lambda p: setattr(p,'dT_sur',0.0)),
 ('T_sur = T_a - 6 K', lambda p: setattr(p,'dT_sur',6.0)),
 ('T_sur = T_a - 12 K', lambda p: setattr(p,'dT_sur',12.0)),
 ('h_c 8', lambda p: setattr(p,'h_c',8.0)),
 ('h_c 16', lambda p: setattr(p,'h_c',16.0)),
]
DES = [('Kain polos', None, 0.0), ('Binder saja', None, 0.0), ('Binder + PCM', None, None),
       ('MXene 2%', 'mxene', None), ('Grafena 2%', 'gnp', None)]
def one(i):
    label, fn = VARIANTS[i]
    p = mod(fn); row = dict(case=label)
    for scen, win in (('daily', (48, 72)), ('shade', (None, None))):
        for name, f, wc in DES:
            comp = None if name == 'Kain polos' else composite_props(p, filler=f, w_cap=p.w_cap if wc is None else wc)
            o = run(Column(p, comp), scen, dt=(60.0 if scen == 'daily' else None)); m = metrics(o, *win)
            row[f'{scen}|{name}'] = float(m['mean'])
    return row
if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool(1) as pool:
        out = pool.map(one, range(len(VARIANTS)))
    json.dump(out, open('results_sens.json', 'w'), indent=1, ensure_ascii=False)
    print('done')
