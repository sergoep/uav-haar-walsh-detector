#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 20260616
OUTDIR = Path('.')
FIGDIR = OUTDIR / 'figures'
FIGDIR.mkdir(exist_ok=True)

# -------- helper functions for examples --------
def normN(x):
    return np.sqrt(np.mean(x*x))

def center_vec(x):
    return x - np.mean(x)

def center_batch(X):
    return X - X.mean(axis=1, keepdims=True)

def fwht_vec(X):
    A = np.array(X, dtype=float, copy=True)
    squeeze = False
    if A.ndim == 1:
        A = A[None, :]
        squeeze = True
    n = A.shape[1]
    h = 1
    while h < n:
        A = A.reshape(A.shape[0], -1, 2*h)
        x = A[:, :, :h].copy()
        y = A[:, :, h:2*h].copy()
        A[:, :, :h] = x + y
        A[:, :, h:2*h] = x - y
        A = A.reshape(A.shape[0], n)
        h *= 2
    A = A / np.sqrt(n)
    return A[0] if squeeze else A

def gen_legitimate_batch(n, N, rng):
    idx = np.arange(N)
    X = np.zeros((n, N))
    for i in range(n):
        K = rng.integers(2, 5)
        x = np.zeros(N)
        for _ in range(K):
            f = rng.uniform(0.5, 7.0)
            phase = rng.uniform(0, 2*np.pi)
            amp = rng.uniform(0.35, 1.0) / (f**0.25)
            x += amp*np.sin(2*np.pi*f*idx/N + phase)
        e = rng.normal(0, 0.04, N)
        for k in range(1, N):
            e[k] = 0.85*e[k-1] + e[k]
        x += e + rng.normal(0, 0.08, N) + rng.uniform(-0.4, 0.4)
        x *= rng.uniform(0.7, 1.3)
        X[i] = x
    return X

def basis_subspace_V(N, level):
    J = int(np.log2(N))
    block = 2 ** (J - level)
    nb = 2 ** level
    cols = []
    for b in range(nb):
        v = np.zeros(N)
        v[b*block:(b+1)*block] = 1/np.sqrt(block)
        cols.append(v)
    return np.stack(cols, axis=1)

def bases(N, r=3):
    J = int(np.log2(N))
    one = np.ones(N) / np.sqrt(N)
    level = J - r
    V = basis_subspace_V(N, level)
    M = V - np.outer(one, one @ V)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    Bc = U[:, :np.sum(S > 1e-10)]
    Pvc = V @ V.T
    Pfin = np.eye(N) - Pvc
    U, S, _ = np.linalg.svd(Pfin, full_matrices=False)
    Bf = U[:, :np.sum(S > 1e-10)]
    return Bc, Bf, Pfin

def random_unit_from_basis(B, y, rng):
    for _ in range(30):
        coeff = rng.normal(size=B.shape[1])
        v = B @ coeff
        if y is not None:
            yy = y @ y
            if yy > 1e-12:
                v = v - (v @ y)/yy*y
        nv = np.sqrt(np.mean(v*v))
        if nv > 1e-10:
            return v / nv
    return None

def centered_spherical_batch(S, B, theta, rng):
    X = []
    for s in S:
        y = center_vec(s)
        u = random_unit_from_basis(B, y, rng)
        if u is None:
            X.append(s.copy())
        else:
            X.append(np.mean(s) + np.cos(theta)*y + np.sin(theta)*normN(y)*u)
    return np.array(X)

def fine_preserving_batch(S, Bc, Pfin, rng):
    X = []
    for s in S:
        y = center_vec(s)
        h = Pfin @ y
        g = y - h
        gn = normN(g)
        if gn < 1e-12 or Bc.shape[1] < 2:
            X.append(s.copy()); continue
        z = random_unit_from_basis(Bc, None, rng) * gn
        if normN(z-g) < 1e-4:
            z = -z
        X.append(np.mean(s) + h + z)
    return np.array(X)

def walsh_power_batch(S, rng):
    Y = center_batch(S)
    C = fwht_vec(Y)
    signs = rng.choice([-1,1], size=C.shape)
    Y2 = fwht_vec(C * signs)
    return S.mean(axis=1, keepdims=True) + Y2

def make_example_windows():
    N=256
    rng = np.random.default_rng(SEED)
    Bc, Bf, Pfin = bases(N, r=3)
    base = gen_legitimate_batch(1, N, rng)
    scenarios = {
        'Model B-fine': centered_spherical_batch(base, Bf, 0.70, rng),
        'Model B-coarse': centered_spherical_batch(base, Bc, 0.70, rng),
        'Model C-Haar-preserving': fine_preserving_batch(base, Bc, Pfin, rng),
        'Model A-Walsh-power': walsh_power_batch(base, rng),
    }
    fig, axes = plt.subplots(2,2, figsize=(12,7), constrained_layout=True)
    axes = axes.ravel()
    for ax, (name, anom) in zip(axes, scenarios.items()):
        ax.plot(base[0], label='reference', linewidth=1.1)
        ax.plot(anom[0], label='stress window', linewidth=1.0)
        ax.set_title(name)
        ax.set_xlabel('sample index')
        ax.set_ylabel('amplitude')
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.savefig(FIGDIR/'example_windows.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

def make_bar_figures():
    results = pd.read_csv(OUTDIR/'synthetic_results.csv')
    invariants = pd.read_csv(OUTDIR/'invariant_checks.csv')
    methods = ['Energy', 'FFT/STFT', 'Wavelet/WPT', 'Haar only', 'Walsh only', 'Haar-Walsh + conformal']
    scenarios = results['Scenario'].drop_duplicates().tolist()

    x = np.arange(len(scenarios)); width = 0.13
    fig, ax = plt.subplots(figsize=(12,5), constrained_layout=True)
    for i, method in enumerate(methods):
        vals = [results[(results['Scenario']==sc)&(results['Method']==method)]['ROC_AUC'].iloc[0] for sc in scenarios]
        ax.bar(x + (i-(len(methods)-1)/2)*width, vals, width=width, label=method)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, rotation=12)
    ax.set_ylim(0,1.05); ax.set_ylabel('ROC-AUC')
    ax.set_title('Synthetic ROC-AUC across scenarios and methods')
    ax.grid(True, axis='y', alpha=0.25); ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.savefig(FIGDIR/'auc_bars.png', dpi=220, bbox_inches='tight'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12,5), constrained_layout=True)
    for i, method in enumerate(methods):
        vals = [results[(results['Scenario']==sc)&(results['Method']==method)]['TPR_conf'].iloc[0] for sc in scenarios]
        ax.bar(x + (i-(len(methods)-1)/2)*width, vals, width=width, label=method)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, rotation=12)
    ax.set_ylim(0,1.05); ax.set_ylabel('TPR at conformal $\\alpha=0.05$')
    ax.set_title('Conformal detection power across scenarios and methods')
    ax.grid(True, axis='y', alpha=0.25); ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.savefig(FIGDIR/'tpr_bars.png', dpi=220, bbox_inches='tight'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,4.6), constrained_layout=True)
    arr = invariants[['max_abs_energy_diff','max_abs_mean_diff','max_abs_center_energy_diff']].values + 1e-18
    labels = ['energy', 'mean', 'centered energy']
    width=0.22; x=np.arange(len(invariants))
    for i in range(arr.shape[1]):
        ax.bar(x + (i-1)*width, arr[:,i], width=width, label=labels[i])
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(invariants['Scenario'], rotation=12)
    ax.set_ylabel('maximum absolute invariant error (log scale)')
    ax.set_title('Invariant preservation errors at machine precision')
    ax.grid(True, axis='y', alpha=0.25); ax.legend(frameon=False)
    fig.savefig(FIGDIR/'invariant_checks.png', dpi=220, bbox_inches='tight'); plt.close(fig)

if __name__ == '__main__':
    make_example_windows()
    make_bar_figures()
    print('Figures written to', FIGDIR)
