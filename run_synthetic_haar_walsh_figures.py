#!/usr/bin/env python3
"""
Synthetic offline experiments and figure generation for
"Calibrated Haar--Walsh Structural Diagnostics for Energy-Preserving Anomalies in Short UAV RF/GNSS Signal Windows".

Safety boundary:
This script performs only recorded/simulated data transformations for detector auditing.
It does not generate RF waveforms, jamming, spoofing, or operational interference procedures.

Python >= 3.10
Dependencies: numpy, pandas, scikit-learn, matplotlib
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt

SEED = 20260616
OUTDIR = Path('.')
FIGDIR = OUTDIR / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)


def normN(x):
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return np.sqrt(np.mean(np.abs(x)**2))
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


def haar_detail_energies_batch(X):
    X = np.asarray(X)
    N = X.shape[1]
    J = int(np.log2(N))
    Ps = []
    for j in range(J + 1):
        block = 2 ** (J - j)
        Y = X.reshape(X.shape[0], -1, block)
        M = Y.mean(axis=2, keepdims=True)
        Ps.append(np.repeat(M, block, axis=2).reshape(X.shape[0], N))
    ens = []
    for j in range(J):
        d = Ps[j+1] - Ps[j]
        ens.append(np.mean(d*d, axis=1))
    return np.vstack(ens).T


def features_all(X, r=3, deltaH=1e-8, deltaS=1e-8):
    X = np.asarray(X)
    Y = center_batch(X)
    he = haar_detail_energies_batch(Y)
    FH = he[:, -r:].sum(axis=1) / (np.mean(Y*Y, axis=1) + deltaH)
    C = fwht_vec(Y)
    E = C*C
    total = E.sum(axis=1) + 1e-20
    P = E / total[:, None]
    FE = -np.sum(np.where(P > 0, P*np.log(P + 1e-300), 0.0), axis=1)
    FS = (np.sum(np.abs(C), axis=1)**2) / (np.sum(C*C, axis=1) + deltaS)
    return np.vstack([FH, FE, FS]).T, he


def fft_score_batch(X, cutoff=0.25):
    Y = center_batch(X)
    N = Y.shape[1]
    C = np.fft.rfft(Y, axis=1)
    power = np.abs(C)**2
    freqs = np.fft.rfftfreq(N)
    return power[:, freqs >= cutoff].sum(axis=1) / (power.sum(axis=1) + 1e-9)


def stft_entropy_batch(X, frame=64, hop=32):
    X = np.asarray(X)
    Y = center_batch(X)
    N = X.shape[1]
    win = np.hanning(frame)
    chunks = []
    for start in range(0, N-frame+1, hop):
        seg = Y[:, start:start+frame] * win
        C = np.fft.rfft(seg, axis=1)
        chunks.append(np.abs(C)**2)
    P = np.concatenate(chunks, axis=1)
    P = P / (P.sum(axis=1, keepdims=True) + 1e-20)
    return -np.sum(np.where(P > 0, P*np.log(P + 1e-300), 0.0), axis=1)


def wavelet_packet_proxy_batch(X):
    _, he = features_all(X)
    P = he / (he.sum(axis=1, keepdims=True) + 1e-20)
    return -np.sum(np.where(P > 0, P*np.log(P + 1e-300), 0.0), axis=1)


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
    for _ in range(50):
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
            X.append(s.copy())
            continue
        z = random_unit_from_basis(Bc, None, rng) * gn
        if normN(z - g) < 1e-4:
            z = -z
        X.append(np.mean(s) + h + z)
    return np.array(X)


def walsh_power_batch(S, rng):
    Y = center_batch(S)
    C = fwht_vec(Y)
    signs = rng.choice([-1, 1], size=C.shape)
    Y2 = fwht_vec(C * signs)
    return S.mean(axis=1, keepdims=True) + Y2


def robust_fit(V):
    V = np.asarray(V)
    med = np.median(V, axis=0)
    mad = np.median(np.abs(V-med), axis=0)*1.4826 + 1e-9
    return med, mad


def robust_score(V, med, mad):
    V = np.asarray(V)
    if V.ndim == 1:
        return np.abs(V-med)/mad
    return np.max(np.abs(V-med), axis=1) / mad.max() if med.ndim == 1 and V.ndim == 2 else np.max(np.abs(V-med)/mad, axis=1)


def metrics(score_leg, score_anom, score_cal, alpha=0.05):
    y = np.r_[np.zeros(len(score_leg)), np.ones(len(score_anom))]
    s = np.r_[score_leg, score_anom]
    auc = roc_auc_score(y, s)
    pr = average_precision_score(y, s)
    tau = np.quantile(score_leg, 1-alpha, method='higher')
    tpr = np.mean(score_anom > tau)
    m = len(score_cal)
    k = math.ceil((m+1)*(1-alpha))
    tau_c = np.inf if k > m else np.sort(score_cal)[k-1]
    fpr, tpr_curve, _ = roc_curve(y, s)
    return {
        'ROC_AUC': auc,
        'PR_AUC': pr,
        'TPR_at_005': tpr,
        'Emp_FA': np.mean(score_leg > tau_c),
        'TPR_conf': np.mean(score_anom > tau_c),
        'roc_fpr': fpr,
        'roc_tpr': tpr_curve,
        'leg_scores': score_leg,
        'anom_scores': score_anom,
        'cal_scores': score_cal,
    }


def run_once(N=256, n_train=800, n_cal=400, n_test=500, seed=SEED, scenario='fine'):
    rng = np.random.default_rng(seed)
    Bc, Bf, Pfin = bases(N, r=3)
    train = gen_legitimate_batch(n_train, N, rng)
    cal = gen_legitimate_batch(n_cal, N, rng)
    leg = gen_legitimate_batch(n_test, N, rng)
    base = gen_legitimate_batch(n_test, N, rng)

    if scenario == 'fine':
        anom = centered_spherical_batch(base, Bf, 0.70, rng)
    elif scenario == 'coarse':
        anom = centered_spherical_batch(base, Bc, 0.70, rng)
    elif scenario == 'fine_preserving':
        anom = fine_preserving_batch(base, Bc, Pfin, rng)
    elif scenario == 'walsh_power':
        anom = walsh_power_batch(base, rng)
    else:
        raise ValueError(scenario)

    feature_funcs = {
        'Energy': lambda X: np.mean(X*X, axis=1),
        'FFT/STFT': lambda X: np.vstack([fft_score_batch(X), stft_entropy_batch(X)]).T,
        'Wavelet/WPT': lambda X: wavelet_packet_proxy_batch(X),
        'Haar only': lambda X: features_all(X)[0][:, 0],
        'Walsh only': lambda X: features_all(X)[0][:, 1:3],
        'Haar-Walsh + conformal': lambda X: features_all(X)[0],
    }

    rows = []
    store = {}
    for name, fn in feature_funcs.items():
        tr = fn(train)
        ca = fn(cal)
        le = fn(leg)
        an = fn(anom)
        med, mad = robust_fit(tr)
        score_leg = robust_score(le, med, mad)
        score_an = robust_score(an, med, mad)
        score_cal = robust_score(ca, med, mad)
        met = metrics(score_leg, score_an, score_cal)
        rows.append({k: v for k, v in met.items() if isinstance(v, (int, float, np.floating))} | {'Method': name})
        store[name] = met

    inv = {
        'max_abs_energy_diff': np.max(np.abs(np.mean(anom*anom, axis=1)-np.mean(base*base, axis=1))),
        'max_abs_mean_diff': np.max(np.abs(np.mean(anom, axis=1)-np.mean(base, axis=1))),
        'max_abs_center_energy_diff': np.max(np.abs(np.mean(center_batch(anom)**2, axis=1)-np.mean(center_batch(base)**2, axis=1))),
    }
    return pd.DataFrame(rows), inv, {'train': train, 'cal': cal, 'leg': leg, 'base': base, 'anom': anom, 'store': store}


def plot_example_windows(example_dict):
    scenario_order = ['Model B-fine', 'Model B-coarse', 'Model C-Haar-preserving', 'Model A-Walsh-power']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes = axes.ravel()
    for ax, label in zip(axes, scenario_order):
        base = example_dict[label]['base'][0]
        anom = example_dict[label]['anom'][0]
        ax.plot(base, label='reference', linewidth=1.2)
        ax.plot(anom, label='stress window', linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel('sample index')
        ax.set_ylabel('amplitude')
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc='upper right', frameon=False)
    fig.savefig(FIGDIR / 'example_windows.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_auc_bars(results):
    methods = ['Energy', 'FFT/STFT', 'Wavelet/WPT', 'Haar only', 'Walsh only', 'Haar-Walsh + conformal']
    scenarios = results['Scenario'].unique().tolist()
    x = np.arange(len(scenarios))
    width = 0.13
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for i, method in enumerate(methods):
        vals = [results[(results['Scenario']==sc)&(results['Method']==method)]['ROC_AUC'].iloc[0] for sc in scenarios]
        ax.bar(x + (i - (len(methods)-1)/2)*width, vals, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Synthetic ROC-AUC across scenarios and methods')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.savefig(FIGDIR / 'auc_bars.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_tpr_bars(results):
    methods = ['Energy', 'FFT/STFT', 'Wavelet/WPT', 'Haar only', 'Walsh only', 'Haar-Walsh + conformal']
    scenarios = results['Scenario'].unique().tolist()
    x = np.arange(len(scenarios))
    width = 0.13
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for i, method in enumerate(methods):
        vals = [results[(results['Scenario']==sc)&(results['Method']==method)]['TPR_conf'].iloc[0] for sc in scenarios]
        ax.bar(x + (i - (len(methods)-1)/2)*width, vals, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('TPR at conformal $\\alpha=0.05$')
    ax.set_title('Conformal detection power across scenarios and methods')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.savefig(FIGDIR / 'tpr_bars.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_invariant_checks(invariants):
    scenarios = invariants['Scenario'].tolist()
    arr = invariants[['max_abs_energy_diff', 'max_abs_mean_diff', 'max_abs_center_energy_diff']].values + 1e-18
    labels = ['energy', 'mean', 'centered energy']
    x = np.arange(len(scenarios))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for i in range(arr.shape[1]):
        ax.bar(x + (i-1)*width, arr[:, i], width=width, label=labels[i])
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=12)
    ax.set_ylabel('maximum absolute invariant error (log scale)')
    ax.set_title('Invariant preservation errors at machine precision')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIGDIR / 'invariant_checks.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_roc_curves(example_dict):
    scenario = 'Model B-fine'
    store = example_dict[scenario]['store']
    methods = ['Energy', 'FFT/STFT', 'Wavelet/WPT', 'Haar only', 'Walsh only', 'Haar-Walsh + conformal']
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    for method in methods:
        ax.plot(store[method]['roc_fpr'], store[method]['roc_tpr'], label=f"{method} (AUC={store[method]['ROC_AUC']:.3f})")
    ax.plot([0,1],[0,1], linestyle='--', linewidth=1)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curves on Model B-fine')
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(FIGDIR / 'roc_model_b_fine.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    scenarios = [
        ('Model B-fine', 'fine'),
        ('Model B-coarse', 'coarse'),
        ('Model C-Haar-preserving', 'fine_preserving'),
        ('Model A-Walsh-power', 'walsh_power'),
    ]
    all_rows = []
    inv_rows = []
    example_dict = {}
    for i, (label, scenario) in enumerate(scenarios):
        df, inv, payload = run_once(seed=SEED + 1000*i, scenario=scenario)
        df.insert(0, 'Scenario', label)
        all_rows.append(df)
        inv['Scenario'] = label
        inv_rows.append(inv)
        example_dict[label] = payload

    results = pd.concat(all_rows, ignore_index=True)
    invariants = pd.DataFrame(inv_rows)
    results.to_csv(OUTDIR / 'synthetic_results.csv', index=False)
    invariants.to_csv(OUTDIR / 'invariant_checks.csv', index=False)

    plot_example_windows(example_dict)
    plot_auc_bars(results)
    plot_tpr_bars(results)
    plot_invariant_checks(invariants)
    plot_roc_curves(example_dict)

    print('\n=== Synthetic results ===')
    print(results.round(3).to_string(index=False))
    print('\n=== Invariant checks ===')
    print(invariants.to_string(index=False))
    print(f'\nFigures written to: {FIGDIR.resolve()}')


if __name__ == '__main__':
    main()
