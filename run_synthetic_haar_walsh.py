#!/usr/bin/env python3
"""
Synthetic offline experiments for:
Calibrated Haar--Walsh Structural Diagnostics for Energy-Preserving Anomalies.

Safety boundary:
This script performs only recorded/simulated data transformations for detector auditing.
It does not generate RF waveforms, jamming, spoofing, or operational interference procedures.

Python >= 3.10
Dependencies: numpy, pandas, scikit-learn, matplotlib
"""

import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

SEED = 20260616

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
        # AR(1)-like colored component
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
    return np.max(np.abs(V-med)/mad, axis=1)

def metrics(score_leg, score_anom, score_cal, alpha=0.05):
    y = np.r_[np.zeros(len(score_leg)), np.ones(len(score_anom))]
    s = np.r_[score_leg, score_anom]
    auc = roc_auc_score(y, s)
    pr = average_precision_score(y, s)
    tau = np.quantile(score_leg, 1-alpha, method="higher")
    tpr = np.mean(score_anom > tau)
    m = len(score_cal)
    k = math.ceil((m+1)*(1-alpha))
    tau_c = np.inf if k > m else np.sort(score_cal)[k-1]
    return {
        "ROC_AUC": auc,
        "PR_AUC": pr,
        "TPR_at_005": tpr,
        "Emp_FA": np.mean(score_leg > tau_c),
        "TPR_conf": np.mean(score_anom > tau_c),
    }

def run_once(N=256, n_train=800, n_cal=400, n_test=500, seed=SEED, scenario="fine"):
    rng = np.random.default_rng(seed)
    Bc, Bf, Pfin = bases(N, r=3)
    train = gen_legitimate_batch(n_train, N, rng)
    cal = gen_legitimate_batch(n_cal, N, rng)
    leg = gen_legitimate_batch(n_test, N, rng)
    base = gen_legitimate_batch(n_test, N, rng)

    if scenario == "fine":
        anom = centered_spherical_batch(base, Bf, 0.70, rng)
    elif scenario == "coarse":
        anom = centered_spherical_batch(base, Bc, 0.70, rng)
    elif scenario == "fine_preserving":
        anom = fine_preserving_batch(base, Bc, Pfin, rng)
    elif scenario == "walsh_power":
        anom = walsh_power_batch(base, rng)
    else:
        raise ValueError(scenario)

    feature_funcs = {
        "Energy": lambda X: np.mean(X*X, axis=1),
        "FFT/STFT": lambda X: np.vstack([fft_score_batch(X), stft_entropy_batch(X)]).T,
        "Wavelet/WPT": lambda X: wavelet_packet_proxy_batch(X),
        "Haar only": lambda X: features_all(X)[0][:, 0],
        "Walsh only": lambda X: features_all(X)[0][:, 1:3],
        "Haar-Walsh + conformal": lambda X: features_all(X)[0],
    }

    rows = []
    for name, fn in feature_funcs.items():
        tr = fn(train)
        ca = fn(cal)
        le = fn(leg)
        an = fn(anom)
        med, mad = robust_fit(tr)
        met = metrics(robust_score(le, med, mad), robust_score(an, med, mad), robust_score(ca, med, mad))
        rows.append({"Method": name, **met})

    inv = {
        "max_abs_energy_diff": np.max(np.abs(np.mean(anom*anom, axis=1)-np.mean(base*base, axis=1))),
        "max_abs_mean_diff": np.max(np.abs(np.mean(anom, axis=1)-np.mean(base, axis=1))),
        "max_abs_center_energy_diff": np.max(np.abs(np.mean(center_batch(anom)**2, axis=1)-np.mean(center_batch(base)**2, axis=1))),
    }
    return pd.DataFrame(rows), inv

def main():
    scenarios = [
        ("Model B-fine", "fine"),
        ("Model B-coarse", "coarse"),
        ("Model C-Haar-preserving", "fine_preserving"),
        ("Model A-Walsh-power", "walsh_power"),
    ]
    all_rows = []
    inv_rows = []
    for i, (label, scenario) in enumerate(scenarios):
        df, inv = run_once(seed=SEED + 1000*i, scenario=scenario)
        df.insert(0, "Scenario", label)
        all_rows.append(df)
        inv["Scenario"] = label
        inv_rows.append(inv)

    results = pd.concat(all_rows, ignore_index=True)
    invariants = pd.DataFrame(inv_rows)
    results.to_csv("synthetic_results.csv", index=False)
    invariants.to_csv("invariant_checks.csv", index=False)

    print("\n=== Synthetic results ===")
    print(results.round(3).to_string(index=False))
    print("\n=== Invariant checks ===")
    print(invariants.to_string(index=False))

if __name__ == "__main__":
    main()
