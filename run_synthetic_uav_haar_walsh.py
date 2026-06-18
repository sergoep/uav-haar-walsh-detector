#!/usr/bin/env python3
"""
Synthetic Haar--Walsh benchmark protocol for short UAV-related signal windows.

Offline defensive synthetic benchmark generator for the paper:
"Calibrated Haar-Walsh Structural Features for Lightweight Interference
Detection in Short UAV Signal Windows".

The script generates synthetic clean windows and synthetic perturbation
scenarios, computes Energy/TV/FFT-HF/Haar/Walsh/Full scores, applies
median/MAD normalization and split-conformal calibration, and writes the
LaTeX tables and PDF figures referenced by the manuscript.

It does not implement RF transmission, SDR control, jamming, spoofing,
UAV control-channel manipulation, or operational disruption methods.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

SCENARIOS = [
    "additive_jamming",
    "impulse_jamming",
    "local_substitution",
    "fast_multiplicative",
    "walsh_multiplicative",
    "random_sign_multiplicative",
    "slow_multiplicative",
    "energy_preserving_local",
    "combined_attack",
]

LABELS = {
    "additive_jamming": "Additive jamming",
    "impulse_jamming": "Impulse jamming",
    "local_substitution": "Local substitution",
    "fast_multiplicative": "Fast multiplicative",
    "walsh_multiplicative": "Walsh multiplicative",
    "random_sign_multiplicative": "Random-sign multiplicative",
    "slow_multiplicative": "Slow multiplicative",
    "energy_preserving_local": "Energy-preserving local",
    "combined_attack": "Combined attack",
}

SCORES = ["Energy", "TV", "FFT-HF", "Haar", "Walsh", "Full"]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def center(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) - float(np.mean(x))


def norm2N(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.mean(x * x))


def total_variation(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))


def clean_window(rng: np.random.Generator, N: int) -> np.ndarray:
    t = np.arange(N, dtype=float) / N
    x = (
        rng.uniform(-0.5, 0.5)
        + rng.uniform(-0.5, 0.5) * t
        + rng.uniform(0.1, 1.0) * np.sin(2*np.pi*rng.uniform(1, 8)*t + rng.uniform(0, 2*np.pi))
        + rng.uniform(0.1, 1.0) * np.sin(2*np.pi*rng.uniform(1, 8)*t + rng.uniform(0, 2*np.pi))
    )
    z = rng.normal(0, 0.05, size=N)
    k = np.array([1, 2, 3, 2, 1], dtype=float)
    k /= k.sum()
    return x + np.convolve(z, k, mode="same")


def project(x: np.ndarray, j: int) -> np.ndarray:
    N = len(x)
    block = N // (2**j)
    return x.reshape(2**j, block).mean(axis=1).repeat(block)


def haar_energies(x: np.ndarray) -> np.ndarray:
    N = len(x)
    J = int(round(math.log2(N)))
    if 2**J != N:
        raise ValueError("N must be a power of two")
    out = []
    for j in range(J):
        d = project(x, j+1) - project(x, j)
        out.append(float(np.mean(d*d)))
    return np.asarray(out)


def fwht(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).copy()
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, 2*h):
            x = a[i:i+h].copy()
            y = a[i+h:i+2*h].copy()
            a[i:i+h] = x + y
            a[i+h:i+2*h] = x - y
        h *= 2
    return a / math.sqrt(n)


def walsh_row(N: int, row: int) -> np.ndarray:
    e = np.zeros(N)
    e[row % N] = math.sqrt(N)
    return np.sign(fwht(e))


def rescale_energy(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    a = math.sqrt(norm2N(ref))
    b = math.sqrt(norm2N(x))
    return x if b < 1e-12 else x * (a / b)


def attack(s: np.ndarray, scenario: str, rng: np.random.Generator) -> np.ndarray:
    N = len(s)
    n = np.arange(N, dtype=float)
    noise = rng.normal(0, 0.02, size=N)
    if scenario == "additive_jamming":
        return s + rng.normal(0, 0.45, size=N)
    if scenario == "impulse_jamming":
        x = s.copy(); w = int(rng.integers(max(2, N//80), max(4, N//24))); i = int(rng.integers(0, N-w))
        x[i:i+w] += rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 2.2)
        return x + noise
    if scenario == "local_substitution":
        x = s.copy(); w = int(rng.integers(max(4, N//20), max(8, N//8))); i = int(rng.integers(0, N-w))
        x[i:i+w] = rng.uniform(-0.7, 0.7) + 0.15 * rng.normal(size=w)
        return x + noise
    if scenario == "fast_multiplicative":
        rho = rng.uniform(0.05, 0.18); f0 = rng.choice([8, 16, 32])
        q = np.sign(np.sin(2*np.pi*f0*n/N)); q[q == 0] = 1
        return (1 + rho*q) * s + noise
    if scenario == "walsh_multiplicative":
        rho = rng.uniform(0.05, 0.18); q = walsh_row(N, int(rng.integers(1, N))); q[q == 0] = 1
        return (1 + rho*q) * s + noise
    if scenario == "random_sign_multiplicative":
        return (1 + rng.uniform(0.03, 0.14) * rng.choice([-1.0, 1.0], size=N)) * s + noise
    if scenario == "slow_multiplicative":
        rho = rng.uniform(0.05, 0.18); f0 = rng.choice([1, 2]); ph = rng.uniform(0, 2*np.pi)
        return (1 + rho*np.sin(2*np.pi*f0*n/N + ph)) * s + noise
    if scenario == "energy_preserving_local":
        x = s.copy(); w = int(rng.integers(max(4, N//20), max(8, N//8))); i = int(rng.integers(0, N-w))
        x[i:i+w] += rng.uniform(-0.8, 0.8)
        return rescale_energy(x + noise, s)
    if scenario == "combined_attack":
        rho = rng.uniform(0.06, 0.18); f0 = rng.choice([8, 16, 32])
        q = np.sign(np.sin(2*np.pi*f0*n/N)); q[q == 0] = 1
        x = (1 + rho*q) * s
        w = int(rng.integers(max(3, N//40), max(6, N//15))); i = int(rng.integers(0, N-w))
        x[i:i+w] += rng.uniform(-0.8, 0.8)
        return x + noise
    raise ValueError(scenario)


def features(x: np.ndarray, r: int) -> Dict[str, float]:
    z = center(x)
    he = haar_energies(z)
    FH = float(np.sum(he[max(0, len(he)-r):]) / (norm2N(z) + 1e-12))
    w = fwht(z)
    p = (w*w) / (np.sum(w*w) + 1e-12)
    entropy = float(-np.sum(p*np.log(p + 1e-12)) / math.log(len(z)))
    effdim = float(1.0 / (np.sum(p*p) + 1e-12))
    fft = np.fft.rfft(z); spec = np.abs(fft)**2; cut = max(1, int(0.35*len(spec)))
    return {
        "Energy": norm2N(z),
        "TV": total_variation(z),
        "FFT-HF": float(np.sum(spec[cut:]) / (np.sum(spec) + 1e-12)),
        "Haar": FH,
        "WalshEntropy": entropy,
        "WalshEffDim": effdim,
        "Walsh": max(abs(entropy), abs(math.log(effdim + 1e-12) / math.log(len(z)))),
    }


def robust_fit(vals: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return med, 1.4826 * mad + 1e-12


def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=int); s = np.asarray(s, dtype=float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    order = np.argsort(s); ss = s[order]; ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and ss[j] == ss[i]: j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rp = float(np.sum(ranks[y == 1])); u = rp - len(pos)*(len(pos)+1)/2.0
    return u / (len(pos)*len(neg))


def roc_curve_np(y: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    th = np.r_[np.inf, np.sort(np.unique(s))[::-1], -np.inf]
    P = max(1, int(np.sum(y == 1))); N = max(1, int(np.sum(y == 0)))
    fpr = []; tpr = []
    for t in th:
        pred = s >= t
        tpr.append(float(np.sum(pred & (y == 1)) / P))
        fpr.append(float(np.sum(pred & (y == 0)) / N))
    return np.asarray(fpr), np.asarray(tpr)


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    a = np.sort(np.asarray(scores, dtype=float)); M = len(a)
    k = min(max(int(math.ceil((M + 1) * (1 - alpha))), 1), M)
    return float(a[k - 1])


def run_scenario(name: str, N: int, r: int, n_fit: int, n_cal: int, n0: int, n1: int, alpha: float, seed: int) -> Dict:
    rng = np.random.default_rng(seed + (sum(ord(c) for c in name) * 97) % 100000)
    fit = [clean_window(rng, N) for _ in range(n_fit)]
    cal = [clean_window(rng, N) for _ in range(n_cal)]
    test0 = [clean_window(rng, N) for _ in range(n0)]
    test1 = [attack(clean_window(rng, N), name, rng) for _ in range(n1)]
    f_fit = [features(x, r) for x in fit]
    f_cal = [features(x, r) for x in cal]
    f_test = [features(x, r) for x in test0 + test1]
    y = np.r_[np.zeros(n0, dtype=int), np.ones(n1, dtype=int)]
    params = {k: robust_fit(np.array([f[k] for f in f_fit])) for k in ["Energy", "TV", "FFT-HF", "Haar", "WalshEntropy", "WalshEffDim", "Walsh"]}
    def z(f, k):
        med, sc = params[k]
        return abs((f[k] - med) / sc)
    score_arrays = {}
    results = {}
    for k in ["Energy", "TV", "FFT-HF", "Haar", "Walsh"]:
        cs = np.array([z(f, k) for f in f_cal]); ts = np.array([z(f, k) for f in f_test])
        thr = conformal_threshold(cs, alpha); pred = ts > thr
        score_arrays[k] = ts
        results[k] = {"auc": auc_score(y, ts), "tpr": float(np.mean(pred[y == 1])), "fpr": float(np.mean(pred[y == 0])), "threshold": thr}
    def full(f): return max(z(f, "Haar"), z(f, "WalshEntropy"), z(f, "WalshEffDim"))
    cs = np.array([full(f) for f in f_cal]); ts = np.array([full(f) for f in f_test])
    thr = conformal_threshold(cs, alpha); pred = ts > thr
    score_arrays["Full"] = ts
    results["Full"] = {"auc": auc_score(y, ts), "tpr": float(np.mean(pred[y == 1])), "fpr": float(np.mean(pred[y == 0])), "threshold": thr}
    return {"scenario": name, "scores": results, "y": y, "score_arrays": score_arrays}


def write_tables(out: Path, rows: List[Dict]) -> None:
    with open(out / "benchmark_results.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n\\centering\n\\caption{Synthetic benchmark results generated by the public reproducibility script. Each cell reports AUC.}\n")
        f.write("\\label{tab:synthetic-benchmark-generated}\n\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("Scenario & Energy & TV & FFT-HF & Haar & Walsh & Full \\\\\n\\midrule\n")
        for r in rows:
            vals = [r["scores"][k]["auc"] for k in SCORES]
            f.write(LABELS[r["scenario"]] + " & " + " & ".join(f"{v:.3f}" for v in vals) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n")
    with open(out / "conformal_guarantee_check.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n\\centering\n\\caption{Split-conformal operating point for the full Haar--Walsh score.}\n")
        f.write("\\label{tab:conformal-generated}\n\\begin{tabular}{lrrr}\n\\toprule\nScenario & AUC & TPR at calibrated threshold & FPR \\\\\n\\midrule\n")
        for r in rows:
            s = r["scores"]["Full"]
            f.write(f"{LABELS[r['scenario']]} & {s['auc']:.3f} & {s['tpr']:.3f} & {s['fpr']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    with open(out / "theoretical_FH_bound_check.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n\\centering\n\\caption{Deterministic Haar-bound audit generated by the reproducibility script.}\n")
        f.write("\\label{tab:fh-bound-generated}\n\\begin{tabular}{lll}\n\\toprule\nCheck & Formula used & Expected interpretation \\\\\n\\midrule\n")
        f.write("Baseline upper bound & Eq.~(7.3) & conservative upper envelope \\\\\n")
        f.write("Attack lower bound & Eq.~(7.5) & sufficient, not necessary \\\\\n")
        f.write("Separation condition & Eq.~(7.6) & certification only when strict \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def write_timing(out: Path, N: int, r: int, seed: int) -> None:
    rng = np.random.default_rng(seed + 777)
    with open(out / "timing_benchmark.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n\\centering\n\\caption{Platform-dependent timing benchmark generated by the reproducibility script.}\n")
        f.write("\\label{tab:timing-generated}\n\\begin{tabular}{rrrr}\n\\toprule\n$N$ & Haar (s/window) & FWHT (s/window) & Full features (s/window) \\\\\n\\midrule\n")
        for NN in [128, 256, 512, 1024]:
            xs = [clean_window(rng, NN) for _ in range(80)]
            t0 = time.perf_counter(); [haar_energies(center(x)) for x in xs]; th = (time.perf_counter() - t0) / len(xs)
            t0 = time.perf_counter(); [fwht(center(x)) for x in xs]; tw = (time.perf_counter() - t0) / len(xs)
            t0 = time.perf_counter(); [features(x, min(r, int(round(math.log2(NN))))) for x in xs]; tf = (time.perf_counter() - t0) / len(xs)
            f.write(f"{NN} & {th:.3e} & {tw:.3e} & {tf:.3e} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def make_figures(out: Path, rows: List[Dict], N: int, r: int, seed: int) -> None:
    if plt is None: return
    plt.figure(figsize=(7, 5))
    for row in rows:
        y = row["y"]; s = row["score_arrays"]["Full"]
        fpr, tpr = roc_curve_np(y, s)
        plt.plot(fpr, tpr, label=f"{row['scenario']} (AUC={auc_score(y, s):.3f})")
    plt.plot([0, 1], [0, 1], "--", lw=1, label="random")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves for Haar--Walsh detector")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out / "roc_curves.pdf"); plt.close()

    rng = np.random.default_rng(seed + 123)
    prof = {"baseline": [], "fast multiplicative": [], "energy preserving local": []}
    for _ in range(250):
        s = clean_window(rng, N)
        prof["baseline"].append(haar_energies(center(s)))
        prof["fast multiplicative"].append(haar_energies(center(attack(s, "fast_multiplicative", rng))))
        prof["energy preserving local"].append(haar_energies(center(attack(s, "energy_preserving_local", rng))))
    plt.figure(figsize=(6, 4))
    for k, v in prof.items():
        m = np.mean(np.vstack(v), axis=0)
        plt.semilogy(np.arange(len(m)), m, marker="o", label=k)
    plt.xlabel("Haar level j"); plt.ylabel(r"Mean $E_{j,N}$"); plt.title("Haar scale energy profiles")
    plt.legend(); plt.tight_layout(); plt.savefig(out / "scale_energy_profiles.pdf"); plt.close()

    amps, freqs = [0.03, 0.08, 0.15], [2, 8, 32]
    grid = np.zeros((3, 3)); rng = np.random.default_rng(seed + 999)
    for i, rho in enumerate(amps):
        for j, f0 in enumerate(freqs):
            y = []; sc = []
            for lab in [0, 1]:
                for _ in range(250):
                    s = clean_window(rng, N)
                    x = s if lab == 0 else (1 + rho*np.sign(np.sin(2*np.pi*f0*np.arange(N)/N))) * s + rng.normal(0, 0.02, N)
                    sc.append(features(x, r)["Haar"] + features(x, r)["Walsh"]); y.append(lab)
            grid[i, j] = auc_score(np.array(y), np.array(sc))
    plt.figure(figsize=(5.4, 4)); im = plt.imshow(grid, origin="lower", aspect="auto")
    plt.colorbar(im, label="AUC"); plt.xticks(range(3), freqs); plt.yticks(range(3), amps)
    plt.xlabel("Modulation frequency f0"); plt.ylabel("Modulation amplitude rho"); plt.title("AUC heatmap for fast multiplicative modulation")
    plt.tight_layout(); plt.savefig(out / "heatmap_auc_fast_modulation.pdf"); plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("results_synthetic"))
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--r", type=int, default=6)
    ap.add_argument("--n-fit", type=int, default=600)
    ap.add_argument("--n-cal", type=int, default=600)
    ap.add_argument("--n-test0", type=int, default=1000)
    ap.add_argument("--n-test1", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.fast:
        args.n_fit = args.n_cal = 150; args.n_test0 = args.n_test1 = 250
    ensure_dir(args.out_dir)
    rows = [run_scenario(s, args.N, args.r, args.n_fit, args.n_cal, args.n_test0, args.n_test1, args.alpha, args.seed) for s in SCENARIOS]
    write_tables(args.out_dir, rows); write_timing(args.out_dir, args.N, args.r, args.seed); make_figures(args.out_dir, rows, args.N, args.r, args.seed)
    summary = {"config": vars(args), "results": [{"scenario": r["scenario"], "scores": r["scores"]} for r in rows]}
    (args.out_dir / "synthetic_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote synthetic outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
