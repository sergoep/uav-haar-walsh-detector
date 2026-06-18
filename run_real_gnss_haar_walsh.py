#!/usr/bin/env python3
"""
Real-data Haar--Walsh validation runner for public GNSS/RF I/Q datasets.

This script is intentionally dataset-flexible. It supports two workflows:

1) Download the public Zenodo record 4629685 (Raw IQ dataset for GNSS GPS
   jamming signal classification) and process the extracted files.
2) Process an already downloaded directory of I/Q files from Zenodo/TEXBAT/FGI/Tuni.

The output is a set of LaTeX tables and a ROC figure that can be included in
uav_haar_walsh_final_revised_realdata.tex.

Examples:
    python run_real_gnss_haar_walsh.py --download-zenodo4629685 --data-dir data/raw_iq_zenodo --extract
    python run_real_gnss_haar_walsh.py --data-dir data/raw_iq_zenodo --out-dir results_real --N 256 --max-windows-per-class 2000
    python run_real_gnss_haar_walsh.py --data-dir /path/to/TEXBAT --raw-dtype int16 --iq-mode magnitude --N 512

Notes:
    - The script does not fabricate results. If no readable files are found, it exits with a clear error.
    - Labels are inferred from directory/file names. You can override them with a CSV manifest via --manifest.
    - For very large raw files, use --max-samples-per-file to limit memory use.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

ZENODO_4629685_FILES_API = "https://zenodo.org/api/records/4629685"

CLEAN_PATTERNS = [
    r"no[_\- ]?jam", r"nojamm", r"clean", r"clear", r"authentic", r"normal",
    r"no[_\- ]?jamming", r"no[_\- ]?spoof", r"without[_\- ]?jam",
]
ATTACK_PATTERNS = [
    r"jam", r"jamming", r"dme", r"narrow", r"chirp", r"fm", r"am",
    r"spoof", r"spoofer", r"meacon", r"interference", r"rfi", r"attack",
]


@dataclass
class WindowRecord:
    x: np.ndarray
    y: int
    scenario: str
    source: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_zenodo4629685(data_dir: Path, extract: bool = False) -> None:
    """Download Zenodo record 4629685 using only stdlib+requests if available."""
    ensure_dir(data_dir)
    try:
        import requests
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("The requests package is required for downloading. Install it or download manually.") from exc

    print(f"Fetching metadata: {ZENODO_4629685_FILES_API}")
    r = requests.get(ZENODO_4629685_FILES_API, timeout=60)
    r.raise_for_status()
    meta = r.json()
    files = meta.get("files", [])
    if not files:
        raise RuntimeError("No files found in Zenodo metadata.")
    for f in files:
        key = f.get("key") or f.get("filename") or "downloaded_file"
        if key != "Raw_IQ_Dataset.zip" and not key.endswith(".m"):
            continue
        url = (f.get("links") or {}).get("self") or (f.get("links") or {}).get("download")
        if not url:
            continue
        out = data_dir / key
        if out.exists() and out.stat().st_size > 0:
            print(f"Already present: {out}")
            continue
        print(f"Downloading {key} -> {out}")
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(out, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=2**20):
                    if chunk:
                        fh.write(chunk)
    if extract:
        zip_path = data_dir / "Raw_IQ_Dataset.zip"
        if zip_path.exists():
            extract_dir = data_dir / "Raw_IQ_Dataset_extracted"
            ensure_dir(extract_dir)
            print(f"Extracting {zip_path} -> {extract_dir}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)


def infer_label_and_scenario(path: Path) -> Optional[Tuple[int, str]]:
    s = str(path).lower()
    clean = any(re.search(p, s) for p in CLEAN_PATTERNS)
    attack = any(re.search(p, s) for p in ATTACK_PATTERNS)
    # If both match, specific no-jamming patterns win for clean.
    if clean:
        return 0, "clean/no-jamming"
    if attack:
        if re.search(r"dme", s):
            return 1, "DME jamming"
        if re.search(r"narrow", s):
            return 1, "narrowband jamming"
        if re.search(r"chirp", s):
            return 1, "chirp jamming"
        if re.search(r"\bfm\b|single[_\- ]?fm", s):
            return 1, "FM jamming"
        if re.search(r"\bam\b|single[_\- ]?am", s):
            return 1, "AM jamming"
        if re.search(r"spoof", s):
            return 1, "spoofing"
        if re.search(r"meacon", s):
            return 1, "meaconing"
        return 1, "interference/jamming"
    return None


def load_manifest(manifest: Path) -> List[Tuple[Path, int, str]]:
    rows: List[Tuple[Path, int, str]] = []
    with open(manifest, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"path", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Manifest must contain columns: path,label[,scenario]")
        base = manifest.parent
        for row in reader:
            p = Path(row["path"])
            if not p.is_absolute():
                p = base / p
            y = int(row["label"])
            scenario = row.get("scenario") or ("clean" if y == 0 else "attack")
            rows.append((p, y, scenario))
    return rows


def discover_files(data_dir: Path, manifest: Optional[Path]) -> List[Tuple[Path, int, str]]:
    if manifest is not None:
        return load_manifest(manifest)
    exts = {".npy", ".npz", ".mat", ".csv", ".txt", ".bin", ".dat", ".iq", ".cfile"}
    out: List[Tuple[Path, int, str]] = []
    for p in data_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        inferred = infer_label_and_scenario(p)
        if inferred is None:
            continue
        y, scenario = inferred
        out.append((p, y, scenario))
    return out


def _load_mat(path: Path) -> np.ndarray:
    try:
        import scipy.io as sio
    except Exception as exc:
        raise RuntimeError("scipy is required to read .mat files. Install scipy or convert the file to .npy/.csv.") from exc
    obj = sio.loadmat(path)
    arrays = []
    for k, v in obj.items():
        if k.startswith("__"):
            continue
        arr = np.asarray(v)
        if arr.size >= 32 and np.issubdtype(arr.dtype, np.number):
            arrays.append(arr.squeeze())
    if not arrays:
        raise ValueError(f"No numeric arrays found in {path}")
    arrays.sort(key=lambda a: a.size, reverse=True)
    return arrays[0]


def load_numeric_file(path: Path, raw_dtype: str, max_samples_per_file: Optional[int]) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path, mmap_mode="r")
        arr = np.asarray(arr)
    elif ext == ".npz":
        z = np.load(path)
        keys = sorted(z.files, key=lambda k: np.asarray(z[k]).size, reverse=True)
        arr = np.asarray(z[keys[0]])
    elif ext == ".mat":
        arr = _load_mat(path)
    elif ext in {".csv", ".txt"}:
        arr = np.loadtxt(path, delimiter="," if ext == ".csv" else None)
    else:
        dt = np.dtype(raw_dtype)
        count = None if max_samples_per_file is None else int(max_samples_per_file) * 2
        arr = np.fromfile(path, dtype=dt, count=-1 if count is None else count)
    arr = np.asarray(arr)
    if max_samples_per_file is not None and arr.size > max_samples_per_file * 4:
        arr = arr.reshape(-1)[: max_samples_per_file * 4]
    return arr


def to_real_window_signal(arr: np.ndarray, iq_mode: str) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if np.iscomplexobj(arr):
        c = arr.reshape(-1)
    else:
        if arr.ndim >= 2 and min(arr.shape[-2:]) == 2:
            a = arr.reshape(-1, 2)
            c = a[:, 0].astype(float) + 1j * a[:, 1].astype(float)
        else:
            flat = arr.reshape(-1)
            if flat.size >= 2 and flat.size % 2 == 0:
                # For raw I/Q datasets, interleaved I,Q is the most common convention.
                c = flat[0::2].astype(float) + 1j * flat[1::2].astype(float)
            else:
                c = flat.astype(float)
    if iq_mode == "magnitude":
        y = np.abs(c)
    elif iq_mode == "i":
        y = np.real(c)
    elif iq_mode == "q":
        y = np.imag(c)
    elif iq_mode == "power":
        y = np.abs(c) ** 2
    else:
        raise ValueError(f"Unknown iq_mode: {iq_mode}")
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    return y


def segment_signal(y: np.ndarray, N: int, stride: int, max_windows: int) -> List[np.ndarray]:
    if y.size < N:
        return []
    wins = []
    for start in range(0, y.size - N + 1, stride):
        x = np.asarray(y[start : start + N], dtype=float)
        if np.all(np.isfinite(x)) and np.std(x) > 0:
            wins.append(x)
        if len(wins) >= max_windows:
            break
    return wins


def center(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)


def total_variation(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))


def haar_detail_energies(x: np.ndarray) -> np.ndarray:
    """Return normalized Haar detail energies for levels j=0..J-1."""
    x = np.asarray(x, dtype=float)
    N = x.size
    J = int(round(math.log2(N)))
    if 2**J != N:
        raise ValueError("N must be a power of two")
    def project(arr: np.ndarray, j: int) -> np.ndarray:
        block = N // (2**j)
        return arr.reshape(2**j, block).mean(axis=1).repeat(block)
    e = []
    for j in range(J):
        d = project(x, j + 1) - project(x, j)
        e.append(float(np.mean(d * d)))
    return np.array(e, dtype=float)


def fwht(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).copy()
    n = a.size
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            x = a[i : i + h].copy()
            y = a[i + h : i + 2 * h].copy()
            a[i : i + h] = x + y
            a[i + h : i + 2 * h] = x - y
        h *= 2
    return a / math.sqrt(n)


def feature_vector(x: np.ndarray, r: int = 3, delta: float = 1e-12) -> Dict[str, float]:
    z = center(x)
    N = z.size
    norm2 = float(np.mean(z * z))
    he = haar_detail_energies(z)
    fine = float(np.sum(he[max(0, len(he)-r):]))
    F_H = fine / (norm2 + delta)
    w = fwht(z)
    p = (w * w) / (np.sum(w * w) + delta)
    entropy = float(-np.sum(p * np.log(p + delta)) / math.log(N))
    effdim = float(1.0 / (np.sum(p * p) + delta))
    fft = np.fft.rfft(z)
    spec = np.abs(fft) ** 2
    cut = max(1, int(0.35 * spec.size))
    fft_hf = float(np.sum(spec[cut:]) / (np.sum(spec) + delta))
    return {
        "Energy": float(np.mean(x * x)),
        "TV": total_variation(z),
        "FFT-HF": fft_hf,
        "Haar": F_H,
        "WalshEntropy": entropy,
        "WalshEffDim": effdim,
    }


def robust_fit(values: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, 1.4826 * mad + 1e-12


def auc_score(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Mann-Whitney U with average ranks for ties.
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=float)
    i = 0
    while i < scores.size:
        j = i + 1
        while j < scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg
        i = j
    rank_pos = np.sum(ranks[y == 1])
    u = rank_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    cal_scores = np.sort(np.asarray(cal_scores, dtype=float))
    M = cal_scores.size
    k = int(math.ceil((M + 1) * (1 - alpha)))
    k = min(max(k, 1), M)
    return float(cal_scores[k - 1])


def roc_curve_np(y: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    P = max(1, int(np.sum(y == 1)))
    N = max(1, int(np.sum(y == 0)))
    tpr = []
    fpr = []
    for t in thresholds:
        pred = scores >= t
        tpr.append(np.sum(pred & (y == 1)) / P)
        fpr.append(np.sum(pred & (y == 0)) / N)
    return np.array(fpr), np.array(tpr)


def bootstrap_auc_ci(y: np.ndarray, scores: np.ndarray, B: int, rng: np.random.Generator) -> Tuple[float, float]:
    if B <= 0:
        return float("nan"), float("nan")
    vals = []
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if idx_pos.size == 0 or idx_neg.size == 0:
        return float("nan"), float("nan")
    for _ in range(B):
        ip = rng.choice(idx_pos, size=idx_pos.size, replace=True)
        ineg = rng.choice(idx_neg, size=idx_neg.size, replace=True)
        idx = np.r_[ip, ineg]
        vals.append(auc_score(y[idx], scores[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def write_latex_tables(out_dir: Path, dataset_name: str, summary: Dict, results: List[Dict]) -> None:
    ensure_dir(out_dir)
    # Dataset summary.
    with open(out_dir / "real_data_dataset_summary.tex", "w", encoding="utf-8") as fh:
        fh.write("\\begin{table}[H]\n\\centering\n")
        fh.write("\\caption{Public real-data validation source and window counts.}\n")
        fh.write("\\label{tab:real-data-summary}\n")
        fh.write("\\begin{tabular}{lrrr}\n\\toprule\n")
        fh.write("Dataset & Clean windows & Attack windows & Window length \\\\\n\\midrule\n")
        fh.write(f"{dataset_name} & {summary['clean_windows']} & {summary['attack_windows']} & {summary['N']} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    # Result table.
    with open(out_dir / "real_data_results.tex", "w", encoding="utf-8") as fh:
        fh.write("\\begin{table}[H]\n\\centering\n")
        fh.write("\\caption{Real-data validation results generated by the public Python protocol. The split-conformal threshold is fitted only on clean calibration windows.}\n")
        fh.write("\\label{tab:real-data-results}\n")
        fh.write("\\resizebox{\\linewidth}{!}{%\n")
        fh.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        fh.write("Score & AUC & 95\\% bootstrap CI & Conformal threshold & FPR & TPR \\\\\n\\midrule\n")
        for r in results:
            ci = "--" if math.isnan(r["auc_lo"]) else f"[{r['auc_lo']:.3f}, {r['auc_hi']:.3f}]"
            fh.write(f"{r['score']} & {r['auc']:.3f} & {ci} & {r['threshold']:.3g} & {r['fpr']:.3f} & {r['tpr']:.3f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n")


def plot_roc(out_dir: Path, y: np.ndarray, score_arrays: Dict[str, np.ndarray]) -> None:
    if plt is None:
        return
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    plt.figure(figsize=(6.0, 4.2))
    for name, scores in score_arrays.items():
        fpr, tpr = roc_curve_np(y, scores)
        auc = auc_score(y, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Real-data ROC curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "realdata_roc_curves.pdf")
    plt.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw_iq"))
    ap.add_argument("--out-dir", type=Path, default=Path("results_real"))
    ap.add_argument("--manifest", type=Path, default=None, help="CSV with columns path,label[,scenario]")
    ap.add_argument("--download-zenodo4629685", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--max-windows-per-file", type=int, default=500)
    ap.add_argument("--max-windows-per-class", type=int, default=3000)
    ap.add_argument("--max-samples-per-file", type=int, default=2_000_000)
    ap.add_argument("--raw-dtype", default="float32", help="dtype for raw .bin/.dat/.iq files")
    ap.add_argument("--iq-mode", choices=["magnitude", "i", "q", "power"], default="magnitude")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--dataset-name", default="Public GNSS/RF I/Q dataset")
    args = ap.parse_args(argv)

    if 2 ** int(round(math.log2(args.N))) != args.N:
        raise ValueError("--N must be a power of two")

    if args.download_zenodo4629685:
        download_zenodo4629685(args.data_dir, extract=args.extract)

    files = discover_files(args.data_dir, args.manifest)
    if not files:
        raise RuntimeError(
            "No labeled numeric files found. Either extract the dataset, provide --manifest, "
            "or rename directories/files so clean/no-jamming and jam/spoof classes are identifiable."
        )

    rng = np.random.default_rng(args.seed)
    windows: List[WindowRecord] = []
    per_class = {0: 0, 1: 0}
    skipped = []
    for p, y, scenario in files:
        if per_class[y] >= args.max_windows_per_class:
            continue
        try:
            arr = load_numeric_file(p, args.raw_dtype, args.max_samples_per_file)
            sig = to_real_window_signal(arr, args.iq_mode)
            max_for_file = min(args.max_windows_per_file, args.max_windows_per_class - per_class[y])
            wins = segment_signal(sig, args.N, args.stride, max_for_file)
            for w in wins:
                windows.append(WindowRecord(w, y, scenario, str(p)))
            per_class[y] += len(wins)
        except Exception as exc:
            skipped.append((str(p), str(exc)))

    if per_class[0] < 20 or per_class[1] < 20:
        details = "\n".join([f"  - {p}: {e}" for p, e in skipped[:10]])
        raise RuntimeError(
            f"Insufficient windows: clean={per_class[0]}, attack={per_class[1]}. "
            f"Try --manifest, --raw-dtype, --iq-mode, smaller --N, or larger --max-samples-per-file.\nSkipped examples:\n{details}"
        )

    # Split clean windows into calibration and test. All attack windows are test.
    clean_idx = np.array([i for i, r in enumerate(windows) if r.y == 0])
    attack_idx = np.array([i for i, r in enumerate(windows) if r.y == 1])
    rng.shuffle(clean_idx)
    n_cal = max(20, int(0.5 * clean_idx.size))
    cal_idx = clean_idx[:n_cal]
    test_idx = np.r_[clean_idx[n_cal:], attack_idx]
    rng.shuffle(test_idx)

    feats = [feature_vector(r.x) for r in windows]
    base_names = ["Energy", "TV", "FFT-HF", "Haar", "WalshEntropy", "WalshEffDim"]
    # Robustly normalized full score fitted on clean calibration only.
    params = {name: robust_fit(np.array([feats[i][name] for i in cal_idx])) for name in base_names}
    for f in feats:
        zvals = [abs((f[name] - params[name][0]) / params[name][1]) for name in base_names]
        f["Full"] = float(max(zvals))

    score_names = ["Energy", "TV", "FFT-HF", "Haar", "WalshEntropy", "WalshEffDim", "Full"]
    y_test = np.array([windows[i].y for i in test_idx], dtype=int)
    results = []
    score_arrays = {}
    for name in score_names:
        cal_scores = np.array([feats[i][name] for i in cal_idx], dtype=float)
        test_scores = np.array([feats[i][name] for i in test_idx], dtype=float)
        if name != "Full":
            med, sc = robust_fit(cal_scores)
            cal_scores_eval = np.abs((cal_scores - med) / sc)
            test_scores_eval = np.abs((test_scores - med) / sc)
        else:
            cal_scores_eval = cal_scores
            test_scores_eval = test_scores
        thr = conformal_threshold(cal_scores_eval, args.alpha)
        pred = test_scores_eval > thr
        fpr = float(np.mean(pred[y_test == 0])) if np.any(y_test == 0) else float("nan")
        tpr = float(np.mean(pred[y_test == 1])) if np.any(y_test == 1) else float("nan")
        auc = auc_score(y_test, test_scores_eval)
        lo, hi = bootstrap_auc_ci(y_test, test_scores_eval, args.bootstrap, rng)
        results.append({"score": name, "auc": auc, "auc_lo": lo, "auc_hi": hi, "threshold": thr, "fpr": fpr, "tpr": tpr})
        score_arrays[name] = test_scores_eval

    ensure_dir(args.out_dir)
    summary = {
        "dataset_name": args.dataset_name,
        "clean_windows": int(per_class[0]),
        "attack_windows": int(per_class[1]),
        "calibration_clean_windows": int(cal_idx.size),
        "test_windows": int(test_idx.size),
        "N": int(args.N),
        "stride": int(args.stride),
        "iq_mode": args.iq_mode,
        "raw_dtype": args.raw_dtype,
        "skipped_files": skipped[:25],
    }
    write_latex_tables(args.out_dir, args.dataset_name, summary, results)
    plot_roc(args.out_dir, y_test, score_arrays)
    with open(args.out_dir / "real_data_run_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "results": results}, fh, indent=2)
    print(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"Wrote LaTeX outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
