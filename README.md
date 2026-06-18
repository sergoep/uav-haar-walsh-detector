# UAV Haar--Walsh Real-Data Validation Runner

This repository contains the reproducible Python code associated with the paper:

**Calibrated Haar--Walsh Structural Features for Lightweight Interference Detection in Short UAV Signal Windows**

The repository provides a real-data validation runner for public GNSS/RF I/Q datasets. The script implements Haar--Walsh structural feature extraction, robust median/MAD normalization, split-conformal thresholding, ROC/AUC evaluation, bootstrap confidence intervals, and LaTeX/figure output generation.

## Contents

* `run_real_gnss_haar_walsh.py` — real-data validation runner for public GNSS/RF I/Q datasets.
* `requirements.txt` — Python dependencies.
* `LICENSE` — MIT license.
* `README.md` — repository documentation.

## What the script does

The script supports two workflows:

1. Download and process Zenodo record `4629685`, the public Raw IQ dataset for GNSS GPS jamming signal classification.
2. Process an already downloaded directory of public I/Q files from Zenodo, TEXBAT-style datasets, FGI repositories, Tuni datasets, or other compatible GNSS/RF I/Q sources.

The script generates:

* `results_real/real_data_dataset_summary.tex`
* `results_real/real_data_results.tex`
* `results_real/real_data_run_summary.json`
* `results_real/figures/realdata_roc_curves.pdf`

## Reproducibility

Python version: 3.10 or later.

Install dependencies:

```bash
pip install -r requirements.txt
```

Download and process Zenodo record 4629685:

```bash
python run_real_gnss_haar_walsh.py --download-zenodo4629685 --data-dir data/raw_iq_zenodo --extract
python run_real_gnss_haar_walsh.py --data-dir data/raw_iq_zenodo --out-dir results_real --N 256 --max-windows-per-class 2000
```

Process a manually downloaded I/Q dataset:

```bash
python run_real_gnss_haar_walsh.py --data-dir /path/to/dataset --out-dir results_real --N 256
```

Process a dataset using a CSV manifest:

```bash
python run_real_gnss_haar_walsh.py --manifest manifest.csv --out-dir results_real --N 256
```

The manifest must contain at least:

```text
path,label
```

where `label = 0` denotes clean/no-jamming windows and `label = 1` denotes attack/interference/spoofing/jamming windows.

## Scope and safe use

This repository is limited to offline defensive analysis of already available public GNSS/RF I/Q recordings.

It does not provide RF emission parameters, jamming procedures, spoofing procedures, UAV control-channel manipulation, real-time disruption methods, SDR attack scripts, or operational deployment instructions.

## Data

Large public raw I/Q datasets are not redistributed in this repository. Users should download them from their original public sources and process them locally with the provided script.

## Citation

If you use this code, please cite the corresponding paper and the archived Zenodo release once available.

Zenodo DOI: to be added after the first public release.

## License

This project is released under the MIT License.
