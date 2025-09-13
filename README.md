# GDEM — Reproduction (Graph Distillation via Eigenbasis Matching)

This repo reproduces **GDEM** (Graph Distillation via Eigenbasis Matching): given a large graph, we distill a small **synthetic graph** \((U', X')\) whose spectral (graph-Fourier) characteristics match the original, so standard GNNs trained on the synthetic graph reach similar accuracy at a fraction of the cost.

**Pipeline**
1) Precompute Laplacian eigenpairs  
2) Initialize synthetic features  
3) Run distillation (GDEM)  
4) Evaluate multiple GNN backbones on the distilled graph

---

## Environment

- OS: Ubuntu 22.04 (Linux 5.15)
- Python: **3.12** (avoid 3.13 due to C-extensions)
- CUDA GPU: works with **16 GB VRAM** instead of **NVIDIA A800 80GB PCIe** in their paper.

Quick setup:
```bash
python3 -m venv .gdem && source .gdem/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric ogb deeprobust numpy scipy scikit-learn tqdm networkx
```

---

## Repo layout

- `preprocess.py` — dataset splits, LCC; **precompute eigenpairs** → `./data/<dataset>/eigenvalues.npy`, `eigenvectors.npy`
- `feat_init.py` — initialize synthetic features → `./initial_feat/<dataset>/x_init_<rate>_<expID>.pt`
- `distill.py` — run **GDEM**; prints `Mean ACC ± Std`
- `test_other_arcs.py` — evaluate **GCN / SGC / APPNP (PPNP) / ChebNet / BernNet / GPR-GNN** on distilled graph
- `config/config_distill.json` — per-dataset defaults (e.g., `eigen_k`, `ratio`, lrs)

Supported datasets: `citeseer, pubmed, ogbn-arxiv, flickr, reddit, squirrel`

> **Squirrel fetch tip**: if PyG 404s the raw file, use the preprocessed variant or place `squirrel.npz` under `data/squirrel/raw/`.

---

## Reproduction steps

### 1) Precompute eigenpairs
For large graphs (arxiv/reddit), the script uses sparse partial eigensolves—do **not** densify.
```bash
python preprocess.py --dataset citeseer
python preprocess.py --dataset pubmed
python preprocess.py --dataset squirrel
python preprocess.py --dataset flickr
python preprocess.py --dataset ogbn-arxiv
python preprocess.py --dataset reddit
```

### 2) Initialize synthetic features
```bash
# example
python feat_init.py --dataset citeseer --reduction_rate 0.25 --gpu_id 0
```

### 3) Distill (GDEM)
```bash
# Citeseer
python distill.py --dataset citeseer --reduction_rate 0.25 --gpu_id 0
python distill.py --dataset citeseer --reduction_rate 0.50 --gpu_id 0

# Pubmed
python distill.py --dataset pubmed --reduction_rate 0.25 --gpu_id 0
python distill.py --dataset pubmed --reduction_rate 0.50 --gpu_id 0
python distill.py --dataset pubmed --reduction_rate 1.00 --gpu_id 0

# Squirrel / Flickr
python distill.py --dataset squirrel --reduction_rate 0.25 --gpu_id 0
python distill.py --dataset squirrel --reduction_rate 0.50 --gpu_id 0
python distill.py --dataset squirrel --reduction_rate 1.00 --gpu_id 0

python distill.py --dataset flickr  --reduction_rate 0.001 --gpu_id 0
python distill.py --dataset flickr  --reduction_rate 0.005 --gpu_id 0
python distill.py --dataset flickr  --reduction_rate 0.010 --gpu_id 0

# OGBN-Arxiv / Reddit
python distill.py --dataset ogbn-arxiv --reduction_rate 0.001 --gpu_id 0
python distill.py --dataset ogbn-arxiv --reduction_rate 0.005 --gpu_id 0
python distill.py --dataset ogbn-arxiv --reduction_rate 0.010 --gpu_id 0

python distill.py --dataset reddit --reduction_rate 0.0005 --gpu_id 0
python distill.py --dataset reddit --reduction_rate 0.0010 --gpu_id 0
python distill.py --dataset reddit --reduction_rate 0.0050 --gpu_id 0
```

### 4) Cross-architecture evaluation
```bash
# GCN (default)
python test_other_arcs.py --dataset citeseer --reduction_rate 0.25 --gpu_id 0 --test_model GCN

# Others (examples)
python test_other_arcs.py --dataset citeseer --reduction_rate 0.25 --gpu_id 0 --test_model SGC --k 2
python test_other_arcs.py --dataset pubmed   --reduction_rate 0.25 --gpu_id 0 --test_model APPNP --k 10 --alpha 0.1
python test_other_arcs.py --dataset citeseer --reduction_rate 0.25 --gpu_id 0 --test_model ChebNet --k 2
python test_other_arcs.py --dataset pubmed   --reduction_rate 0.25 --gpu_id 0 --test_model BernNet --k 5 --lr_conv 0.01 --wd_conv 5e-4 --dprate 0.0
python test_other_arcs.py --dataset citeseer --reduction_rate 0.25 --gpu_id 0 --test_model GPRGNN --k 10 --alpha 0.1
```
Backbone aliases: `GCN`, `SGC`, `APPNP`, `ChebNet`, `BernNet`, `GPRGNN`.  
Useful flags: `--k` (order), `--alpha` (APPNP), `--lr_conv/--wd_conv/--dprate` (Bern/GPR).

---

## Hyperparameters (what matters most)

- `eigen_k` — total spectral modes cached (low+high). Bigger = better fidelity, more memory.
- `ratio` — share of **low-frequency** modes (e.g., `0.8` → 80% low-freq, 20% high-freq).  
  Heterophilous graphs (Squirrel/Flickr) often like **more high-freq** (lower `ratio`).
- `epoch`, `e1/e2` — outer epochs and inner steps (features / eigenbasis).
- `alpha/beta/gamma` — weights for eigen-match, classifier, orthogonality.
- `lr_feat`, `lr_eigenvec`, `lr_gnn`, `wd_gnn`, `dropout`, `hidden_dim`, `nlayers` — standard training knobs.

---

## Memory & speed tips (16 GB friendly)

- Compute eigenpairs in **FP64**, save **FP32** (stable solve, half the size).
- Reduce `eigen_k` and/or `hidden_dim` (e.g., 128 or 64) if VRAM is tight.
- For large graphs (arxiv/reddit): **sparse partial** eigensolves only; never densify the Laplacian.
- If `CUDA error: invalid device ordinal` → use `--gpu_id 0`.
- If `x_init_*.pt` missing → run `feat_init.py` for the same `dataset/reduction_rate/expID`.

---

## Results

> In tables below, *our reduction → paper r* maps our `reduction_rate` to the paper’s reported r%.  
> Metrics are **mean ± std** over 10 runs. **Δ** is ours − paper (pp).

### Table 1 — Distillation accuracy (ours vs. paper)

| Dataset (our reduction → paper r) | Our distill | Paper distill | Δ |
|---|---:|---:|---:|
| Citeseer (0.25 → 0.90%) | 72.18 ± 0.54 | 72.30 ± 0.30 | −0.12 |
| Citeseer (0.50 → 1.80%) | 71.64 ± 0.23 | 72.60 ± 0.60 | −0.96 |
| Pubmed (0.25 → 0.08%) | 77.75 ± 0.59 | 77.70 ± 0.70 | +0.05 |
| Pubmed (0.50 → 0.15%) | 78.36 ± 2.05 | 78.40 ± 1.80 | −0.04 |
| Pubmed (1.00 → 0.30%) | 78.13 ± 0.87 | 78.20 ± 0.80 | −0.07 |
| Squirrel (0.25 → 0.60%) | 29.11 ± 2.62 | 28.40 ± 2.00 | +0.71 |
| Squirrel (0.50 → 1.20%) | 28.48 ± 2.88 | 28.20 ± 2.40 | +0.28 |
| Squirrel (1.00 → 2.50%) | 26.35 ± 2.99 | 27.80 ± 1.60 | −1.45 |
| ogbn-arxiv (0.001 → 0.05%) | 63.76 ± 0.89 | 63.70 ± 0.80 | +0.06 |
| ogbn-arxiv (0.005 → 0.25%) | 63.89 ± 0.60 | 63.80 ± 0.60 | +0.09 |
| ogbn-arxiv (0.01 → 0.50%) | 64.08 ± 0.33 | 64.10 ± 0.30 | −0.02 |
| Flickr (0.001 → 0.10%) | 49.28 ± 2.36 | 49.90 ± 0.80 | −0.62 |
| Flickr (0.005 → 0.50%) | 48.74 ± 2.18 | 49.40 ± 1.30 | −0.66 |
| Flickr (0.01 → 1.00%) | 49.34 ± 2.35 | 49.90 ± 0.60 | −0.56 |
| Reddit (0.0005 → 0.05%) | 92.88 ± 0.26 | 92.90 ± 0.30 | −0.02 |
| Reddit (0.001 → 0.10%) | 93.05 ± 0.20 | 93.10 ± 0.20 | −0.05 |
| Reddit (0.005 → 0.50%) | 93.23 ± 0.40 | 93.20 ± 0.40 | +0.03 |

### Table 2 — Cross-architecture evaluation (ours vs. paper)
*(Average of six backbones; std across those means; only rows with paper values.)*

| Dataset (reduction_rate) | Our AVG (std) | Paper AVG (std) | Δ |
|---|---:|---:|---:|
| Citeseer (0.25 → maps to paper r=1.80%) | 72.04 (0.14) | 72.68 (0.51) | −0.64 |
| Pubmed (0.50 → 0.15%) | 78.43 (0.50) | 77.92 (0.83) | +0.51 |
| Squirrel (0.50 → 1.20%) | 24.67 (1.67) | 27.22 (1.09) | −2.55 |
| ogbn-arxiv (0.005 → 0.50%) | 59.09 (3.40) | 63.02 (0.69) | −3.93 |
| Flickr (0.005 → 0.50%) | 44.57 (2.66) | 49.33 (0.60) | −4.76 |
| Reddit (0.001 → 0.10%) | 89.45 (2.30) | 91.47 (1.35) | −2.02 |

**3 key insights**
- **Homophily vs. heterophily:** Ratios help on homophilous graphs (Reddit/Arxiv). Heterophilous (Squirrel/Flickr) benefit from **more high-freq modes** (lower `ratio`, larger `k2`).
- **Small gaps are explainable:** Citeseer@1.8% (−0.96 pp) and other deltas align with **VRAM/config** limits and **separate training loops / seeds**; overall match is strong.
- **Quick eval gains:** Tune eval a bit—SGC `k=1–3`, APPNP `α=0.1–0.2`, Cheb/Bern smaller `k` + lower `lr`. For stability, **FP64 eigensolve → save FP32**.

---

## Troubleshooting

- **CUDA: invalid device ordinal** → `--gpu_id 0`
- **Missing `x_init_*.pt`** → run `feat_init.py` for the same `dataset/reduction_rate/expID`
- **Missing eigen `.npy`** → run `preprocess.py` first (per dataset)
- **OOM / huge dense arrays** → use **sparse partial** eigenpairs; avoid `toarray()` on large graphs
- **Numerical drift** → compute eigenpairs in **float64**, then save **float32**

---

## Acknowledgements

- This repo reproduces the **GDEM** method. Please cite the original paper/authors when using these results.
- Thanks to the PyTorch Geometric, OGB, and Deeprobust communities.

**Happy distilling!**
