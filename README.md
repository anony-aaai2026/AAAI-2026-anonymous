# AAAI-2026-anonymous

This repository contains code to reproduce the results in our AAAI 2026 submission.

## 📄 Overview

We implement the full pipeline for **online cost-aware active regression** (referred to as **QueryMarket** in the paper), including:

- Full implementation of Algorithms 1–3 from the main paper:
  - **OVBAL** (Our proposed method)
  - **BIA** (Buy-it-all baseline)
  - **RS** (Random Sampling baseline)
- **Synthetic data simulation** (`toy_example.py`)
- **Real-world data experiment** using the CASP dataset (`real_world_data.py`)

The scripts reproduce all key figures and tables from the main paper, including:
- **Figure 2, 3, 4a** from `toy_example.py`
- **Figure 4b, 5, Table 2** from `real_world_data.py`
- Additional analysis plots can be reproduced by uncommenting blocks at the end of the scripts

> ℹ️ The random seed is fixed (`np.random.seed(42)`) to ensure reproducibility.

---

## 📂 File Structure

| File | Description |
|------|-------------|
| `toy_example.py` | Main script for the toy synthetic experiment (Figures 2, 3, 4a, Table 1) |
| `real_world_data.py` | Script for CASP dataset experiment (Figures 4b, 5, Table 2) |
| `requirements.txt` | Required Python dependencies |
| `README.md` | This file |

---

## 📦 Installation

**Requirements:**
- Python 3.11.7

Install required packages via:

```bash
pip install -r requirements.txt

```
## ▶️ Running the Code

### Toy example:
```bash
python toy_example.py

```

- Real world data:
```
python real_world_data.py
```

This code uses the CASP protein structure dataset.

- **Filename expected:** `dataset/CASP.csv`
- **Source:** [CASP dataset from UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure)
- **License:** Open-access (UCI datasets are generally free for academic use)
- **Format:** CSV file with physicochemical features for protein tertiary structure prediction

To run the code successfully:
1. Download the CSV from the UCI repository  
2. Create a directory `dataset/` if it doesn't exist  
3. Place the file at `dataset/CASP.csv`


## 🔁 Algorithm ↔ Code Mapping

| Paper Component                 | Code Location                                      | Description                                |
|--------------------------------|---------------------------------------------------|--------------------------------------------|
| Algorithm 1 (OVBAL)            | `online_active_learning()`                        | Main method with threshold-based query     |
| Algorithm 2 (BIA)              | `online_active_learning()`                        | Buy-it-all baseline                        |
| Algorithm 3 (RS)               | `online_active_learning()`                        | Random sampling                            |
| Eq. (8b) – OVBAL threshold update | `update_tau_online()`                           | UPV threshold τₜ update                    |
| Figure 2                       | `compute_per_feature_tracking_error()`            | Per-feature MSE plot                       |
| Figure 3                       | `extract_query_mse()`, `extract_query_cost_mse_threshold()` | MSE vs cost/queries              |
| Figure 4a–4b | `toy_example.py`, `real_world_data.py` | Saved as `cumulative_cost_toy.pdf`, `cumulative_cost_with_budget_protein.pdf` |
| Table 1 / Table 2              | End of `toy_example.py` and `real_world_data.py`  | MSE, cost, efficiency summary printout     |


## Notes
- All identifying information has been removed for double-blind review.

- This code is intended for evaluation purposes during peer review.
