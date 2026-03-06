# FL-based E-Health System with Fluke

Tai lieu nay dap ung day du 6 requirement va da train/test thuc te.

## Requirement 1 - Chon medical dataset

- Dataset: `data/heart_disease_uci/heart_disease_uci.csv`
- Bai toan: du doan benh tim nhi phan
  - Label goc: `num`
  - Mapping: `num == 0` (khong benh), `num > 0` (co benh)
- Da them loader vao Fluke:
  - `heart_disease_uci` trong `fluke_package/fluke/data/datasets.py`
  - Co xu ly missing values + one-hot categorical + train/test split

## Requirement 2 - Thiet ke Horizontal FL (HFL)

Kien truc:
- 1 server, 4 clients
- Moi client giu local data shard
- Server aggregate model moi round

Task 2 da co **ca 2 truong hop IID va Non-IID**:

- Non-IID:
  - split theo Dirichlet (`distribution: dir`, `beta: 0.8`)
  - config: `config/ehealth/exp_heart_noniid_*.yaml`
  - script train: `python scripts/run_ehealth_fluke.py`

- IID:
  - split dong deu (`distribution: iid`)
  - config: `config/ehealth/exp_heart_iid_*.yaml`
  - script train: `python scripts/run_ehealth_fluke_iid.py`

Algorithms cho ca 2 case:
- `FedAvg` (baseline)
- `FedProx` (heterogeneity-aware)
- `DPFedAVG` (privacy-preserving)

## Requirement 3 - Handle heterogeneity

- Data heterogeneity: Non-IID Dirichlet split
- Algorithm heterogeneity-aware: `FedProx` voi `mu: 0.1`

## Requirement 4 - Privacy-preserving FL

HFL privacy:
- Dung `DPFedAVG` (Opacus DP-SGD)
- Config: `noise_mul: 0.4`, `max_grad_norm: 1.0`

VFL privacy:
- Embedding clipping
- Gaussian noise truoc khi gui embedding

## Requirement 5 - Quality, fairness, cost metrics

Scripts:
- Non-IID summary: `python scripts/summarize_hfl_results.py`
- IID summary: `python scripts/summarize_hfl_results_iid.py`

Outputs:
- `runs/ehealth_summary/hfl_comparison.csv` (Non-IID)
- `runs/ehealth_summary/hfl_comparison_iid.csv` (IID)

Metric da do:
- Quality: `global_accuracy`, `global_macro_f1`
- Fairness: `fairness_client_acc_std`, `fairness_client_acc_gap`, `fairness_client_min_acc`
- Cost: `runtime_seconds`, `total_comm_cost`, `cost_per_acc_comm`, `cost_per_acc_time`

Ket qua HFL Non-IID (round cuoi):

| method | global_accuracy | global_macro_f1 |
| --- | --- | --- |
| DPFedAVG | 0.63587 | 0.58021 |
| FedProx | 0.56522 | 0.45747 |
| FedAvg | 0.55978 | 0.42533 |

Ket qua HFL IID (round cuoi):

| method | global_accuracy | global_macro_f1 |
| --- | --- | --- |
| DPFedAVG | 0.69565 | 0.68945 |
| FedProx | 0.66304 | 0.66300 |
| FedAvg | 0.65217 | 0.61667 |

## Requirement 6 - Vertical FL (VFL) design

Fluke khong co VFL native end-to-end nhu HFL, nen VFL duoc cai dat rieng trong project tren cung dataset y te.

Pipeline VFL 2-party:
- Party A giu nua trai features
- Party B giu nua phai features
- Server giu head model de tinh loss va cap nhat
- Parties gui embedding trung gian cho server

Code:
- `scripts/vfl_heart.py`

Output:
- `runs/ehealth_summary/vfl_heart_metrics.csv`

Ket qua VFL:
- `test_accuracy`: `0.79891`
- `test_macro_f1`: `0.79044`
- `fairness_dp_gap`: `0.30470`
- `fairness_eod_gap`: `0.02926`
- `fairness_group_acc_gap`: `0.03394`

## Chay lai toan bo train/test

```bash
pip install -e ./fluke_package

# HFL Non-IID
python scripts/run_ehealth_fluke.py
python scripts/summarize_hfl_results.py

# HFL IID
python scripts/run_ehealth_fluke_iid.py
python scripts/summarize_hfl_results_iid.py

# VFL
python scripts/vfl_heart.py
```
