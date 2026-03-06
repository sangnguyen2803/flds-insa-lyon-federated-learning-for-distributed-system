from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "FedAvg": ROOT / "runs" / "ehealth_hfl_heart_iid_fedavg",
    "FedProx": ROOT / "runs" / "ehealth_hfl_heart_iid_fedprox",
    "DPFedAVG": ROOT / "runs" / "ehealth_hfl_heart_iid_dpfedavg",
}
OUT_DIR = ROOT / "runs" / "ehealth_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def summarize_one(name: str, run_dir: Path) -> dict:
    global_df = pd.read_csv(run_dir / "global_metrics.csv")
    locals_df = pd.read_csv(run_dir / "locals_metrics.csv")
    comm_df = pd.read_csv(run_dir / "comm_costs.csv")
    run_df = pd.read_csv(run_dir / "run_metrics.csv")

    last_global = global_df.iloc[-1]
    final_round = int(last_global["round"])
    final_locals = locals_df[locals_df["round"] == final_round]
    acc_values = final_locals["accuracy"]
    runtime = float(run_df.loc[run_df["metric"] == "run_time_seconds", "value"].iloc[0])
    total_comm = float(comm_df["comm_costs"].sum())

    return {
        "method": name,
        "final_round": final_round,
        "global_accuracy": float(last_global["accuracy"]),
        "global_macro_f1": float(last_global["macro_f1"]),
        "fairness_client_acc_std": float(acc_values.std()),
        "fairness_client_acc_gap": float(acc_values.max() - acc_values.min()),
        "fairness_client_min_acc": float(acc_values.min()),
        "runtime_seconds": runtime,
        "total_comm_cost": total_comm,
        "cost_per_acc_comm": float(total_comm / max(last_global["accuracy"], 1e-8)),
        "cost_per_acc_time": float(runtime / max(last_global["accuracy"], 1e-8)),
    }


def main() -> None:
    out_df = pd.DataFrame([summarize_one(name, p) for name, p in RUNS.items()])
    out_df = out_df.sort_values("global_accuracy", ascending=False)
    out_csv = OUT_DIR / "hfl_comparison_iid.csv"
    out_md = OUT_DIR / "hfl_comparison_iid.md"
    out_df.to_csv(out_csv, index=False)

    headers = list(out_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in out_df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
