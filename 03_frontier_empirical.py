# 03_frontier_empirical.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def minmax01(x):
    x = pd.Series(x, dtype=float)
    lo, hi = x.min(), x.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=x.index)
    return (x - lo) / (hi - lo)

def is_dominated(i_row, others):
    """Return True if i_row is dominated by any row in 'others' on (E,C,S)."""
    worse_or_equal = (others[["E01","C01","S01"]].values <= i_row[["E01","C01","S01"]].values).all(axis=1)
    strictly_better_some = (others[["E01","C01","S01"]].values <  i_row[["E01","C01","S01"]].values).any(axis=1)
    return bool(np.any(worse_or_equal & strictly_better_some))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kpis", required=True, help="Path to KPI CSV (driver-day level)")
    ap.add_argument("--outdir", default="./outputs", help="Directory for outputs")
    ap.add_argument("--bau_service", type=int, default=10, help="Service minutes to treat as BAU")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.kpis)

    # column names
    need = ["date","driver_id","distance_km","worked_hours","co2_g_per_km","cost_per_km_vnd","S_DOBI","service_minutes_assumed"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Missing column in KPIs: {c}")

    d = df[df["service_minutes_assumed"] == args.bau_service].copy()
    if d.empty:
        raise SystemExit(f"No BAU rows with service_minutes_assumed == {args.bau_service}")

    # Rename to E,C,S and min-max normalise for frontier checks
    d = d.rename(columns={
        "co2_g_per_km": "E",
        "cost_per_km_vnd": "C",
        "S_DOBI": "S"
    })

    d["E01"] = minmax01(d["E"])
    d["C01"] = minmax01(d["C"])
    d["S01"] = minmax01(d["S"])

    # Pareto check (non-dominated = frontier)
    rows = []
    arr = d.reset_index(drop=True)
    for i in range(len(arr)):
        irow = arr.loc[i]
        others = arr.drop(index=i)
        dom = is_dominated(irow, others)
        rows.append(not dom)
    arr["is_frontier"] = rows

    # Save CSVs
    arr.to_csv(Path(args.outdir, "frontier_bau_points.csv"), index=False)
    arr[arr["is_frontier"]].to_csv(Path(args.outdir, "frontier_bau_only.csv"), index=False)

    # Plots
    def scatter2(x, y, xlabel, ylabel, fname):
        plt.figure()
        plt.scatter(arr[~arr["is_frontier"]][x], arr[~arr["is_frontier"]][y], alpha=0.5, s=12, label="Dominated")
        plt.scatter(arr[arr["is_frontier"]][x], arr[arr["is_frontier"]][y], alpha=0.9, s=22, label="Frontier")
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.legend()
        plt.tight_layout()
        plt.savefig(Path(args.outdir, fname), dpi=200)
        plt.close()

    scatter2("E01","C01","E (min–max)","C (min–max)","frontier_bau_E_vs_C.png")
    scatter2("E01","S01","E (min–max)","S (min–max)","frontier_bau_E_vs_S.png")
    scatter2("C01","S01","C (min–max)","S (min–max)","frontier_bau_C_vs_S.png")

    print("✓ Wrote frontier tables and plots to:", args.outdir)

if __name__ == "__main__":
    main()
