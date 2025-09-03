# 05_validation_service_time_robustness.py
# Validate robustness to unloading/service-time assumptions:
# - Rank stability of S_DOBI across service_minutes_assumed
# - Summary means (E, C, S, worked_hours) per service and deltas vs BAU

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kpis", required=True, help="KPI CSV containing multiple service_minutes_assumed scenarios")
    ap.add_argument("--outdir", default="./outputs", help="Directory to write outputs")
    ap.add_argument("--bau_service", type=int, default=10, help="BAU service minutes (for delta columns)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = pd.read_csv(args.kpis)

    # Basic checks
    needed = ["date","driver_id","service_minutes_assumed","S_DOBI","co2_g_per_km","cost_per_km_vnd","worked_hours"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns in KPIs: {miss}")

    # Normalize types
    # (date as string is fine as long as consistent; if needed, parse)
    df["driver_id"] = df["driver_id"].astype(str)
    df["service_minutes_assumed"] = pd.to_numeric(df["service_minutes_assumed"], errors="coerce")

    services = sorted(df["service_minutes_assumed"].dropna().unique().tolist())
    if len(services) < 2:
        raise ValueError("Need >= 2 distinct 'service_minutes_assumed' values in the KPI file for a robustness check.")

    key = ["date","driver_id"]

    # ---- Build a wide table of S_DOBI ranks per service ----
    wide = None
    for s in services:
        sub = df[df["service_minutes_assumed"] == s].copy()
        # Keep only key + S_DOBI, and rename to unique names
        sub = sub[key + ["S_DOBI"]].copy()
        sub.rename(columns={
            "S_DOBI": f"S_DOBI_{s}"
        }, inplace=True)
        # Rank across common key universe (ascending => lower burden ranks lower)
        sub[f"rank_S_DOBI_{s}"] = sub[f"S_DOBI_{s}"].rank(method="average", ascending=True)
        # Merge by key
        if wide is None:
            wide = sub
        else:
            # columns are already uniquely named; no suffixes needed, avoids MergeError
            wide = wide.merge(sub, on=key, how="inner")

    # Save the rank matrix (useful for debugging/reporting)
    wide.to_csv(outdir / "service_time_rank_matrix.csv", index=False)

    # ---- Pairwise Kendall tau between service-time ranks ----
    rows = []
    for i, si in enumerate(services):
        for sj in services[i+1:]:
            xi = wide[f"rank_S_DOBI_{si}"]
            xj = wide[f"rank_S_DOBI_{sj}"]
            tau, p = kendalltau(xi, xj, nan_policy="omit")
            rows.append({
                "service_i": si,
                "service_j": sj,
                "kendall_tau": tau,
                "p_value": p,
                "N_pairs": int(np.sum(~(xi.isna() | xj.isna())))
            })
    corr = pd.DataFrame(rows)
    corr.to_csv(outdir / "service_time_rank_corr.csv", index=False)

    # ---- Per-service summaries and deltas vs BAU ----
    summ = (
        df.groupby("service_minutes_assumed", as_index=False)
          .agg(
              N_driverdays=("S_DOBI","size"),
              E_mean=("co2_g_per_km","mean"),
              C_mean=("cost_per_km_vnd","mean"),
              S_mean=("S_DOBI","mean"),
              worked_hours_mean=("worked_hours","mean"),
              hours_gt_12_pct=("worked_hours", lambda s: 100.0 * (s > 12.0).mean()),
              hours_gt_12_5_pct=("worked_hours", lambda s: 100.0 * (s > 12.5).mean()),
              hours_gt_13_pct=("worked_hours", lambda s: 100.0 * (s > 13.0).mean()),
          )
          .rename(columns={"service_minutes_assumed":"service_minutes"})
          .sort_values("service_minutes")
    )

    if (summ["service_minutes"] == args.bau_service).any():
        bau = summ.loc[summ["service_minutes"] == args.bau_service].iloc[0]
        for c in ["E_mean","C_mean","S_mean","worked_hours_mean"]:
            summ[f"{c}_delta_vs{args.bau_service}"] = summ[c] - float(bau[c])
    else:
        # If BAU not present, still write summary without deltas
        pass

    summ.to_csv(outdir / "service_time_summary.csv", index=False)

    print("✓ Wrote:")
    print("  -", outdir / "service_time_rank_matrix.csv")
    print("  -", outdir / "service_time_rank_corr.csv")
    print("  -", outdir / "service_time_summary.csv")
    print("Note: high Kendall’s τ (e.g., >0.8) indicates rank stability of S_DOBI across service-time assumptions.")

if __name__ == "__main__":
    main()
