# 04c_extract_pareto_3d.py  robust 3-pillar frontier (E, Ec, S) with S fallback
import argparse, os, pandas as pd, numpy as np

S_SYNONYMS = [
    "S_dobi","dobi","dobi_score","DOBI","S",
    "worked_hours","duty_hours","drive_hours",
    "DriveTime_h","drive_time_h","total_hours"
]

def first_present(df, names):
    for n in names:
        if n in df.columns: return n
    return None

def dominate(a, b, cols):
    """Return True if row a dominates row b (all objectives are MINIMIZED)."""
    better_or_equal = all(a[c] <= b[c] + 1e-12 for c in cols)
    strictly_better = any(a[c] <  b[c] - 1e-12 for c in cols)
    return better_or_equal and strictly_better

def per_day_frontier(df, obj_cols):
    flags = []
    # Assume df is one (driver_id, date)
    X = df[obj_cols].to_numpy()
    n = len(df)
    is_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_front[i]: 
            continue
        for j in range(n):
            if i==j: 
                continue
            # j dominates i?
            if np.all(X[j] <= X[i] + 1e-12) and np.any(X[j] < X[i] - 1e-12):
                is_front[i] = False
                break
    flags = is_front
    return pd.Series(flags, index=df.index)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counterfactual_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--s_col", default=None,
                    help="Name of Social column. If omitted, will try DOBI names, then hours.")
    ap.add_argument("--s_scale", choices=["none","minmax_day","minmax_global"], default="minmax_day",
                    help="Optional scaling for S to harmonize units across resequencers/shifts.")
    ap.add_argument("--e_col", default="E_g_per_km")
    ap.add_argument("--ec_col", default="Ec_cost_per_km_vnd")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cf = pd.read_csv(args.counterfactual_csv)

    # Key columns
    need = ["driver_id","date", args.e_col, args.ec_col]
    for k in need:
        if k not in cf.columns:
            raise SystemExit(f"Missing required column '{k}' in {args.counterfactual_csv}")

    # Social column detection/derivation
    s_col = args.s_col or first_present(cf, S_SYNONYMS)
    if s_col is None:
        raise SystemExit("No Social column found. Provide --s_col or add one of: " + ", ".join(S_SYNONYMS))
    s_raw = pd.to_numeric(cf[s_col], errors="coerce")
    if not np.isfinite(s_raw).any():
        raise SystemExit(f"S column '{s_col}' has no numeric values.")

    cf["_S_raw"] = s_raw

    # Optional scaling
    if args.s_scale == "none":
        cf["_S_used"] = cf["_S_raw"]
    elif args.s_scale == "minmax_day":
        cf["_S_used"] = cf.groupby(["driver_id","date"])["_S_raw"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)
        )
    else:  # minmax_global
        smin, smax = np.nanmin(cf["_S_raw"].values), np.nanmax(cf["_S_raw"].values)
        cf["_S_used"] = (cf["_S_raw"] - smin) / (smax - smin + 1e-12)

    # Build per-day frontier
    obj_cols = [args.e_col, args.ec_col, "_S_used"]  # all minimized
    cf["on_frontier"] = False
    for (drv, dt), sub in cf.groupby(["driver_id","date"], sort=False):
        flags = per_day_frontier(sub, obj_cols)
        cf.loc[flags.index, "on_frontier"] = flags

    perday_path = os.path.join(args.out_dir, "per_day_with_frontier_flag.csv")
    cf.to_csv(perday_path, index=False)
    print(f"Wrote {perday_path} with frontier flags.")

    pooled = cf[cf["on_frontier"]].copy()
    pooled_path = os.path.join(args.out_dir, "pooled_frontier.csv")
    pooled.to_csv(pooled_path, index=False)
    print(f"Wrote {pooled_path} with {len(pooled)} efficient points.")

if __name__ == "__main__":
    main()
