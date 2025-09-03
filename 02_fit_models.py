import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def coalesce(df, logical_name, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Required column for '{logical_name}' not found. Tried: {candidates}")
    return None

def find_low_speed_share_col(df):
    # prefer time-weighted with explicit threshold suffix (e.g., low_speed_share_time_t15)
    for col in df.columns:
        if re.fullmatch(r"low_speed_share_time_t\d+", col):
            return col
    for col in df.columns:
        if col.startswith("low_speed_share_time"):
            return col
    for col in df.columns:
        if re.fullmatch(r"low_speed_share_leg_t\d+", col):
            return col
    raise KeyError("No low_speed_share_* column found (expected e.g., 'low_speed_share_time_t15').")

def tidy_from_results(results, model_name):
    """Build a tidy frame from results attributes (robust to statsmodels version)."""
    params = results.params
    se = results.bse
    tvals = results.tvalues
    pvals = results.pvalues
    ci = results.conf_int()
    out = pd.DataFrame({
        "model": model_name,
        "term": params.index,
        "coef": params.values,
        "se": se.values,
        "t": tvals.values,
        "p": pvals.values,
        "ci_low": ci[0].values,
        "ci_high": ci[1].values,
    })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kpis", required=True, help="Path to KPI CSV (driver-day level)")
    ap.add_argument("--outdir", default="./outputs", help="Directory to write outputs")
    ap.add_argument("--bau_service", type=int, default=10, help="Service minutes to treat as BAU (e.g., 10)")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.kpis)

    # discover key columns
    date_col   = coalesce(df, "date", ["date"], required=False)
    driver_col = coalesce(df, "driver_id", ["driver_id", "driver", "driver_code"], required=True)
    dist_col   = coalesce(df, "distance_km", ["distance_km"], required=True)
    drive_h    = coalesce(df, "drive_hours", ["drive_hours"], required=True)
    stops_col  = coalesce(df, "stops", ["stops", "n_stops", "n_stop_legs"], required=True)
    worked_col = coalesce(df, "worked_hours", ["worked_hours"], required=True)
    co2_int    = coalesce(df, "co2_g_per_km", ["co2_g_per_km", "co2_per_km_g"], required=True)
    cost_int   = coalesce(df, "cost_per_km_vnd", ["cost_per_km_vnd"], required=True)
    svc_col    = coalesce(df, "service_minutes_assumed", ["service_minutes_assumed", "service_minutes"], required=True)
    lowshare   = find_low_speed_share_col(df)

    # filter to BAU service minutes
    d = df[df[svc_col] == args.bau_service].copy()
    if d.empty:
        raise ValueError(f"No rows found with {svc_col} == {args.bau_service}. Check KPI file & --bau_service.")

    # features
    d["avg_speed_kmh"] = d[dist_col] / d[drive_h].replace({0: np.nan})
    d["stops_per_km"]  = d[stops_col] / d[dist_col].replace({0: np.nan})

    keep = [driver_col, dist_col, drive_h, stops_col, worked_col, co2_int, cost_int,
            "avg_speed_kmh", "stops_per_km", lowshare]
    dd = d[keep].replace([np.inf, -np.inf], np.nan).dropna().copy()

    # Model 1: Emissions intensity (CO2 g/km)
    m1 = smf.ols(
        formula=f"{co2_int} ~ avg_speed_kmh + stops_per_km",
        data=dd
    ).fit(cov_type="cluster", cov_kwds={"groups": dd[driver_col]})

    # Model 2: Worked hours (level)
    m2 = smf.ols(
        formula=f"{worked_col} ~ {dist_col} + {stops_col} + {lowshare}",
        data=dd
    ).fit(cov_type="cluster", cov_kwds={"groups": dd[driver_col]})

    # Model 3: Cost per km (intensity)
    m3 = smf.ols(
        formula=f"{cost_int} ~ {dist_col} + stops_per_km + {lowshare}",
        data=dd
    ).fit(cov_type="cluster", cov_kwds={"groups": dd[driver_col]})

    # tidy outputs (version-proof)
    out1 = tidy_from_results(m1, "Emissions_intensity")
    out2 = tidy_from_results(m2, "Worked_hours")
    out3 = tidy_from_results(m3, "Cost_per_km")

    model_summary = pd.concat([out1, out2, out3], ignore_index=True)
    model_summary.to_csv(Path(args.outdir) / "model_summaries_tidy.csv", index=False)

    # quickview of a few terms (if present)
    def pick(df_sum, model_name, term_name):
        sub = df_sum[(df_sum["model"] == model_name) & (df_sum["term"] == term_name)]
        return None if sub.empty else sub.iloc[0][["model","term","coef","se","t","p","ci_low","ci_high"]]

    wanted = []
    wanted += [pick(model_summary, "Emissions_intensity", "avg_speed_kmh")]
    wanted += [pick(model_summary, "Emissions_intensity", "stops_per_km")]
    wanted += [pick(model_summary, "Worked_hours", dist_col)]
    wanted += [pick(model_summary, "Worked_hours", stops_col)]
    wanted += [pick(model_summary, "Worked_hours", lowshare)]
    wanted += [pick(model_summary, "Cost_per_km", dist_col)]
    quick = pd.DataFrame([w for w in wanted if w is not None])
    quick.to_csv(Path(args.outdir) / "model_quickview.csv", index=False)

    # plain-text summaries (handy to inspect)
    (Path(args.outdir) / "model_emissions_summary.txt").write_text(m1.summary().as_text(), encoding="utf-8")
    (Path(args.outdir) / "model_hours_summary.txt").write_text(m2.summary().as_text(), encoding="utf-8")
    (Path(args.outdir) / "model_cost_summary.txt").write_text(m3.summary().as_text(), encoding="utf-8")

    print("✓ Wrote:", Path(args.outdir) / "model_summaries_tidy.csv")
    print("✓ Wrote:", Path(args.outdir) / "model_quickview.csv")
    print("✓ Wrote: model_*_summary.txt")

if __name__ == "__main__":
    main()
