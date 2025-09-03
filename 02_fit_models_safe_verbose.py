# 02_fit_models_safe_verbose.py
# Robust, FE-based models across multiple KPI files (service-time / threshold variants)
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def find_congestion_col(df: pd.DataFrame) -> str:
    """
    Find the first time-weighted low-speed share column in the dataframe.
    Expected names like: low_speed_share_time_t12 / t15 / t18
    """
    cands = [c for c in df.columns if c.startswith("low_speed_share_time_t")]
    if not cands:
        raise KeyError("No column like 'low_speed_share_time_tXX' found in KPI file.")
    # Prefer t15 if present; else take the first
    for prefer in ["low_speed_share_time_t15", "low_speed_share_time_t12", "low_speed_share_time_t18"]:
        if prefer in cands:
            return prefer
    return cands[0]


def add_derived_columns(df: pd.DataFrame, cong_col: str) -> pd.DataFrame:
    out = df.copy()

    # parse date to weekday (Mon=0..Sun=6)
    # robust parse; treat already-parsed dates too
    out["date_parsed"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    if out["date_parsed"].isna().all():
        # try ISO parse if needed
        out["date_parsed"] = pd.to_datetime(out["date"], errors="coerce")
    out["weekday"] = out["date_parsed"].dt.weekday

    # basic guards
    if "distance_km" not in out.columns:
        raise KeyError("distance_km not found in KPI file.")
    if "stops" not in out.columns:
        raise KeyError("stops not found in KPI file.")
    if "worked_hours" not in out.columns:
        raise KeyError("worked_hours not found in KPI file.")
    if "co2_g_per_km" not in out.columns:
        raise KeyError("co2_g_per_km not found in KPI file.")
    if "driver_id" not in out.columns:
        raise KeyError("driver_id not found in KPI file.")

    # intensity controls
    out["stops_per_km"] = out["stops"] / out["distance_km"].replace(0, np.nan)
    out["stops_per_worked_hour"] = out["stops"] / out["worked_hours"].replace(0, np.nan)

    # give a standard alias for congestion
    out["low_speed_share"] = out[cong_col].astype(float)

    # drop rows with critical NA (keep FE even if NaN in driver_id handled upstream)
    out = out.dropna(subset=[
        "co2_g_per_km", "worked_hours", "distance_km", "stops",
        "stops_per_km", "stops_per_worked_hour", "low_speed_share", "weekday", "driver_id"
    ])
    # cast FE keys to string (patsy-friendly)
    out["driver_id"] = out["driver_id"].astype(str)
    out["weekday"] = out["weekday"].astype(int)
    return out


def fit_ols_hc3(formula: str, data: pd.DataFrame):
    """
    Fit OLS with HC3 robust SEs and return the fitted model + tidy table.
    """
    model = smf.ols(formula, data=data).fit(cov_type="HC3")
    params = model.params
    bse = model.bse
    tvals = model.tvalues
    pvals = model.pvalues
    ci = model.conf_int()
    ci.columns = ["ci_low", "ci_high"]

    tidy = pd.concat(
        [
            params.rename("coef"),
            bse.rename("se"),
            tvals.rename("t"),
            pvals.rename("p"),
            ci,
        ],
        axis=1,
    ).reset_index().rename(columns={"index": "term"})
    return model, tidy


def safe_vif(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Compute VIF only on continuous predictors to avoid FE dummy explosion.
    """
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.DataFrame(columns=["variable", "VIF"])

    X = df[available].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    if X.shape[1] < 2 or X.shape[0] < 5:
        # VIF needs at least 2 columns and a few rows
        return pd.DataFrame(columns=["variable", "VIF"])

    # add intercept
    X = sm.add_constant(X, has_constant="add")
    vif_rows = []
    # compute VIF for non-constant cols
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        try:
            v = variance_inflation_factor(X.values, i)
            vif_rows.append({"variable": col, "VIF": float(v)})
        except Exception:
            vif_rows.append({"variable": col, "VIF": np.nan})
    return pd.DataFrame(vif_rows)


def label_from_filename(path: str) -> str:
    """
    Derive a short label from filename, e.g., kpis_driverday_15_drive.csv -> svc15
    """
    base = os.path.basename(path)
    if "_12_" in base or "12_drive" in base:
        return "svc12"
    if "_15_" in base or "15_drive" in base:
        return "svc15"
    if "_18_" in base or "18_drive" in base:
        return "svc18"
    # fallback
    return "svcXX"


def run_for_file(kpi_path: str, outdir: str, verbose: bool = False):
    label = label_from_filename(kpi_path)
    if verbose:
        print(f"\n=== Processing {kpi_path} (label: {label}) ===")

    df = pd.read_csv(kpi_path)

    # pick congestion column for this file
    cong_col = find_congestion_col(df)
    if verbose:
        print(f"Using congestion column: {cong_col}")

    df = add_derived_columns(df, cong_col)

    # ---------- H1: Emissions intensity ----------
    # co2_g_per_km ~ low_speed_share + stops_per_km + FE(driver_id, weekday)
    f_h1 = "co2_g_per_km ~ low_speed_share + stops_per_km + C(driver_id) + C(weekday)"
    m1, t1 = fit_ols_hc3(f_h1, df)
    vif1 = safe_vif(df, ["low_speed_share", "stops_per_km"])

    # ---------- H2: Workload–stops ----------
    # worked_hours ~ stops + distance_km + low_speed_share + FE
    f_h2 = "worked_hours ~ stops + distance_km + low_speed_share + C(driver_id) + C(weekday)"
    m2, t2 = fit_ols_hc3(f_h2, df)
    vif2 = safe_vif(df, ["stops", "distance_km", "low_speed_share"])

    # ---------- H4: Congestion burden ----------
    # worked_hours ~ low_speed_share + distance_km + stops + FE (order changed to foreground congestion)
    f_h4 = "worked_hours ~ low_speed_share + distance_km + stops + C(driver_id) + C(weekday)"
    m4, t4 = fit_ols_hc3(f_h4, df)
    vif4 = safe_vif(df, ["low_speed_share", "distance_km", "stops"])

    # write per-file outputs
    per_dir = os.path.join(outdir, f"models_{label}")
    ensure_dir(per_dir)

    # save tidy tables
    t1.assign(model="H1_Emissions_intensity", service=label).to_csv(os.path.join(per_dir, "model_H1_tidy.csv"), index=False)
    t2.assign(model="H2_Workload_stops", service=label).to_csv(os.path.join(per_dir, "model_H2_tidy.csv"), index=False)
    t4.assign(model="H4_Congestion_burden", service=label).to_csv(os.path.join(per_dir, "model_H4_tidy.csv"), index=False)

    # save text summaries (including condition number)
    with open(os.path.join(per_dir, "model_H1_summary.txt"), "w", encoding="utf-8") as f:
        f.write(m1.summary().as_text())
        f.write("\n\n[Note] Condition number: %.2f\n" % m1.condition_number)
    with open(os.path.join(per_dir, "model_H2_summary.txt"), "w", encoding="utf-8") as f:
        f.write(m2.summary().as_text())
        f.write("\n\n[Note] Condition number: %.2f\n" % m2.condition_number)
    with open(os.path.join(per_dir, "model_H4_summary.txt"), "w", encoding="utf-8") as f:
        f.write(m4.summary().as_text())
        f.write("\n\n[Note] Condition number: %.2f\n" % m4.condition_number)

    # save VIFs
    vif1.assign(model="H1", service=label).to_csv(os.path.join(per_dir, "vif_H1.csv"), index=False)
    vif2.assign(model="H2", service=label).to_csv(os.path.join(per_dir, "vif_H2.csv"), index=False)
    vif4.assign(model="H4", service=label).to_csv(os.path.join(per_dir, "vif_H4.csv"), index=False)

    # one combined tidy per file
    tidy_all = pd.concat([t1.assign(model="H1", service=label),
                          t2.assign(model="H2", service=label),
                          t4.assign(model="H4", service=label)],
                         ignore_index=True)
    tidy_all.to_csv(os.path.join(per_dir, "model_summaries_tidy.csv"), index=False)

    # quick effect-view CSV (subset of key terms)
    def pick_terms(tt, keys):
        return tt[tt["term"].isin(keys)].copy()

    key_terms = ["low_speed_share", "stops_per_km", "stops", "distance_km", "Intercept"]
    eff = pd.concat([
        pick_terms(t1.assign(model="H1", service=label), key_terms),
        pick_terms(t2.assign(model="H2", service=label), key_terms),
        pick_terms(t4.assign(model="H4", service=label), key_terms)
    ], ignore_index=True)
    eff.to_csv(os.path.join(per_dir, "model_effects_key_terms.csv"), index=False)

    # Return for combined files
    return tidy_all, eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kpis_glob", required=True, help="Glob for KPI CSVs, e.g. .\\kpis_driverday_*_drive.csv")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    files = sorted(glob.glob(args.kpis_glob))
    if not files:
        print("!! No files matched. Check your path/glob.")
        return

    if args.verbose:
        print("Outdir:", os.path.abspath(args.outdir))
        print("Files to process:")
        for f in files:
            print(" -", f)

    combined_tidy = []
    combined_eff = []

    for f in files:
        try:
            tidy, eff = run_for_file(f, args.outdir, verbose=args.verbose)
            combined_tidy.append(tidy)
            combined_eff.append(eff)
        except Exception as e:
            print(f"!! Failed on {f}: {e}")

    if combined_tidy:
        all_tidy = pd.concat(combined_tidy, ignore_index=True)
        all_tidy.to_csv(os.path.join(args.outdir, "model_summaries_tidy_by_service.csv"), index=False)
    if combined_eff:
        all_eff = pd.concat(combined_eff, ignore_index=True)
        all_eff.to_csv(os.path.join(args.outdir, "model_effects_key_terms_by_service.csv"), index=False)

    print("✓ Done. Combined outputs written to outdir.")


if __name__ == "__main__":
    main()
