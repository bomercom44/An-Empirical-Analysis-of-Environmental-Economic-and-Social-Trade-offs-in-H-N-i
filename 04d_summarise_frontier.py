# 04d_summarise_frontier.py  3-pillar summaries (E, Ec, S), materiality, prices, picks
import argparse, os, numpy as np, pandas as pd

def iqr(x):
    x = np.asarray(x, float)
    if x.size == 0: return (np.nan, np.nan)
    q1, q3 = np.nanpercentile(x, [25, 75])
    return q1, q3

def pick_argmin(df, col):
    idx = df[col].astype(float).idxmin()
    return df.loc[idx]

def norm01_series(s):
    s = s.astype(float).to_numpy()
    mn, mx = np.nanmin(s), np.nanmax(s)
    return (s - mn) / (mx - mn + 1e-12)

def inside_minmax_day(perday):
    """Ensure per-day social column '_S_used' exists; build from S_DOBI or worked_hours if needed."""
    if "_S_used" in perday.columns:
        return perday
    # pick a social proxy
    S_cands = [c for c in ["S_DOBI","worked_hours","duty_hours","_S_raw","_S","S","DOBI","dobi","dobi_score"] if c in perday.columns]
    if not S_cands:
        raise SystemExit("No social column found in per_day_with_frontier_flag.csv; expected one of S_DOBI/worked_hours/ or _S_used.")
    scol = S_cands[0]
    perday["_S_used"] = perday.groupby(["driver_id","date"])[scol].transform(
        lambda x: (x.astype(float) - x.astype(float).min()) / (x.astype(float).max() - x.astype(float).min() + 1e-12)
    )
    return perday

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf_csv", required=True, help="sim_out/counterfactual_points_ecfixed.csv (has E_bau_gkm & Ec_bau_vndkm)")
    ap.add_argument("--perday_csv", required=True, help="sim_out/per_day_with_frontier_flag.csv (has on_frontier; may have _S_used)")
    ap.add_argument("--pooled_csv", required=True, help="sim_out/pooled_frontier.csv (subset on_frontier=True)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--materiality_e_pct", type=float, default=2.0)
    ap.add_argument("--materiality_ec_pct", type=float, default=2.0)
    ap.add_argument("--materiality_s_abs", type=float, default=0.02)
    ap.add_argument("--weights", default="0.40,0.35,0.25")  # Balanced J(w)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    wE, wC, wS = [float(x) for x in args.weights.split(",")]

    # Load input frames
    cf = pd.read_csv(args.cf_csv)
    perday = pd.read_csv(args.perday_csv)
    pooled = pd.read_csv(args.pooled_csv)

    # Harmonize social column on per-day
    perday = inside_minmax_day(perday)

    # Required keys present?
    for name, df in [("cf_csv", cf), ("perday_csv", perday), ("pooled_csv", pooled)]:
        if not {"driver_id","date"}.issubset(df.columns):
            raise SystemExit(f"{name} is missing driver_id/date.")

    # Bring _S_used and on_frontier into cf (use the safest join keys present on both)
    join_cols = ["driver_id","date"]
    for k in ["shift_min","service_min","resequencer"]:
        if k in cf.columns and k in perday.columns:
            join_cols.append(k)

    cf2 = cf.merge(
        perday[join_cols + ["_S_used","on_frontier"]].drop_duplicates(),
        on=join_cols, how="left"
    )

    # ---- BAU baselines (de-duplicated) ----
    # BAU mask defined on perday
    pm = perday
    is_bau = (pm.get("shift_min", 0) == 0) & (pm.get("service_min", 10) == 10)
    if "resequencer" in pm.columns:
        is_bau = is_bau & (pm["resequencer"].astype(str).str.upper() == "BAU")
    pm = pm.assign(is_bau=is_bau)

    # group to ONE S_bau per (driver_id,date)
    s_bau = (pm.loc[pm["is_bau"], ["driver_id","date","_S_used"]]
               .groupby(["driver_id","date"], as_index=False)["_S_used"]
               .first()
               .rename(columns={"_S_used":"S_bau"}))

    cf2 = cf2.merge(s_bau, on=["driver_id","date"], how="left")

    # Fallback if day has no explicit BAU: take FIRST observed S
    missS = cf2["S_bau"].isna()
    if missS.any():
        approx = (perday.groupby(["driver_id","date"], as_index=False)["_S_used"]
                        .first()
                        .rename(columns={"_S_used":"S_bau_fallback"}))
        cf2 = cf2.merge(approx, on=["driver_id","date"], how="left")
        cf2["S_bau"] = cf2["S_bau"].fillna(cf2["S_bau_fallback"])
        cf2.drop(columns=["S_bau_fallback"], inplace=True, errors="ignore")

    # Sanity checks for columns well use
    need = ["E_g_per_km","Ec_cost_per_km_vnd","_S_used","E_bau_gkm","Ec_bau_vndkm","S_bau"]
    for c in need:
        if c not in cf2.columns:
            raise SystemExit(f"Missing column after merges: {c}")

    # Deltas vs BAU
    cf2["dE_gkm"]    = cf2["E_g_per_km"]        - cf2["E_bau_gkm"]
    cf2["dEc_vndkm"] = cf2["Ec_cost_per_km_vnd"]- cf2["Ec_bau_vndkm"]
    cf2["dS"]        = cf2["_S_used"]           - cf2["S_bau"]

    # Save full deltas
    full_deltas_path = os.path.join(args.outdir, "frontier_deltas_vs_BAU_all.csv")
    cf2.to_csv(full_deltas_path, index=False)

    # Restrict to the actual frontier: match keys present in pooled
    kcols = ["driver_id","date"]
    for k in ["shift_min","service_min","resequencer"]:
        if k in pooled.columns and k in cf2.columns:
            kcols.append(k)

    pooled_key = pooled[kcols].drop_duplicates()
    fd = cf2.merge(pooled_key, on=kcols, how="inner")

    deltas_path = os.path.join(args.outdir, "frontier_deltas_vs_BAU.csv")
    fd.to_csv(deltas_path, index=False)

    # Materiality (improvements are negative deltas)
    fd["mat_E"]  = (fd["dE_gkm"]     <= -abs(args.materiality_e_pct)/100.0  * fd["E_bau_gkm"])
    fd["mat_Ec"] = (fd["dEc_vndkm"]  <= -abs(args.materiality_ec_pct)/100.0 * fd["Ec_bau_vndkm"])
    fd["mat_S"]  = (fd["dS"]         <= -abs(args.materiality_s_abs))

    triple_any = (fd["dE_gkm"]<0) & (fd["dEc_vndkm"]<0) & (fd["dS"]<0)
    triple_mat = fd["mat_E"] & fd["mat_Ec"] & fd["mat_S"]

    # Price metrics
    mask_green = fd["dE_gkm"] < 0
    pog = fd.loc[mask_green, "dEc_vndkm"] / (-fd.loc[mask_green, "dE_gkm"])
    mask_welf  = fd["dS"] < 0
    pow_ = fd.loc[mask_welf, "dEc_vndkm"] / (-fd.loc[mask_welf, "dS"])

    # Summaries
    def med(x): return float(np.nanmedian(x)) if len(x) else np.nan
    pog_q1, pog_q3 = iqr(pog.dropna())
    pow_q1, pow_q3 = iqr(pow_.dropna())

    summary = {
        "N_frontier_points":            int(len(fd)),
        "triple_win_any_share":         float(triple_any.mean()) if len(fd) else np.nan,
        "triple_win_material_share":    float(triple_mat.mean()) if len(fd) else np.nan,
        "median_dE_gkm":                med(fd["dE_gkm"]),
        "median_dEc_vndkm":             med(fd["dEc_vndkm"]),
        "median_dS":                    med(fd["dS"]),
        "price_of_green_vnd_per_gkm_median":    med(pog),
        "price_of_green_IQR_low":       float(pog_q1), "price_of_green_IQR_high": float(pog_q3),
        "price_of_welfare_vnd_per_Sunit_median": med(pow_),
        "price_of_welfare_IQR_low":     float(pow_q1), "price_of_welfare_IQR_high": float(pow_q3),
        "materiality_e_pct":            args.materiality_e_pct,
        "materiality_ec_pct":           args.materiality_ec_pct,
        "materiality_s_abs":            args.materiality_s_abs
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.outdir, "frontier_summary_3pillar.csv"), index=False)

    # Pairwise projections for plotting
    fd[["driver_id","date","dE_gkm","dS","dEc_vndkm"]].to_csv(
        os.path.join(args.outdir,"proj_dE_vs_dS.csv"), index=False)
    fd[["driver_id","date","dE_gkm","dEc_vndkm"]].to_csv(
        os.path.join(args.outdir,"proj_dE_vs_dEc.csv"), index=False)
    fd[["driver_id","date","dEc_vndkm","dS"]].to_csv(
        os.path.join(args.outdir,"proj_dEc_vs_dS.csv"), index=False)

    # Managerial picks (per day among that day's frontier)
    picks = []
    for (drv, dt), sub in fd.groupby(["driver_id","date"], sort=False):
        e_pick = pick_argmin(sub, "E_g_per_km")
        s_pick = pick_argmin(sub, "_S_used")
        c_pick = pick_argmin(sub, "Ec_cost_per_km_vnd")
        E_n = norm01_series(sub["E_g_per_km"])
        C_n = norm01_series(sub["Ec_cost_per_km_vnd"])
        S_n = norm01_series(sub["_S_used"])
        J = wE*E_n + wC*C_n + wS*S_n
        b_pick = sub.iloc[int(np.argmin(J))]

        def pack(tag, r):
            return {
                "driver_id": drv, "date": dt, "pick": tag,
                "resequencer": r.get("resequencer", "NA"),
                "shift_min": r.get("shift_min", np.nan),
                "service_min": r.get("service_min", np.nan),
                "E_g_per_km": r.get("E_g_per_km", np.nan),
                "Ec_cost_per_km_vnd": r.get("Ec_cost_per_km_vnd", np.nan),
                "_S_used": r.get("_S_used", np.nan),
                "dE_gkm": r.get("dE_gkm", np.nan),
                "dEc_vndkm": r.get("dEc_vndkm", np.nan),
                "dS": r.get("dS", np.nan)
            }

        picks += [pack("E-lean", e_pick), pack("S-lean", s_pick),
                  pack("C-lean", c_pick), pack("Balanced", b_pick)]

    pd.DataFrame(picks).to_csv(os.path.join(args.outdir, "frontier_managerial_picks.csv"), index=False)

    print("Wrote:")
    for fn in [
        "frontier_deltas_vs_BAU_all.csv",
        "frontier_deltas_vs_BAU.csv",
        "frontier_summary_3pillar.csv",
        "proj_dE_vs_dS.csv",
        "proj_dE_vs_dEc.csv",
        "proj_dEc_vs_dS.csv",
        "frontier_managerial_picks.csv",
    ]:
        print(" -", os.path.join(args.outdir, fn))

if __name__ == "__main__":
    main()
