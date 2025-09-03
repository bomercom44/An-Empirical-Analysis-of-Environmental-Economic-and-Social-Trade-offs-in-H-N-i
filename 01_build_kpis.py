# 01_build_kpis.py
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ------------------------- utilities -------------------------
def read_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def coalesce_columns(df: pd.DataFrame, name: str, candidates, required=True):
    """Pick the first column that exists in df from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Required column for '{name}' not found. Tried: {candidates}")
    return None


def parse_hhmmss_to_seconds(val):
    """Accept seconds as number OR 'HH:MM:SS' string; return seconds (float) or NaN."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).strip()
    parts = s.split(":")
    if len(parts) == 3:
        try:
            h, m, sec = map(float, parts)
            return h * 3600 + m * 60 + sec
        except Exception:
            return np.nan
    # try plain numeric string
    try:
        return float(s)
    except Exception:
        return np.nan


def normalize_vehicle_name(v: str) -> str:
    """Lowercase, remove '(...)' suffixes, collapse whitespace."""
    if v is None:
        return ""
    v = str(v).strip().lower()
    v = re.sub(r"\s*\(.*?\)\s*", " ", v)  # drop parenthetical qualifiers like "(Euro 2)"
    v = re.sub(r"\s+", " ", v)
    return v.strip()


def fuel_type_from_vehicle(vehicle_name: str) -> str:
    """
    Heuristic: Suzuki Super Carry Pro → petrol; others → diesel.
    Adjust here if your fleet differs.
    """
    nv = normalize_vehicle_name(vehicle_name)
    if "suzuki super carry" in nv:
        return "petrol"
    return "diesel"


def minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


# ------------------------- main -------------------------
def main(cfg_path: str):
    cfg = read_cfg(cfg_path)

    # ---- paths ----
    data_in = cfg.get("data_in", "./schedule_with_arrival_times.csv")
    outputs_dir = cfg.get("outputs_dir", "./outputs")
    legs_out = cfg.get("legs_csv", os.path.join(outputs_dir, "legs_enriched.csv"))
    kpis_out = cfg.get("kpis_csv", os.path.join(outputs_dir, "kpis_driverday.csv"))
    logs_dir = cfg.get("logs_dir", os.path.join(outputs_dir, "logs"))
    ensure_dir(outputs_dir)
    ensure_dir(logs_dir)

    # ---- SOP / timing ----
    depot_load_min = int(cfg.get("depot_load_minutes", 30))
    paid_break_min = int(cfg.get("paid_break_minutes", 60))
    service_minutes_list = cfg.get("service_minutes_list", [10])  # BAU must be in here
    late_cutoff_hhmm = cfg.get("late_cutoff_hhmm", "18:30")  # not used in this step, kept for later

    # ---- EF / costing ----
    co2_per_litre = cfg.get("co2_per_litre", {"diesel": 2.68, "petrol": 2.31})
    l_per_100km_map = cfg.get("l_per_100km", {})
    # Build a normalized lookup so name variants map correctly
    norm_l100_map = {normalize_vehicle_name(k): float(v) for k, v in l_per_100km_map.items()}

    costing_mode = cfg.get("costing_mode", "monthly_totals")
    monthly_totals = cfg.get("monthly_totals_vnd", {})
    fuel_total_vnd = monthly_totals.get("fuel_total", None)
    maint_total_vnd = monthly_totals.get("maintenance_total", None)

    monthly_costs = cfg.get("monthly_costs", {})
    working_days_per_month = int(cfg.get("working_days_per_month", 26))
    # Avoid double-counting maintenance if also provided as a monthly pool
    monthly_costs_sans_maint = {k: v for k, v in monthly_costs.items() if k.lower() != "maintenance"}

    # ---- DOBI / thresholds ----
    thresholds = cfg.get("low_speed_thresholds", [15])
    t0 = float(thresholds[0])
    wts = cfg.get("dobi_weights", [0.4, 0.3, 0.3])
    w1, w2, w3 = [float(x) for x in wts]

    # ---- geocoding quality ----
    geocode_quality_field = cfg.get("geocode_quality_field", "geometry_type")

    # ------------------ read raw ------------------
    df_raw = pd.read_csv(data_in)

    # Trim only string columns (no FutureWarning)
    for col in df_raw.select_dtypes(include="object").columns:
        df_raw[col] = df_raw[col].str.strip()

    # Discover columns
    date_col = coalesce_columns(df_raw, "date", ["date", "service_date", "day"], required=True)
    driver_col = coalesce_columns(df_raw, "driver_id", ["driver_id", "driver", "driver_code"], required=True)
    vehicle_col = coalesce_columns(df_raw, "vehicle", ["vehicle", "vehicle_name", "model"], required=True)
    dist_col = coalesce_columns(df_raw, "distance_km", ["api_distance_km", "distance_km", "dist_km", "distance"], True)
    t_col = coalesce_columns(df_raw, "travel_time", ["travel_time", "drive_seconds", "duration"], True)
    to_label_col = coalesce_columns(df_raw, "to_label", ["to_label", "to_name", "dest_label", "destination"], False)
    from_label_col = coalesce_columns(df_raw, "from_label", ["from_label", "from_name", "origin_label", "origin"], False)
    arrival_col = coalesce_columns(df_raw, "arrival_time", ["arrival_hhmm", "arrival_time", "arrived_at"], False)
    geoq_col = geocode_quality_field if geocode_quality_field in df_raw.columns else None

    df = df_raw.copy()

    # ---- date parsing (respect config) ----
    date_fmt = cfg.get("date_format", None)  # e.g. "%d/%m/%Y"
    dayfirst = bool(cfg.get("dayfirst", True))
    dates = df[date_col].astype(str).str.strip()

    if date_fmt and str(date_fmt).lower() == "mixed":
        parsed = pd.to_datetime(dates, dayfirst=dayfirst, errors="coerce")
    elif date_fmt:
        parsed = pd.to_datetime(dates, format=date_fmt, dayfirst=dayfirst, errors="coerce")
    else:
        parsed = pd.to_datetime(dates, dayfirst=dayfirst, errors="coerce")

    bad = parsed.isna().sum()
    if bad:
        examples = dates[parsed.isna()].head(5).tolist()
        raise ValueError(f"Could not parse {bad} date rows. First examples: {examples}")

    df[date_col] = parsed.dt.date
    df[driver_col] = df[driver_col].astype(str)
    df[vehicle_col] = df[vehicle_col].astype(str)

    # ---- distance/time ----
    df["distance_km"] = pd.to_numeric(df[dist_col], errors="coerce")
    if "drive_seconds" in df.columns:
        df["drive_seconds"] = pd.to_numeric(df["drive_seconds"], errors="coerce")
    else:
        df["drive_seconds"] = df[t_col].apply(parse_hhmmss_to_seconds)

    # Drop impossible legs
    df = df[(df["distance_km"] > 0) & (df["drive_seconds"] > 0)].copy()

    # Speeds
    df["drive_hours"] = df["drive_seconds"] / 3600.0
    df["speed_kmh"] = df["distance_km"] / df["drive_hours"]

    # Geocoding quality (if present)
    df["geocode_quality"] = df[geoq_col].astype(str) if geoq_col else np.nan

    # ---- litres and CO2 per leg ----
    def leg_litres(row):
        nv = normalize_vehicle_name(row[vehicle_col])
        if nv not in norm_l100_map:
            raise KeyError(
                f"Vehicle '{row[vehicle_col]}' missing in config:l_per_100km (normalized='{nv}')."
            )
        l100 = norm_l100_map[nv]
        return (float(l100) / 100.0) * float(row["distance_km"])

    df["litres_leg"] = df.apply(leg_litres, axis=1)
    df["fuel_type"] = df[vehicle_col].apply(fuel_type_from_vehicle)
    df["co2_kg_leg"] = df.apply(
        lambda r: r["litres_leg"] * co2_per_litre.get(r["fuel_type"], 2.68), axis=1
    )

    # ---- group to driver-day base ----
    group_keys = [date_col, driver_col]

    if to_label_col:
        to_lbl = df[to_label_col].astype(str).str.lower()
        # A "stop" is anything not depot-like
        df["is_stop_leg"] = ~to_lbl.isin(["company", "car-park", "car park", "depot"])
    else:
        # If labels absent, treat all legs as potential stops (conservative)
        df["is_stop_leg"] = True

    base = (
        df.groupby(group_keys, as_index=False)
          .agg(
              distance_km=("distance_km", "sum"),
              drive_seconds=("drive_seconds", "sum"),
              litres=("litres_leg", "sum"),
              co2_kg=("co2_kg_leg", "sum"),
              n_legs=("distance_km", "size"),
              n_stop_legs=("is_stop_leg", "sum")
          )
    )
    base["drive_hours"] = base["drive_seconds"] / 3600.0
    base["stops"] = base["n_stop_legs"].astype(int)

    # low-speed shares (leg-weighted and time-weighted) at threshold t0
    low_leg = (
        df.assign(low=(df["speed_kmh"] < t0).astype(int))
          .groupby(group_keys)["low"].mean()
          .reset_index()
          .rename(columns={"low": f"low_speed_share_leg_t{int(t0)}"})
    )
    base = base.merge(low_leg, on=group_keys, how="left")

    low_time = (
        df.assign(low=(df["speed_kmh"] < t0).astype(int), w=df["drive_seconds"])
          .groupby(group_keys)
          .apply(lambda g: np.average(g["low"], weights=g["w"]))
          .reset_index(name=f"low_speed_share_time_t{int(t0)}")
    )
    base = base.merge(low_time, on=group_keys, how="left")

    # ---- KPI scenarios over service time assumptions ----
    cong_col = f"low_speed_share_time_t{int(t0)}"
    all_kpis = []

    for svc_min in service_minutes_list:
        tmp = base.copy()
        tmp["service_minutes"] = float(svc_min)
        tmp["worked_hours"] = (
            tmp["drive_hours"]
            + depot_load_min / 60.0
            + (float(svc_min) / 60.0) * tmp["stops"]
            + paid_break_min / 60.0
        )

        # DOBI components
        cong = tmp[cong_col].astype(float)
        dens = tmp["stops"] / tmp["worked_hours"].replace({0: np.nan})
        tmp["worked_hours_median_driver"] = tmp.groupby(driver_col)["worked_hours"].transform("median")
        varcomp = (tmp["worked_hours"] - tmp["worked_hours_median_driver"]).abs()

        tmp["_T_cong"] = minmax(cong)
        tmp["_D_stops"] = minmax(dens)
        tmp["_V_shift"] = minmax(varcomp)
        tmp["S_DOBI"] = w1 * tmp["_T_cong"] + w2 * tmp["_D_stops"] + w3 * tmp["_V_shift"]

        tmp["service_minutes_assumed"] = float(svc_min)
        all_kpis.append(tmp)

    kpis = pd.concat(all_kpis, ignore_index=True)

    # ---- costing ----
    if costing_mode.lower() == "monthly_totals":
        if fuel_total_vnd is None or maint_total_vnd is None:
            raise ValueError(
                "costing_mode='monthly_totals' requires monthly_totals_vnd.fuel_total and .maintenance_total in config.yml"
            )

        litres_per_day = base[group_keys + ["litres"]].copy()
        dist_per_day = base[group_keys + ["distance_km"]].copy()
        total_litres = litres_per_day["litres"].sum()
        total_distance = dist_per_day["distance_km"].sum()

        if total_litres <= 0:
            raise ValueError("Total litres is zero; check L/100km map and distances.")
        if total_distance <= 0:
            raise ValueError("Total distance is zero; check distance data.")

        kpis = kpis.merge(litres_per_day, on=group_keys, how="left", suffixes=("", "_lit"))
        kpis = kpis.merge(dist_per_day, on=group_keys, how="left", suffixes=("", "_dist"))

        kpis["fuel_cost_vnd"] = (kpis["litres"] / total_litres) * float(fuel_total_vnd)
        kpis["maint_cost_vnd"] = (kpis["distance_km"] / total_distance) * float(maint_total_vnd)
    else:
        # Unit-price fallback (not your current mode, but supported)
        prices = cfg.get("fuel_price_vnd", {"diesel": 21000, "ron95": 23500})
        legs_daily = (
            df.groupby(group_keys + ["fuel_type"], as_index=False)
              .agg(lit=("litres_leg", "sum"))
        )
        piv = legs_daily.pivot(index=group_keys, columns="fuel_type", values="lit").fillna(0.0).reset_index()
        for col in ["diesel", "petrol"]:
            if col not in piv.columns:
                piv[col] = 0.0
        piv["fuel_cost_vnd"] = piv["diesel"] * prices.get("diesel", 21000) + piv["petrol"] * prices.get("ron95", 23500)

        dist_per_day = base[group_keys + ["distance_km"]].copy()
        kpis = kpis.merge(piv[group_keys + ["fuel_cost_vnd"]], on=group_keys, how="left")
        kpis = kpis.merge(dist_per_day, on=group_keys, how="left", suffixes=("", "_dist"))
        kpis["maint_cost_vnd"] = 0.0

    # monthly “other” (labour + other overheads) as per-day flat amount
    monthly_other_total = float(sum(monthly_costs_sans_maint.values()))
    per_day_other = monthly_other_total / float(working_days_per_month)
    kpis["labour_other_cost_vnd"] = per_day_other

    # totals and intensities
    kpis["total_cost_vnd"] = kpis["fuel_cost_vnd"] + kpis["maint_cost_vnd"] + kpis["labour_other_cost_vnd"]
    kpis["cost_per_km_vnd"] = kpis["total_cost_vnd"] / kpis["distance_km"].replace({0: np.nan})
    kpis["cost_per_stop_vnd"] = kpis["total_cost_vnd"] / kpis["stops"].replace({0: np.nan})
    kpis["co2_g_per_km"] = (kpis["co2_kg"] * 1000.0) / kpis["distance_km"].replace({0: np.nan})
    kpis["fuel_L_per_100km"] = (kpis["litres"] / kpis["distance_km"].replace({0: np.nan})) * 100.0

    # ---- save outputs ----
    df.to_csv(legs_out, index=False)
    kpis.to_csv(kpis_out, index=False)

    # ---- reconciliation for BAU (service_minutes_assumed == 10) ----
    if costing_mode.lower() == "monthly_totals":
        bau = kpis[kpis["service_minutes_assumed"] == 10]
        print("== Reconciliation (BAU only) ==")
        print(f"Fuel cost sum (target ≈ {fuel_total_vnd:,}): {bau['fuel_cost_vnd'].sum():,.0f}")
        print(f"Maint cost sum (target ≈ {maint_total_vnd:,}): {bau['maint_cost_vnd'].sum():,.0f}")
        print(f"Other per driver-day applied: {per_day_other:,.0f} VND")
        print(f"BAU driver-days counted: {len(bau):d}")

    # optional: warn if normalized vehicle names in data weren’t mapped
    unseen = sorted(set(normalize_vehicle_name(v) for v in df[vehicle_col].unique()) - set(norm_l100_map.keys()))
    if unseen:
        print("WARNING: vehicles not found in l_per_100km (normalized). Add to config if needed:")
        for u in unseen:
            print("  -", u)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yml")
    args = ap.parse_args()
    main(args.config)
