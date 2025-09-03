import argparse, glob, os, pandas as pd, numpy as np, sys, re

TIME_CANDIDATES = [
    "start_dt_local","leg_start_local","segment_start_local","departure_time_local",
    "start_datetime_local","start_time_local","start_local","start_dt",
    "start_datetime","start_time","t_start_local","start_iso","start_utc","start_ts","arrival_time"
]
HOUR_PAIR_CANDIDATES = [
    ("date","start_time_local"), ("date","start_time"), ("date_local","start_time_local"),
    ("service_date","start_time_local"), ("service_date","start_time"), ("date","arrival_time")
]
SPEED_CANDIDATES = ["speed_kmh","avg_speed_kmh","mean_speed_kmh"]
DIST_CANDS  = ["distance_km","leg_distance_km","km","dist_km"]
DT_HRS_CANDS= ["drive_time_hours","travel_time_hours","duration_hours","time_hours"]
DT_MIN_CANDS= ["drive_time_min","travel_time_min","duration_min","time_min","mins"]
DT_SEC_CANDS= ["drive_time_s","travel_time_s","duration_s","time_s","secs"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs_glob", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--fallback_speed_kmh", type=float, default=18.0)
    return ap.parse_args()

def get_speed_series(df):
    # explicit speed
    for c in SPEED_CANDIDATES:
        if c in df.columns:
            sp = pd.to_numeric(df[c], errors="coerce")
            if sp.notna().any():
                return sp
    # distance / time
    dist = None; dt_hours = None
    for c in DIST_CANDS:
        if c in df.columns:
            dist = pd.to_numeric(df[c], errors="coerce"); break
    for c in DT_HRS_CANDS:
        if c in df.columns:
            dt_hours = pd.to_numeric(df[c], errors="coerce"); break
    if dist is not None and dt_hours is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            sp = dist / dt_hours.replace({0:np.nan})
        if sp.notna().any():
            return sp
    # minutes
    dt_min = None
    for c in DT_MIN_CANDS:
        if c in df.columns:
            dt_min = pd.to_numeric(df[c], errors="coerce"); break
    if dist is not None and dt_min is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            sp = dist / (dt_min/60.0).replace({0:np.nan})
        if sp.notna().any():
            return sp
    # seconds
    dt_sec = None
    for c in DT_SEC_CANDS:
        if c in df.columns:
            dt_sec = pd.to_numeric(df[c], errors="coerce"); break
    if dist is not None and dt_sec is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            sp = dist / (dt_sec/3600.0).replace({0:np.nan})
        if sp.notna().any():
            return sp
    return None

def compute_start_hour(df):
    """
    Prefer start-of-leg hour = (date + arrival_time) - drive_seconds.
    Falls back to TIME_CANDIDATES or (date,time) pairs with dayfirst=True.
    Returns a Series of hours [0..23] (may contain NaN if parsing fails).
    """
    # Best: arrival_time + drive_seconds + date (day-first)
    if {"date","arrival_time","drive_seconds"}.issubset(df.columns):
        try:
            d = pd.to_datetime(df["date"].astype(str), dayfirst=True, errors="coerce")
            t = pd.to_timedelta(df["arrival_time"].astype(str), errors="coerce")
            arr = d + t
            start = arr - pd.to_timedelta(pd.to_numeric(df["drive_seconds"], errors="coerce").fillna(0), unit="s")
            hr = start.dt.hour
            if hr.notna().any():
                return hr
        except Exception:
            pass

    # Next: any single datetime-like column
    for c in TIME_CANDIDATES:
        if c in df.columns:
            try:
                dt = pd.to_datetime(df[c], errors="coerce")
                if dt.notna().any():
                    return dt.dt.hour
            except Exception:
                pass

    # Next: compose from (date, time) with dayfirst
    for dcol, tcol in HOUR_PAIR_CANDIDATES:
        if dcol in df.columns and tcol in df.columns:
            try:
                d = pd.to_datetime(df[dcol].astype(str), dayfirst=True, errors="coerce")
                t = pd.to_timedelta(df[tcol].astype(str), errors="coerce")
                dt = d + t
                if dt.notna().any():
                    return dt.dt.hour
            except Exception:
                pass

    return pd.Series([np.nan]*len(df))

def main():
    args = parse_args()
    paths = glob.glob(args.legs_glob)
    if not paths:
        print(f"No files matched {args.legs_glob}", file=sys.stderr)
        sys.exit(1)

    frames = []
    bad = []
    sample_cols = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            sample_cols.append((os.path.basename(p), list(df.columns)[:40]))
            hr = compute_start_hour(df)
            sp = get_speed_series(df)
            tmp = pd.DataFrame({"hour": hr, "speed_kmh": sp})
            # clean
            tmp["hour"] = pd.to_numeric(tmp["hour"], errors="coerce")
            tmp["speed_kmh"] = pd.to_numeric(tmp["speed_kmh"], errors="coerce")
            tmp = tmp.dropna(subset=["hour","speed_kmh"])
            tmp = tmp[(tmp["hour"]>=0) & (tmp["hour"]<24) & (tmp["speed_kmh"]>0) & (tmp["speed_kmh"]<120)]
            if len(tmp):
                frames.append(tmp[["hour","speed_kmh"]])
            else:
                bad.append(os.path.basename(p))
        except Exception:
            bad.append(os.path.basename(p))

    if not frames:
        print("WARNING: No parsable hour/speed columns found. Writing flat profile.", file=sys.stderr)
        prof = pd.DataFrame({"hour": range(24), "speed_kmh": args.fallback_speed_kmh})
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        prof.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv} (flat {args.fallback_speed_kmh} km/h).")
        print("Sample columns from first legs file(s):", file=sys.stderr)
        for fname, cols in sample_cols[:3]:
            print(f"  {fname}: {cols}", file=sys.stderr)
        sys.exit(0)

    hod = pd.concat(frames, ignore_index=True)
    # robust cast after concat
    hod["hour"] = pd.to_numeric(hod["hour"], errors="coerce")
    hod = hod.dropna(subset=["hour"])
    hod["hour"] = hod["hour"].astype(int).clip(0,23)

    prof = (hod.groupby("hour")["speed_kmh"]
              .median()
              .reindex(range(24))
              .interpolate()
              .bfill()
              .ffill()
              .reset_index())

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    prof.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} with {len(prof)} rows.")
    if bad:
        print(f"Skipped {len(bad)} files that lacked usable columns (ok).", file=sys.stderr)

if __name__ == "__main__":
    main()
