# 04b_posthoc_fix_ec_and_deltas.py  (force km source + robust fixed-cost fallback + logging)
import argparse, os, glob, pandas as pd, numpy as np, yaml, re
from datetime import datetime, date as dtdate, time as dttime, timedelta

CF_KM_CANDS  = ["km_day","distance_km_day","total_km","km","dist_km"]
SCH_KM_CANDS = ["distance_km","leg_distance_km","km","dist_km","trip_km"]
SCH_M_CANDS  = ["distance_m","dist_m","meters","metres","length_m"]
LEGS_KM_CANDS= ["distance_km","leg_distance_km","km","dist_km"]
LEGS_M_CANDS = ["distance_m","drive_m","dist_m","meters","metres","length_m"]
KPIS_KM_CANDS= ["distance_km","km_day","total_km","km","dist_km"]

def hhmm(s): return datetime.strptime(s, "%H:%M").time()
def load_cfg(path):
    with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def norm_keys(df, who):
    df=df.copy()
    if "driver_id" not in df.columns or "date" not in df.columns:
        raise SystemExit(f"{who}: needs driver_id and date")
    df["driver_id"]=df["driver_id"].astype(str).str.strip()
    df["driver_digits"]=df["driver_id"].str.extract(r"(\d+)",expand=False)
    d=pd.to_datetime(df["date"],errors="coerce", dayfirst=False)
    if d.notna().sum()==0:
        for alt in ["service_date","date_local","day","date_str"]:
            if alt in df.columns:
                d=pd.to_datetime(df[alt],errors="coerce"); break
    df["date"]=d.dt.date.astype(str)
    return df

def derive_km_day_from_cf(cf):
    for c in CF_KM_CANDS:
        if c in cf.columns:
            km=pd.to_numeric(cf[c],errors="coerce")
            if km.notna().any():
                out = cf[["driver_id","driver_digits","date"]].assign(km_day=km)
                print(f"[km_day] from CF: {out.dropna(subset=['km_day']).shape[0]} rows")
                return out
    return None

def derive_km_day_from_schedule(sch, force_col=None):
    if force_col and force_col in sch.columns:
        d=pd.to_numeric(sch[force_col],errors="coerce")
        km=d if not re.search(r"(m|meter)",force_col,re.I) else d/1000.0
        out=(sch.assign(km_day=km)[["driver_id","driver_digits","date","km_day"]]
             .groupby(["driver_id","driver_digits","date"],as_index=False)["km_day"].sum())
        print(f"[km_day] from SCHEDULE[{force_col}]: {out.shape[0]} keys")
        return out
    for c in SCH_KM_CANDS:
        if c in sch.columns:
            d=pd.to_numeric(sch[c],errors="coerce")
            out=(sch.assign(km_day=d)[["driver_id","driver_digits","date","km_day"]]
                 .groupby(["driver_id","driver_digits","date"],as_index=False)["km_day"].sum())
            print(f"[km_day] from SCHEDULE[{c}]: {out.shape[0]} keys")
            return out
    for c in SCH_M_CANDS:
        if c in sch.columns:
            d=pd.to_numeric(sch[c],errors="coerce")/1000.0
            out=(sch.assign(km_day=d)[["driver_id","driver_digits","date","km_day"]]
                 .groupby(["driver_id","driver_digits","date"],as_index=False)["km_day"].sum())
            print(f"[km_day] from SCHEDULE[{c}km]: {out.shape[0]} keys")
            return out
    return None

def derive_km_day_from_legs(legs_glob):
    paths=sorted(glob.glob(legs_glob)); rows=[]
    for p in paths:
        try:
            df=pd.read_csv(p)
            if not {"driver_id","date"}.issubset(df.columns): continue
            df=norm_keys(df,f"LEGS:{os.path.basename(p)}")
            km=None
            for c in LEGS_KM_CANDS:
                if c in df.columns: km=pd.to_numeric(df[c],errors="coerce"); break
            if km is None:
                for c in LEGS_M_CANDS:
                    if c in df.columns: km=pd.to_numeric(df[c],errors="coerce")/1000.0; break
            if km is None: continue
            part=(df[["driver_id","driver_digits","date"]].assign(km_day=km)
                  .groupby(["driver_id","driver_digits","date"],as_index=False)["km_day"].sum())
            print(f"[km_day] from LEGS {os.path.basename(p)}: {part.shape[0]} keys")
            rows.append(part)
        except Exception:
            pass
    if not rows: return None
    allp=pd.concat(rows,ignore_index=True)
    out=allp.groupby(["driver_id","driver_digits","date"],as_index=False)["km_day"].sum()
    print(f"[km_day] LEGS union: {out.shape[0]} keys")
    return out

def derive_km_day_from_kpis(kpis_patterns):
    patterns=[p.strip() for p in kpis_patterns.split(";") if p.strip()]
    if os.path.exists(".\\kpis_bau.csv") and ".\\kpis_bau.csv" not in patterns:
        patterns.append(".\\kpis_bau.csv")
    paths=[]
    for pat in patterns:
        paths.extend(glob.glob(pat))
    paths=sorted(set(paths))
    if not paths:
        print("[km_day] KPI fallback: no files matched patterns:", patterns)
        return None
    rows=[]
    for p in paths:
        try:
            df=pd.read_csv(p)
            if not {"driver_id","date"}.issubset(df.columns): continue
            df=norm_keys(df,f"KPI:{os.path.basename(p)}")
            km=None
            for c in KPIS_KM_CANDS:
                if c in df.columns:
                    km=pd.to_numeric(df[c],errors="coerce"); break
            if km is None:
                print(f"[km_day] KPI {os.path.basename(p)} has no distance column")
                continue
            part=(df[["driver_id","driver_digits","date"]].assign(km_day=km)
                  .groupby(["driver_id","driver_digits","date"],as_index=False)["km_day"].sum())
            print(f"[km_day] KPI {os.path.basename(p)}: {part.shape[0]} keys")
            rows.append(part)
        except Exception as e:
            print(f"[km_day] KPI {os.path.basename(p)} skipped: {e}")
            pass
    if not rows: return None
    allp=pd.concat(rows,ignore_index=True)
    out=allp.groupby(["driver_id","driver_digits","date"],as_index=False)["km_day"].sum()
    print(f"[km_day] KPI union: {out.shape[0]} keys from {len(paths)} files")
    return out

def earliest_bau_departure(schedule_df):
    out={}; cand=["start_time_local","start_time","departure_time_local","departure_time","arrival_time"]
    have_ds="drive_seconds" in schedule_df.columns
    for (drv,dt),sub in schedule_df.groupby(["driver_id","date"]):
        t=None
        if have_ds and "arrival_time" in sub.columns:
            try:
                ss=pd.to_datetime(sub["date"].astype(str)+" "+sub["arrival_time"].astype(str),errors="coerce")
                ds=pd.to_numeric(sub["drive_seconds"],errors="coerce")
                start=ss-pd.to_timedelta(ds.fillna(0),unit="s")
                tvals=start.dt.time.dropna()
                if len(tvals): t=min(tvals)
            except Exception: pass
        if t is None:
            for c in cand:
                if c in sub.columns:
                    try: tvals=pd.to_datetime(sub[c].astype(str),errors="coerce",format="%H:%M:%S",exact=False)
                    except Exception: tvals=pd.to_datetime(sub[c].astype(str),errors="coerce")
                    tvals=tvals.dt.time.dropna()
                    if len(tvals): t=min(tvals); break
        out[(drv,dt)]= t or dttime(8,0)
    return out

def build_daily_fixed_cost_map(schedule_df, cfg, cf_points):
    """Fixed pot per day; if schedule lacks a date, fall back to CF to count active drivers that day."""
    mc=cfg["monthly_costs"]
    pot=(mc["driver_vnd"]+mc["assistant_vnd"]+mc["maintenance_vnd"]+mc["other_vnd"]) / float(cfg["working_days"])

    # active drivers per day from schedule
    sch_active = schedule_df[["driver_id","date"]].drop_duplicates()
    sch_ct = sch_active.groupby("date")["driver_id"].nunique()

    # active drivers per day from counterfactuals (robust for March)
    cf_active = cf_points[["driver_id","date"]].drop_duplicates()
    cf_ct = cf_active.groupby("date")["driver_id"].nunique()

    m={}
    dates = sorted(set(sch_ct.index).union(set(cf_ct.index)))
    for dt in dates:
        nd = sch_ct.get(dt, None)
        if nd is None or nd == 0:
            nd = cf_ct.get(dt, 1)
        nd = max(1,int(nd))
        per_driver = pot/nd
        # assign to each active driver that day (use CF if schedule missing)
        active = sch_active.query("date == @dt")
        if active.empty:
            active = cf_active.query("date == @dt")
        for _,r in active.iterrows():
            m[(r["driver_id"], dt)] = per_driver
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--counterfactual_csv", default=r".\sim_out\counterfactual_points_norm.csv")
    ap.add_argument("--schedule_csv", default=r".\schedule_with_arrival_times.csv")
    ap.add_argument("--config", default=r".\sim_config.yaml")
    ap.add_argument("--out_csv", default=r".\sim_out\counterfactual_points_ecfixed.csv")
    ap.add_argument("--schedule_distance_col", default=None)
    ap.add_argument("--legs_glob", default=r".\legs_enriched_*_drive.csv")
    ap.add_argument("--kpis_glob", default=r".\kpis_driverday_*_drive.csv;.\kpis_bau.csv")
    ap.add_argument("--driver_map_csv", default=None)
    ap.add_argument("--force_km_source", choices=["auto","cf","schedule","legs","kpis"], default="auto")
    args=ap.parse_args()

    cfg=load_cfg(args.config)
    fuel_price=float(cfg["fuel"]["price_vnd_per_litre"])
    cfv=float(cfg["fuel"]["carbon_factor_kg_per_litre"])
    uh=cfg.get("unsocial_hours",{})
    uh_en=bool(uh.get("enabled",False))
    uh_cut=hhmm(uh.get("start_before","08:00"))
    uh_prem=float(uh.get("premium_vnd_per_hour",0.0))

    cf_points=norm_keys(pd.read_csv(args.counterfactual_csv),"CF")
    sch=norm_keys(pd.read_csv(args.schedule_csv),"SCHEDULE")

    if args.driver_map_csv and os.path.exists(args.driver_map_csv):
        dm=pd.read_csv(args.driver_map_csv)
        dm["cf_driver_id"]=dm["cf_driver_id"].astype(str).str.strip()
        dm["schedule_driver_id"]=dm["schedule_driver_id"].astype(str).str.strip()
        cf_points=cf_points.merge(dm.rename(columns={"cf_driver_id":"driver_id","schedule_driver_id":"mapped"}),on="driver_id",how="left")
        cf_points["driver_id"]=cf_points["mapped"].fillna(cf_points["driver_id"])
        cf_points.drop(columns=["mapped"],inplace=True)
        cf_points["driver_digits"]=cf_points["driver_id"].str.extract(r"(\d+)",expand=False)

    # ----- Build km_day with forced source if requested -----
    km_day=None; used="(none)"
    if args.force_km_source in {"cf","schedule","legs","kpis"}:
        if args.force_km_source=="cf":
            km_day = derive_km_day_from_cf(cf_points); used="CF"
        elif args.force_km_source=="schedule":
            km_day = derive_km_day_from_schedule(sch, args.schedule_distance_col); used="SCHEDULE"
        elif args.force_km_source=="legs":
            km_day = derive_km_day_from_legs(args.legs_glob); used="LEGS"
        elif args.force_km_source=="kpis":
            km_day = derive_km_day_from_kpis(args.kpis_glob); used="KPIS"
    else:
        km_day = (derive_km_day_from_cf(cf_points) or
                  derive_km_day_from_schedule(sch, args.schedule_distance_col) or
                  derive_km_day_from_legs(args.legs_glob) or
                  derive_km_day_from_kpis(args.kpis_glob))
        used="AUTO"

    if km_day is None:
        raise SystemExit("km_day could not be built from chosen source.")

    print(f"[km_day] Source used: {used}  {km_day.shape[0]} keys")

    # exact join then digits fallback
    m=cf_points.merge(km_day[["driver_id","date","km_day"]],on=["driver_id","date"],how="left")
    miss=m["km_day"].isna()
    if miss.any():
        if "driver_digits" not in km_day.columns:
            km_day["driver_digits"]=km_day["driver_id"].astype(str).str.extract(r"(\d+)",expand=False)
        m=m.merge(km_day[["driver_digits","date","km_day"]].rename(columns={"km_day":"km_day_digits"}),on=["driver_digits","date"],how="left")
        m.loc[miss,"km_day"]=m.loc[miss,"km_day_digits"]; m.drop(columns=["km_day_digits"],inplace=True)

    if m["km_day"].isna().any():
        bad=m.loc[m["km_day"].isna(),["driver_id","driver_digits","date"]].drop_duplicates()
        bad.to_csv(os.path.join(os.path.dirname(args.out_csv) or ".","unmatched_driverday_for_km_day.csv"),index=False)
        raise SystemExit(f"km_day still missing for {bad.shape[0]} driver-days. See unmatched_driverday_for_km_day.csv.")

    # Fill shift/service defaults if missing
    if "shift_min" not in m.columns:   m["shift_min"]=0
    if "service_min" not in m.columns: m["service_min"]=10

    # Infer E_g_per_km if absent
    if "E_g_per_km" not in m.columns:
        lc={c.lower():c for c in m.columns}; got=None
        for n in lc:
            if ("g" in n and "km" in n) or ("per_km" in n and ("g" in n or "co2" in n or "ef" in n)):
                got=lc[n]; break
        if got is not None:
            m["E_g_per_km"]=pd.to_numeric(m[got],errors="coerce")
        elif "E_total_g" in m.columns:
            m["E_g_per_km"]=pd.to_numeric(m["E_total_g"],errors="coerce")/pd.to_numeric(m["km_day"],errors="coerce")
        else:
            raise SystemExit("Could not infer E_g_per_km in CF.")

    # Fixed-cost map (uses schedule, but falls back to CF if schedule lacks the date)
    fixed_map=build_daily_fixed_cost_map(sch,cfg,cf_points)
    bau_start=earliest_bau_departure(sch)
    day0=dtdate(2024,1,1)

    out=[]
    for (drv,dt), sub in m.groupby(["driver_id","date"],sort=False):
        mask=(sub["shift_min"]==0) & (sub["service_min"]==10)
        if "resequencer" in sub.columns: mask=mask & (sub["resequencer"]=="BAU")
        if not mask.any(): mask=(sub["shift_min"]==0)
        bauE=float(sub.loc[mask,"E_g_per_km"].iloc[0])
        bauKM=float(sub.loc[mask,"km_day"].iloc[0])
        base_fixed=fixed_map.get((drv,dt),0.0)
        bau_litres=(bauE/1000.0)/cfv * bauKM
        bau_fuel_vnd=bau_litres*float(cfg["fuel"]["price_vnd_per_litre"])
        bau_Ec_km=(base_fixed+bau_fuel_vnd)/max(1e-9,bauKM)
        t0=bau_start.get((drv,dt),dttime(8,0))
        for _,r in sub.iterrows():
            km=float(r["km_day"]); Eg=float(r["E_g_per_km"])
            litres=(Eg/1000.0)/cfv * km
            fuel_vnd=litres*float(cfg["fuel"]["price_vnd_per_litre"])
            daily_fixed=base_fixed
            if uh_en:
                dep=datetime.combine(day0,t0)+timedelta(minutes=float(r["shift_min"]))
                if dep.time()<uh_cut:
                    early=(datetime.combine(day0,uh_cut)-dep).total_seconds()/3600.0
                    daily_fixed += early*uh_prem
            Ec_km=(daily_fixed+fuel_vnd)/max(1e-9,km)
            out.append({**r,
                "daily_fixed_vnd":daily_fixed,
                "fuel_cost_vnd":fuel_vnd,
                "Ec_cost_per_km_vnd":Ec_km,
                "E_bau_gkm":bauE,
                "Ec_bau_vndkm":bau_Ec_km,
                "dE_gkm":Eg-bauE,
                "dEc_vndkm":Ec_km-bau_Ec_km
            })

    out=pd.DataFrame(out)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv,index=False)
    print(f"Wrote {args.out_csv} (km_day source={used}; fixed-cost fallback uses CF when schedule lacks dates).")
if __name__=="__main__":
    main()
