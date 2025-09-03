import pandas as pd, numpy as np

df = pd.read_csv("outputs/kpis_driverday.csv")
df = df[df["service_minutes_assumed"]==10].copy()

keep = ["date","driver_id","co2_g_per_km","cost_per_km_vnd","S_DOBI"]
df = df[keep].dropna()

def minmax(s):
    lo, hi = s.min(), s.max()
    return (s - lo)/(hi - lo) if hi>lo else s*0

df["E_n"] = minmax(df["co2_g_per_km"])
df["C_n"] = minmax(df["cost_per_km_vnd"])
df["S_n"] = minmax(df["S_DOBI"])

vals = df[["E_n","C_n","S_n"]].to_numpy()

# Pareto test (small N is fine)
nd_mask = []
for i, a in enumerate(vals):
    dominated = False
    for j, b in enumerate(vals):
        if j==i: 
            continue
        if (b<=a).all() and (b<a).any():  # b dominates a
            dominated = True
            break
    nd_mask.append(not dominated)

df["is_pareto"] = nd_mask
df[df["is_pareto"]].to_csv("outputs/frontier_empirical.csv", index=False)
