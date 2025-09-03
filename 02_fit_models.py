import pandas as pd, statsmodels.formula.api as smf

df = pd.read_csv("outputs/kpis_driverday.csv")
# Use BAU scenario for reporting (service_minutes_assumed==10)
df = df[df["service_minutes_assumed"]==10].copy()

# Derived predictors
df["stops_per_km"] = df["stops"] / df["distance_km"]
df["weekday"] = pd.to_datetime(df["date"]).dt.weekday  # 0=Mon

mA = smf.ols(
    "co2_g_per_km ~ speed_kmh + stops_per_km + C(driver_id) + C(weekday)",
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["driver_id"]})

print(mA.summary())
mA.save("outputs/modelA_emissions.pickle")

mB = smf.ols(
    "cost_per_km_vnd ~ distance_km + stops_per_km + C(driver_id) + C(weekday)",
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["driver_id"]})

print(mB.summary()); mB.save("outputs/modelB_cost.pickle")

# pick the congestion share you prefer (time-weighted at 15 km/h)
cong = "low_speed_share_time_t15"  # adjust if you use 12 or 18
mC = smf.ols(
    f"worked_hours ~ distance_km + stops + {cong} + C(driver_id) + C(weekday)",
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["driver_id"]})

print(mC.summary()); mC.save("outputs/modelC_hours.pickle")
