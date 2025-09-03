Models run on: .\outputs\kpis_driverday_15_drive.csv
Tag: 15_drive

Model A (Emissions):
  DV: co2_g_per_km
  IV: avg_speed_kmh, stops_per_km, driver fixed effects (C(driver_id))
  SE: HC1 robust

Model C (Worked hours):
  DV: worked_hours
  IV: distance_km, stops, low_speed_share_time_t15, driver fixed effects (C(driver_id))
  SE: HC1 robust
