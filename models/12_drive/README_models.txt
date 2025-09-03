Models run on: .\outputs\kpis_driverday_12_drive.csv
Tag: 12_drive

Model A (Emissions):
  DV: co2_g_per_km
  IV: avg_speed_kmh, stops_per_km, driver fixed effects (C(driver_id))
  SE: HC1 robust

Model C (Worked hours):
  DV: worked_hours
  IV: distance_km, stops, low_speed_share_time_t12, driver fixed effects (C(driver_id))
  SE: HC1 robust
