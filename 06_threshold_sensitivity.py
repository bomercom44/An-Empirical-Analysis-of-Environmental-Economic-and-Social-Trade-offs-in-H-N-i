def time_weighted_low_share(legs, threshold):
    """
    Returns a driver-day table with the time-weighted share of low-speed driving
    at the given threshold (km/h). No groupby.apply; fully vectorized.
    """
    t = int(threshold)
    g = legs.copy()

    # indicator for "low speed" at this threshold
    g["low"] = (g["speed_kmh"] < threshold).astype(int)
    # non-negative weights
    g["w"] = g["drive_seconds"].clip(lower=0)
    # weighted indicator
    g["lw"] = g["low"] * g["w"]

    # aggregate per driver-day
    agg = (g.groupby(["date", "driver_id"], as_index=False)
             .agg(lw=("lw", "sum"), w=("w", "sum")))

    # time-weighted share
    col = f"low_time_share_t{t}"
    agg[col] = np.where(agg["w"] > 0, agg["lw"] / agg["w"], 0.0)

    return agg[["date", "driver_id", col]]