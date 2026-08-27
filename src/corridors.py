"""Distance-to-corridor tests for candidate structural directions."""
from __future__ import annotations

import numpy as np
from .orientation import local_xy_km


def corridor_distance_km(lat, lon, strike_deg: float, center=None):
    xy, center_geo = local_xy_km(lat, lon, center)
    theta = np.deg2rad(strike_deg)
    along = np.array([np.sin(theta), np.cos(theta)])
    normal = np.array([along[1], -along[0]])
    cross = np.abs(xy @ normal)
    along_coord = xy @ along
    return cross, along_coord, center_geo


def corridor_membership(lat, lon, strike_deg: float, half_width_km=3.0, center=None):
    distance, along, center_geo = corridor_distance_km(lat, lon, strike_deg, center)
    return {
        "mask": distance <= half_width_km,
        "distance_km": distance,
        "along_km": along,
        "center": center_geo,
    }


def compare_corridors(lat, lon, strikes, half_width_km=3.0):
    rows=[]
    for strike in strikes:
        result=corridor_membership(lat,lon,strike,half_width_km)
        mask=result["mask"]
        rows.append({"strike_deg":float(strike),"fraction_inside":float(mask.mean()),"median_distance_km":float(np.median(result["distance_km"]))})
    return rows
