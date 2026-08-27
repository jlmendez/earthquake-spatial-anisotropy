"""Synthetic demonstration of earthquake spatial anisotropy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from directional_geometry import corridor_summary, principal_orientation


def make_catalogue(n: int = 700, center_lat: float = 14.45, center_lon: float = -90.69, strike_deg: float = 120.0, major_sigma_km: float = 6.0, minor_sigma_km: float = 1.2, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    along = rng.normal(0.0, major_sigma_km, n)
    across = rng.normal(0.0, minor_sigma_km, n)
    theta = np.radians(strike_deg)
    east = along * np.sin(theta) + across * np.cos(theta)
    north = along * np.cos(theta) - across * np.sin(theta)
    km_per_deg_lat = 111.195
    km_per_deg_lon = km_per_deg_lat * np.cos(np.radians(center_lat))
    lat = center_lat + north / km_per_deg_lat
    lon = center_lon + east / km_per_deg_lon
    return pd.DataFrame({"latitude": lat, "longitude": lon})


def main() -> None:
    catalog = make_catalogue()
    result = principal_orientation(catalog["latitude"], catalog["longitude"])
    corridor = corridor_summary(catalog, result.axial_deg, width_km=2.0)
    print("Principal epicentral orientation")
    print(f"  azimuth: {result.azimuth_deg:.1f}°")
    print(f"  axial orientation: {result.axial_deg:.1f}°")
    print(f"  anisotropy ratio: {result.anisotropy_ratio:.2f}")
    print("2-km directional corridor")
    print(f"  fraction inside: {corridor['fraction_inside']:.1%}")
    print(f"  median cross-axis distance: {corridor['median_cross_axis_km']:.2f} km")


if __name__ == "__main__":
    main()
