"""Directional geometry utilities for compact earthquake sequences."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle initial bearing in degrees, clockwise from north."""
    phi1, phi2 = np.radians([lat1, lat2])
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


def axial_angle(angle_deg: float) -> float:
    """Map a direction to an axial orientation in [0, 180)."""
    return float(angle_deg % 180.0)


def axial_difference(a_deg: float, b_deg: float) -> float:
    """Smallest difference between two axial directions, in degrees."""
    d = abs((a_deg - b_deg) % 180.0)
    return float(min(d, 180.0 - d))


def local_xy_km(lat: np.ndarray, lon: np.ndarray, lat0: float | None = None, lon0: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular local projection suitable for compact regional clusters."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0 = float(np.mean(lat) if lat0 is None else lat0)
    lon0 = float(np.mean(lon) if lon0 is None else lon0)
    x = EARTH_RADIUS_KM * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = EARTH_RADIUS_KM * np.radians(lat - lat0)
    return x, y


@dataclass(frozen=True)
class OrientationResult:
    azimuth_deg: float
    axial_deg: float
    major_std_km: float
    minor_std_km: float
    anisotropy_ratio: float


def principal_orientation(lat: np.ndarray, lon: np.ndarray) -> OrientationResult:
    """Estimate the dominant epicentral axis with covariance geometry."""
    x, y = local_xy_km(lat, lon)
    xy = np.column_stack([x - x.mean(), y - y.mean()])
    cov = np.cov(xy, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    major = vectors[:, order[0]]
    azimuth = (np.degrees(np.arctan2(major[0], major[1])) + 360.0) % 360.0
    major_std, minor_std = np.sqrt(values)
    ratio = float(major_std / minor_std) if minor_std > 0 else float("inf")
    return OrientationResult(float(azimuth), axial_angle(float(azimuth)), float(major_std), float(minor_std), ratio)


def corridor_distances_km(lat: np.ndarray, lon: np.ndarray, axial_deg: float) -> np.ndarray:
    """Perpendicular distances to a line through the cluster centroid."""
    x, y = local_xy_km(lat, lon)
    x = x - x.mean()
    y = y - y.mean()
    theta = np.radians(axial_deg)
    u = np.array([np.sin(theta), np.cos(theta)])
    normal = np.array([-u[1], u[0]])
    return np.abs(np.column_stack([x, y]) @ normal)


def corridor_summary(frame: pd.DataFrame, axial_deg: float, width_km: float, lat_col: str = "latitude", lon_col: str = "longitude") -> dict[str, float]:
    """Summarize how many epicenters fall within a directional corridor."""
    d = corridor_distances_km(frame[lat_col].to_numpy(), frame[lon_col].to_numpy(), axial_deg)
    inside = d <= width_km
    return {
        "axial_deg": float(axial_deg),
        "width_km": float(width_km),
        "n_events": int(len(d)),
        "n_inside": int(inside.sum()),
        "fraction_inside": float(inside.mean()),
        "median_cross_axis_km": float(np.median(d)),
        "p90_cross_axis_km": float(np.quantile(d, 0.90)),
    }
