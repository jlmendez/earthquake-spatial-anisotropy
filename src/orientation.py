"""Principal-orientation and anisotropy metrics for epicentral point clouds."""
from __future__ import annotations

import numpy as np


def local_xy_km(lat, lon, reference=None):
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    lat0, lon0 = reference or (float(np.mean(lat)), float(np.mean(lon)))
    y = (lat-lat0)*111.32
    x = (lon-lon0)*111.32*np.cos(np.deg2rad(lat0))
    return np.column_stack([x,y]), (lat0,lon0)


def principal_orientation(lat, lon) -> dict:
    xy, center = local_xy_km(lat, lon)
    covariance = np.cov(xy.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]; vectors = vectors[:,order]
    vector = vectors[:,0]
    azimuth = (np.degrees(np.arctan2(vector[0], vector[1])) + 360) % 180
    anisotropy = float(values[0]/max(values[1],1e-12))
    return {"azimuth_deg":float(azimuth),"anisotropy_ratio":anisotropy,"eigenvalues":values,"center":center,"principal_vector":vector}


def directional_spread(azimuth_deg):
    a = np.deg2rad(np.asarray(azimuth_deg,float)*2)
    r = np.hypot(np.mean(np.cos(a)), np.mean(np.sin(a)))
    return {"resultant_length":float(r),"circular_variance":float(1-r)}
