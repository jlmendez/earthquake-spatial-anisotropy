"""Haversine-distance clustering for compact earthquake sequences."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

EARTH_RADIUS_KM=6371.0088


def latlon_to_unit_xyz(lat, lon):
    lat=np.deg2rad(np.asarray(lat,float)); lon=np.deg2rad(np.asarray(lon,float))
    return np.column_stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])


def cluster_epicenters(lat, lon, n_clusters=2, seed=42):
    xyz=latlon_to_unit_xyz(lat,lon)
    model=KMeans(n_clusters=n_clusters,n_init=30,random_state=seed).fit(xyz)
    return model.labels_, model


def haversine_km(lat1,lon1,lat2,lon2):
    p1,p2=np.deg2rad(lat1),np.deg2rad(lat2)
    dp=np.deg2rad(lat2-lat1); dl=np.deg2rad(lon2-lon1)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*EARTH_RADIUS_KM*np.arcsin(np.sqrt(a))
