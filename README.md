# Earthquake Spatial Anisotropy & Epicenter Geometry

Directional-analysis toolkit for earthquake epicenter sequences, focused on azimuth, axial orientation, principal directions and corridor-style spatial anisotropy.

## Why this project matters

Earthquake sequences are not always spatially isotropic. Directional patterns can reveal preferred alignments, fault-controlled geometry or migration trends that are easy to miss in ordinary latitude/longitude scatter plots.

## Highlights

- Great-circle azimuth and back-azimuth calculations
- Axial-angle handling for strike-like directions
- Local Cartesian projection for compact seismic clusters
- Principal orientation from covariance/eigen decomposition
- Corridor distance and inlier statistics
- Reproducible synthetic anisotropic catalogue for demonstration

## Tech stack

Python · NumPy · pandas · SciPy · Matplotlib

## Repository structure

- `src/directional_geometry.py` — reusable directional and corridor geometry
- `src/demo_anisotropy.py` — self-contained synthetic demonstration
- `requirements.txt` — dependencies

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/demo_anisotropy.py
```

The public demo uses synthetic epicenters. Real catalogues can be supplied as latitude/longitude data without changing the core geometry.
