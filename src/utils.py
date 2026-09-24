import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "raw" / "train.csv"

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points (vectorized)."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def load_sample(path=DEFAULT_TRAIN, n=40_000, seed=42):
    """Reproducible random sample so every member works on the same data."""
    df = pd.read_csv(path, parse_dates=["pickup_datetime"])
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

def add_features(df):
    """Add derived trip variables (distance, hour, weekday, speed)."""
    df = df.copy()
    df["distance_km"] = haversine(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    dt = df["pickup_datetime"].dt
    df["hour"] = dt.hour + dt.minute / 60
    df["dayofweek"] = dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    hours = df["trip_duration"].replace(0, np.nan) / 3600
    df["speed_kmh"] = df["distance_km"] / hours
    return df


def clean_trips(df, verbose=True):
    """Remove implausible trips, reporting rows dropped at each step."""
    rules = [
        ("inside NYC bounding box",
         df["pickup_latitude"].between(40.5, 41.0)
         & df["pickup_longitude"].between(-74.3, -73.6)
         & df["dropoff_latitude"].between(40.5, 41.0)
         & df["dropoff_longitude"].between(-74.3, -73.6)),
        ("duration between 1 min and 3 h",
         df["trip_duration"].between(60, 3 * 3600)),
        ("distance >= 0.1 km", df["distance_km"] >= 0.1),
        ("speed between 1 and 100 km/h", df["speed_kmh"].between(1, 100)),
    ]
    n0 = len(df)
    for name, mask in rules:
        before = len(df)
        df = df[mask.loc[df.index]]
        if verbose:
            print(f"{name:<35} removed {before - len(df):>6}  (left {len(df)})")
    if verbose:
        print(f"Total kept: {len(df)}/{n0} ({len(df)/n0:.1%})")
    return df.reset_index(drop=True)