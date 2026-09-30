"""Download and validate the NASA Kepler KOI dataset."""

from pathlib import Path
import pandas as pd
import requests

API_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

FEATURES = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_steff",
    "koi_srad",
    "koi_smass",
    "koi_impact",
    "koi_model_snr",
]

TARGET = "koi_disposition"

QUERY = f"""
SELECT {TARGET}, {", ".join(FEATURES)}
FROM cumulative
WHERE {TARGET} IN ('CANDIDATE', 'FALSE POSITIVE')
"""

def download_dataset(output_path: str | Path) -> Path:
    """Download a small, filtered CSV from the NASA Exoplanet Archive."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        API_URL,
        params={"query": " ".join(QUERY.split()), "format": "csv"},
        timeout=60,
    )
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path

def load_and_validate(path: str | Path) -> pd.DataFrame:
    """Load the dataset and fail if required columns/data are invalid."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    required = set(FEATURES + [TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[FEATURES + [TARGET]].copy()
    df = df[df[TARGET].isin(["CANDIDATE", "FALSE POSITIVE"])].copy()

    if len(df) < 100:
        raise ValueError(f"Dataset unexpectedly small: {len(df)} rows")

    # 1 = candidate, 0 = false positive.
    df["target"] = (df[TARGET] == "CANDIDATE").astype(int)

    if df["target"].nunique() != 2:
        raise ValueError("Target must contain both classes.")

    return df

if __name__ == "__main__":
    out = Path("build/kepler_koi.csv")
    download_dataset(out)
    df = load_and_validate(out)
    print(f"Validated {len(df)} rows and {len(FEATURES)} input features.")
