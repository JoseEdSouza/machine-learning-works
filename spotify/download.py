import os

from glob import glob
from pathlib import Path

import shutil
import kagglehub
import pandas as pd

OUTPUT_PATH = Path(__file__).parent / "data"
KAGGLE_HANDLE = "joebeachcapital/30000-spotify-songs"


def move_or_replace(source: Path, destination: Path):
    """
    Replace the destination file with the source file.
    If the destination exists, it will be overwritten.
    """
    if destination.exists():
        if destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    shutil.move(str(source), str(destination))


def convert_to_parquet(path: Path) -> Path:
    """
    Convert a CSV file to Parquet format.
    This function is a placeholder and should be implemented as needed.
    """
    df = pd.read_csv(path)
    parquet_path = path.with_suffix(".parquet")
    print(f"Converting {path} to {parquet_path}")
    df.to_parquet(parquet_path, index=False)
    path.unlink()
    return parquet_path


# Download latest version
downloaded = kagglehub.dataset_download(KAGGLE_HANDLE, force_download=True)

downloaded_files = glob(f"{downloaded}/*", recursive=True)

# Move the downloaded files to the output path
os.makedirs(OUTPUT_PATH, exist_ok=True)

for downloaded_file in downloaded_files:
    source = Path(downloaded_file)
    if source.suffix == ".csv":
        # Convert CSV to Parquet
        source = convert_to_parquet(source)
    destination = OUTPUT_PATH / source.name
    move_or_replace(source, destination)

print(f"Successfully downloaded files to {OUTPUT_PATH}")
