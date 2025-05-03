import tempfile
import zipfile

from io import BytesIO
from functools import wraps
from pathlib import Path
from typing import Callable

import py7zr
import requests
from tqdm import tqdm

OUTPUT_PATH = Path(__file__).parent / "data"

URL = "https://archive.ics.uci.edu/static/public/555/apartment+for+rent+classified.zip"

def timer[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator to measure the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@timer
def download_zip_dataset(url: str, buffer_: BytesIO):
    """
    Download a zip dataset from a given URL and save it to the specified output path.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading ZIP") as pbar:
        for chunk in response.iter_content(chunk_size=1024):
            buffer_.write(chunk)
            pbar.update(len(chunk))
    
    buffer_.flush()  # Ensure all data is written to the buffer
    buffer_.seek(0)  # Move the cursor to the beginning of the BytesIO object

@timer
def extract_zip_from_buffer(buffer_: BytesIO, output_path: Path):
    """
    Extract a zip file from a BytesIO buffer to the specified output path.
    """
    with zipfile.ZipFile(buffer_, "r") as zip_ref:
        zip_ref.extractall(output_path)


def extract_7z_files_from_folder(
    folder_path: Path,
    output_path: Path,
    filters: list[Callable[[str], bool]] | None = None,
):
    """
    Extract all 7z files from a folder to the specified output path.
    Optionally, filter the files to be extracted based on a list of filter functions.
    Each filter function should take a string (the file name) and return a boolean.
    """
    if filters:
        filters = [f for f in filters if callable(f)]

    files = list(folder_path.glob("*.7z"))

    if not files:
        print(f"No 7z files found in {folder_path}.")
        return
    
    print(f"Found {len(files)} 7z files in {folder_path}.")
    
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    if not output_path.is_dir():
        output_path = output_path.parent

    tqdm.write(f"Extracting {len(files)} 7z files to {output_path}...")
    with tqdm(total=len(files), desc="Extracting 7z files", unit="file") as pbar:
        for file_7z in files:
            if filters and not any(filter_(file_7z.name) for filter_ in filters):
                pbar.update(1)
                continue

            with py7zr.SevenZipFile(file_7z, "r") as ref_7z:
                ref_7z.extractall(output_path)

            try:
                file_7z.unlink()  # Safely remove the 7z file after extraction
            except OSError as e:
                print(f"Error deleting file {file_7z}: {e}")
            
            pbar.update(1)

@timer
def process_and_extract_dataset(url: str, output_path: Path, filters: list[Callable[[str], bool]] | None = None):
    """
    Download a dataset from a URL, extract its contents, and process 7z files with optional filters.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        buffer = BytesIO()
        download_zip_dataset(url, buffer)
        extract_zip_from_buffer(buffer, temp_dir_path)
        extract_7z_files_from_folder(temp_dir_path, output_path, filters=filters)


if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Define filters for the 7z files to be extracted
    filters = [
        lambda f: "10K" in f,
    ]

    # Process and extract the dataset
    process_and_extract_dataset(URL, OUTPUT_PATH, filters=filters)