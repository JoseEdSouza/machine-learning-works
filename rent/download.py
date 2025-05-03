import os
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Callable

import py7zr
import requests

OUTPUT_PATH = Path(__file__).parent / "data"

API_BASE_URL = (
    "https://archive.ics.uci.edu/static/public/555/apartment+for+rent+classified.zip"
)


def download_zip_dataset(url: str, buffer_: BytesIO):
    """
    Download a zip dataset from a given URL and save it to the specified output path.
    """
    response = requests.get(url)
    response.raise_for_status()

    # write in bytes
    buffer_.write(response.content)
    buffer_.seek(0)  # Move the cursor to the beginning of the BytesIO object


def uzip_buffer(buffer_: BytesIO, output_path: Path):
    """
    Unzip the downloaded file and save it to the specified output path.
    """
    with zipfile.ZipFile(buffer_, "r") as zip_ref:
        zip_ref.extractall(output_path)


def unzip_7z_files_in_folder(
    folder_path: Path,
    output_path: Path,
    filters: list[Callable[[str], bool]] | None = None,
):
    """
    Unzip all zip files in the specified folder.
    """
    check_for_filters = bool(filters)
    for file_7z in folder_path.glob("*.7z"):
        if check_for_filters and not any(f(file_7z.name) for f in filters):
            continue

        with py7zr.SevenZipFile(file_7z, "r") as ref_7z:
            ref_7z.extractall(output_path)
        os.remove(file_7z)  # Remove the zip file after extraction


with tempfile.TemporaryDirectory() as temp_dir:
    temp_dir_path = Path(temp_dir)
    buffer = BytesIO()
    download_zip_dataset(API_BASE_URL, buffer)
    uzip_buffer(buffer, temp_dir_path)
    unzip_7z_files_in_folder(temp_dir_path, OUTPUT_PATH, filters=[lambda f: "10K" in f])
