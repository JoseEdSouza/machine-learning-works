import asyncio
import time
from asyncio import Semaphore
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Generator, NoReturn, Sequence, cast

import aiohttp
import polars as pl
from numpy.typing import NDArray
from haversine import haversine, Unit
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio


DATA_BASE_PATH = Path(__file__).parent / "data"

# 0 to 1, if the ratio is 0.2, the workload will be divided across N workers with N * ratio workload
PARALLEL_REQUESTS_RATIO = 0.1
PARALLEL_PROCESSING_RATIO = 0.1
WAIT_TIME_AFTER_REQUEST = True

LENGTH = 10000
RADIUS_METERS = 1000
TAGS = ["restaurant", "cafe", "bar"]
API_URL = "https://overpass-api.de/api/interpreter"

RENT_PATH = DATA_BASE_PATH / "apartments_for_rent_classified_10K.csv"

sem = Semaphore(3)


def timer[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator to measure the execution time of a function.
    Supports both synchronous and asynchronous functions.
    """
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start_time = time.time()
        if asyncio.iscoroutinefunction(func):
            result = asyncio.run(func(*args, **kwargs))
        else:
            result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result

    @wraps(func)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper


def clean_then_cast(column: str, dtype: pl.DataType) -> pl.Expr:
    """
    Replace string "null" with None and cast the column to the given dtype.
    """
    return (
        pl.when(pl.col(column) == "null")
        .then(None)
        .otherwise(pl.col(column))
        .cast(dtype)
        .alias(column)
    )


def load_rent_data(rent_path: Path, length: int | None = None) -> pl.LazyFrame:
    """
    Load rent data from CSV, clean invalid coordinates, and return a LazyFrame
    with columns: id (str), latitude (float), longitude (float).
    """
    # Define schema override
    schema = {
        "id": pl.Utf8,
        "latitude": pl.Utf8,
        "longitude": pl.Utf8,
    }

    # Load CSV lazily
    lf = pl.scan_csv(
        rent_path,
        separator=";",
        encoding="utf8-lossy",
        schema_overrides=schema,
        n_rows=length,
    ).select(["id", "latitude", "longitude"])

    # Apply cleaning and casting to numeric types
    lf = lf.with_columns(
        [
            clean_then_cast("latitude", pl.Float64),
            clean_then_cast("longitude", pl.Float64),
        ]
    )

    # Drop rows with missing coordinates
    lf = lf.drop_nulls(subset=["latitude", "longitude"])

    return lf


type Coordinate = tuple[float, float]


def haversine_meters(p1: Coordinate, p2: Coordinate) -> float:
    """
    Calculate the great-circle distance between two points on the Earth's surface using haversine formula.
    Parameters:
        p1 (Coordinate): Latitude and longitude of the first point in decimal degrees.
        p2 (Coordinate): Latitude and longitude of the second point in decimal degrees.
    Returns:
        float: The great-circle distance between the two points in meters.
    """
    return haversine(p1, p2, unit=Unit.METERS)


def build_tags_query(tags: list[str]) -> str:
    """
    Builds the tags query string for Overpass API.
    """
    return "|".join(tags)


def build_overpass_query_csv(
    locations: list[Coordinate], tags: list[str], radius: int = 1000
) -> str:
    """
    Builds the Overpass API query for the given locations and tags, returning results in CSV format.
    """
    tags_query = "|".join(tags)  # For regex matching
    query_parts = [
        f"""
        node["amenity"~"{tags_query}"](around:{radius},{lat},{lon});
        way["amenity"~"{tags_query}"](around:{radius},{lat},{lon});
        relation["amenity"~"{tags_query}"](around:{radius},{lat},{lon});
        """
        for lat, lon in locations
    ]

    full_query = f"""
        [out:csv(::id, ::type, name, amenity, ::lat, ::lon; true; ",")];
        (
            {"".join(query_parts)}
        );
        out center;
    """

    return full_query


async def get_coordinates(lf: pl.LazyFrame) -> list[Coordinate]:
    """
    Extract latitude and longitude from the DataFrame and return as a list of tuples.
    """
    df = await lf.select("latitude", "longitude").collect_async()
    zipped = df.to_numpy().tolist()
    return list(map(tuple, zipped))


@timer
async def request_overpass_api(query: str) -> BytesIO | None:
    """
    Asynchronously makes a request to the Overpass API with the given query and returns the response.
    """
    print("Sending the request")

    buffer = BytesIO()

    for _ in range(5):
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, data={"data": query}) as response:
                if response.status == 409:
                    print("Too many requests, waiting 10 seconds")
                    await asyncio.sleep(10)
                    continue # Retry on 409 Conflict
                elif response.status == 200:
                    print("Request successful")
                else:
                    response.raise_for_status()

                async for chunk in tqdm_asyncio(
                    response.content.iter_chunked(1024),
                    unit="KB",
                    unit_scale=True,
                    desc="Downloading Overpass Data",
                ):
                    buffer.write(chunk)

                buffer.flush()
                buffer.seek(0)
                return buffer

    return None


def chunkify[T](
    data: Sequence[T],
    chunk_size: int | None = None,
    ratio: float | None = None,
    start: int = 0,
) -> Generator[list[T], None, NoReturn]:
    """
    Generate chunks of data from the given iterable with a specified chunk size, starting from a given index.
    """
    if chunk_size and ratio:
        raise ValueError("Cannot specify both chunk_size and ratio")

    if not chunk_size and not ratio:
        ratio = 1

    if ratio:
        chunk_size = int(len(data) * ratio)

    data = list(data)

    for i in range(start, len(data), chunk_size):
        yield data[i : i + chunk_size]


@timer
async def fetch_overpass_data(
    coords: list[Coordinate], tags: list[str], ratio: float | None = None
) -> pl.LazyFrame:
    
    async def process_chunk_task(chunk: list[Coordinate]) -> pl.DataFrame:
        query = build_overpass_query_csv(chunk, tags)

        async with sem:
            buffer = await request_overpass_api(query)
            if buffer is None:
                raise Exception("Failed to get Overpass API response")
            if WAIT_TIME_AFTER_REQUEST:
                print("Waiting 10 seconds after request")
                await asyncio.sleep(10)  # Avoid hitting the rate limit

        return pl.scan_csv(
            buffer,
            separator=",",
            encoding="utf8",
            schema_overrides={
                "@id": pl.Utf8,
                "name": pl.Utf8,
                "amenity": pl.Utf8,
                "@lat": pl.Float64,
                "@lon": pl.Float64,
            },
        )

    tasks = (process_chunk_task(chunk) for chunk in chunkify(coords, ratio=ratio))
    lfs = await asyncio.gather(*tasks)

    return pl.concat(lfs).sort(by="@id")


def process_apartment_chunk(
    chunk: list[dict[str, Any]], pois_np: NDArray, radius_meters: int
) -> list[dict[str, int]]:
    chunk_results = []
    with tqdm(chunk, desc="Processing apartments") as pbar:
        for row in pbar:
            pbar.set_postfix({"id": row["id"]})

            apt_id = row["id"]
            apt_coord = (row["latitude"], row["longitude"])

            nearby_counts = {}

            for poi_lat, poi_lon, tag in pois_np:
                poi_coord = (poi_lat, poi_lon)
                dist = haversine_meters(apt_coord, poi_coord)

                if dist <= radius_meters:
                    tag = cast(str, tag)
                    nearby_counts[tag] = nearby_counts.get(tag, 0) + 1

            chunk_results.extend(
                [
                    {"id": apt_id, "amenity": tag, "count": count}
                    for tag, count in nearby_counts.items()
                ]
            )
    
    return chunk_results


@timer
def count_pois_near_apartments(
    apartments: pl.DataFrame,
    pois: pl.DataFrame,
    radius_meters: float = 1000.0,
    ratio: float | None = None,
) -> pl.DataFrame:
    pois_np = pois.select(["@lat", "@lon", "amenity"]).to_numpy()

    worker = partial(
        process_apartment_chunk, pois_np=pois_np, radius_meters=radius_meters
    )

    apartment_rows = list(apartments.iter_rows(named=True))
    apartment_chunks = list(chunkify(apartment_rows, ratio=ratio))

    with ThreadPoolExecutor() as executor:
        all_results = executor.map(worker, apartment_chunks)

    results = chain.from_iterable(all_results)

    return pl.DataFrame(results)


@timer
async def main():
    rent = load_rent_data(RENT_PATH, LENGTH)
    coordinates = await get_coordinates(rent)
    result = await fetch_overpass_data(
        coordinates, TAGS, ratio=PARALLEL_REQUESTS_RATIO
    )

    result_loaded = await result.collect_async()
    rent_loaded = await rent.collect_async()

    final_result = count_pois_near_apartments(
        rent_loaded, result_loaded, RADIUS_METERS, ratio=PARALLEL_PROCESSING_RATIO
    )

    print(final_result)
    final_result.write_parquet("result.parquet", compression="zstd")


if __name__ == "__main__":
    asyncio.run(main())
