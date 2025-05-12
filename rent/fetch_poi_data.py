import asyncio
import time
from asyncio import Semaphore
from functools import partial, wraps
from io import BytesIO
from itertools import chain
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Generator,
    Hashable,
    NoReturn,
    Sequence,
    Iterable,
    cast,
)

import aiohttp
import numpy as np
import polars as pl
from haversine import haversine, Unit
from numpy.typing import NDArray
from sklearn.neighbors import BallTree
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio
import hashlib

DATA_BASE_PATH = Path(__file__).parent / "data"
CACHE_BASE_PATH = DATA_BASE_PATH / "cache"

EARTH_RADIUS = 6_371_000.0  # in meters

PARALLEL_REQUESTS_JOBS = 20
PARALLEL_PROCESSING_JOBS = None
SEMAPHORE_LIMIT = 2

WAIT_TIME_AFTER_REQUEST = True
DELAY_BETWEEN_REQUESTS = 20  # seconds

LENGTH = 10_000
RADIUS_METERS = 1_000
TAGS = [
    "bar",
    "pub",
    "restaturant",
    "fast_food",
    "college",
    "university",
    "school",
    "bus_station",
    "hospital",
    "pharmacy",
    "clinic",
    "cinema",
    "police",
    "fast_court",
]


API_URL = "https://overpass-api.de/api/interpreter"

RENT_PATH = DATA_BASE_PATH / "apartments_for_rent_classified_10K.csv"

sem = Semaphore(SEMAPHORE_LIMIT)


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
    locations: Sequence[Coordinate], tags: list[str], radius: int = RADIUS_METERS
) -> str:
    """
    Builds the Overpass API query for the given locations and tags, returning results in CSV format.
    """
    tags_query = "|".join(tags)  # For regex matching
    query_parts = [
        f"""
        node["amenity"~"^{tags_query}$"](around:{radius},{lat},{lon});
        way["amenity"~"^{tags_query}$"](around:{radius},{lat},{lon});
        relation["amenity"~"^{tags_query}$"](around:{radius},{lat},{lon});
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
    return list(map(tuple, zipped))  # type: ignore


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
                    await asyncio.sleep(20)
                    continue  # Retry on 409 Conflict
                elif response.status == 200:
                    print("Request successful")
                else:
                    response.raise_for_status()

                async for chunk in tqdm_asyncio(
                    cast(Iterable[bytes], response.content.iter_chunked(1024)),
                    unit="KB",
                    unit_scale=True,
                    desc="Downloading Overpass Data",
                ):  # type: ignore
                    buffer.write(chunk)

                buffer.flush()
                buffer.seek(0)
                return buffer

    return None


def chunkify[T](
    data: Sequence[T],
    chunk_size: int | None = None,
    ratio: float | None = None,
    n_chunks: int | None = None,
) -> Generator[Sequence[T], None, NoReturn]:
    """
    Generate chunks of data from the given iterable with a specified chunk size, ratio, or number of chunks,
    starting from a given index.
    """
    if sum(bool(param) for param in [chunk_size, ratio, n_chunks]) > 1:
        raise ValueError("Specify only one of chunk_size, ratio, or n_chunks")

    data = np.array(data)

    if len(data) == 0:
        yield []
        return

    if sum(bool(param) for param in [chunk_size, ratio, n_chunks]) != 1:
        raise ValueError("Specify exactly one of chunk_size, ratio, or n_chunks")

    data = np.array(data)
    if len(data) == 0:
        yield []
        return

    if chunk_size is not None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size].tolist()

    elif ratio is not None:
        if ratio <= 0 or ratio > 1:
            raise ValueError("ratio must be between 0 and 1")
        chunk_size = max(1, int(len(data) * ratio))
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size].tolist()

    elif n_chunks is not None:
        if n_chunks <= 0:
            raise ValueError("n_chunks must be greater than 0")
        indices = np.linspace(0, len(data), n_chunks + 1, dtype=int)
        for start, end in zip(indices[:-1], indices[1:]):
            yield data[start:end].tolist()


async def load_or_cache_fetched_data(
    soruce: Hashable, cache_dir: Path, processor: Callable[[], Awaitable[pl.LazyFrame]]
) -> pl.LazyFrame:
    """
    Load or cache fetched data from the Overpass API.
    If the data is already cached, it will be loaded from the cache.
    Otherwise, it will be fetched and saved to the cache.
    """
    source_hash = hashlib.sha256(str(soruce).encode("utf-8")).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"coords_cache_{source_hash}.parquet"
    relative_path = cache_path.relative_to(cache_dir)
    if cache_path.exists():
        print(f"CACHE HIT: Loading cached data from {relative_path}")
        return pl.scan_parquet(cache_path)
    else:
        print(f"CACHE MISS: Fetching data and saving to {relative_path}")
        data = await processor()
        if data.limit(1).collect().is_empty():
            raise ValueError("No data fetched from Overpass API")
        collected_data = await data.collect_async()
        collected_data.write_parquet(cache_path, compression="zstd")
        print(f"Saved fetched data to cache at {relative_path}")
        return data


@timer
async def fetch_overpass_data(
    coords: list[Coordinate], tags: list[str], n_jobs: int | None = None
) -> pl.LazyFrame:

    async def process_chunk_task(chunk: Sequence[Coordinate], pbar: tqdm_asyncio) -> pl.LazyFrame:
        query = build_overpass_query_csv(chunk, tags)

        async with sem:
            buffer = await request_overpass_api(query)
            pbar.update(1)
            if buffer is None:
                raise Exception("Failed to get Overpass API response")
            if WAIT_TIME_AFTER_REQUEST and bool(n_jobs):
                print(f"Waiting {DELAY_BETWEEN_REQUESTS} seconds after request")
                await asyncio.sleep(
                    DELAY_BETWEEN_REQUESTS
                )  # Avoid hitting the rate limit

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

    

    async def process_coordinates_data() -> pl.LazyFrame:
        pbar = tqdm_asyncio(
            total=n_jobs,
            desc="Processing chunks",
            unit="chunk",
        )

        tasks = (
            process_chunk_task(chunk, pbar) for chunk in chunkify(coords, n_chunks=n_jobs)
        )

        lfs = await asyncio.gather(*tasks)

        pbar.close()

        return pl.concat(lfs).unique(subset="@id").sort(by="@id")

    result =  await load_or_cache_fetched_data(
        soruce={"tags": tags, "coords": coords, "radius": RADIUS_METERS},
        cache_dir=CACHE_BASE_PATH,
        processor=process_coordinates_data,
    )

    return result


type Row = dict[str, Any]


def process_apartment_chunk(
    chunk: Sequence[Row],
    amenities: NDArray[str],  # type: ignore
    tree: BallTree,
    radius_rad: float,
) -> list[dict[str, int]]:
    apt_coords_rad = np.radians(
        np.array([[row["latitude"], row["longitude"]] for row in chunk])
    )

    neighbors = tree.query_radius(apt_coords_rad, r=radius_rad)

    results = []
    with tqdm(chunk, desc="Processing apartments") as pbar:
        for apt, poi_indexes in zip(chunk, neighbors):
            apt_id = apt["id"]
            nearby_counts: dict[str, int] = {}
            for idx in poi_indexes:
                tag = str(amenities[idx])
                nearby_counts[tag] = nearby_counts.get(tag, 0) + 1

            results.extend(
                {"id": apt_id, "amenity": tag, "count": count}
                for tag, count in nearby_counts.items()
            )
            pbar.set_postfix({"id": apt_id})
            pbar.update(1)

    return results


@timer
def count_pois_near_apartments(
    apartments: pl.DataFrame,
    pois: pl.DataFrame,
    radius_meters: float = 1000.0,
    n_jobs: int | None = None,
) -> pl.DataFrame:
    pois_np: NDArray = pois.select(["@lat", "@lon", "amenity"]).to_numpy()

    radius_rad = radius_meters / EARTH_RADIUS

    coords = np.radians(
        pois_np[:, :2].astype(np.float64)
    )  # Extract latitude and longitude
    amenities = pois_np[:, 2]  # Extract amenities

    tree = BallTree(coords, metric="haversine", leaf_size=40)  # type: ignore

    worker = partial(
        process_apartment_chunk, amenities=amenities, tree=tree, radius_rad=radius_rad
    )

    if n_jobs is None:
        n_jobs = cpu_count()

    apartment_rows = list(apartments.iter_rows(named=True))
    apartment_chunks = chunkify(apartment_rows, n_chunks=n_jobs)

    with Pool(processes=cpu_count()) as pool:
        all_results = pool.map(worker, apartment_chunks)

    results = chain.from_iterable(all_results)

    return pl.DataFrame(results)


@timer
async def main():
    rent = load_rent_data(RENT_PATH, LENGTH)
    coordinates = await get_coordinates(rent)
    result = await fetch_overpass_data(coordinates, TAGS, n_jobs=PARALLEL_REQUESTS_JOBS)

    result_loaded = await result.collect_async()
    rent_loaded = await rent.collect_async()

    print("Found POIs:", result_loaded.shape[0])

    final_result = count_pois_near_apartments(
        rent_loaded, result_loaded, RADIUS_METERS, n_jobs=PARALLEL_PROCESSING_JOBS
    )

    print(final_result)

    info = {
        "tags": TAGS,
        "radius": RADIUS_METERS,
        "coords": coordinates,
    }
    info_hash = hashlib.sha256(str(info).encode("utf-8")).hexdigest()

    output_path = DATA_BASE_PATH / f"poi-data-count-{info_hash}.parquet"
    final_result.write_parquet(output_path, compression="zstd")
    print(f"Saved results to {output_path.relative_to(DATA_BASE_PATH.parent)}")


if __name__ == "__main__":
    asyncio.run(main())
