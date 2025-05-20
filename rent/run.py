import numpy as np
import time
from sklearn.neighbors import BallTree
from tqdm import tqdm
import random

np.random.seed(42)

# Simulate POIs
num_pois = 100_000
pois_np = np.array(
    [
        [
            random.uniform(-90, 90),
            random.uniform(-180, 180),
            random.choice(["school", "hospital", "restaurant"]),
        ]
        for _ in range(num_pois)
    ]
)

# Simulate Apartments
num_apts = 50_000
apartments = [
    {
        "id": i,
        "latitude": random.uniform(-90, 90),
        "longitude": random.uniform(-180, 180),
    }
    for i in range(num_apts)
]

radius_meters = 1_000.0
radius_rad = radius_meters / 6_371_000


# === Approach 1: tqdm-friendly (1-by-1) ===
def iterative_version():
    poi_coords_rad = np.radians(pois_np[:, :2].astype(float))
    amenities = pois_np[:, 2]
    tree = BallTree(poi_coords_rad, metric="haversine")

    results = []
    for apt in tqdm(apartments, desc="Iterative"):
        apt_id = apt["id"]
        apt_coord_rad = np.radians([[apt["latitude"], apt["longitude"]]])
        [poi_idxs] = tree.query_radius(apt_coord_rad, r=radius_rad)

        nearby_counts = {}
        for idx in poi_idxs:
            tag = str(amenities[idx])
            nearby_counts[tag] = nearby_counts.get(tag, 0) + 1

        results.extend(
            {"id": apt_id, "amenity": tag, "count": count}
            for tag, count in nearby_counts.items()
        )
    return results


# === Approach 2: Vectorized ===
def vectorized_version():
    poi_coords_rad = np.radians(pois_np[:, :2].astype(float))
    amenities = pois_np[:, 2]
    tree = BallTree(poi_coords_rad, metric="haversine")

    apt_coords_rad = np.radians(
        np.array([[apt["latitude"], apt["longitude"]] for apt in apartments])
    )

    neighbors = tree.query_radius(apt_coords_rad, r=radius_rad)

    results = []
    for apt, poi_idxs in zip(apartments, neighbors):
        apt_id = apt["id"]
        nearby_counts = {}
        for idx in poi_idxs:
            tag = str(amenities[idx])
            nearby_counts[tag] = nearby_counts.get(tag, 0) + 1

        results.extend(
            {"id": apt_id, "amenity": tag, "count": count}
            for tag, count in nearby_counts.items()
        )
    return results


def brute_force_version():
    results = []
    for apt in tqdm(apartments, desc="Brute Force"):
        apt_id = apt["id"]
        apt_coord_rad = np.radians([[apt["latitude"], apt["longitude"]]])
        nearby_counts = {}
        for poi in pois_np:
            poi_coord_rad = np.radians(poi[:2].astype(float))
            distance = np.linalg.norm(apt_coord_rad - poi_coord_rad)
            if distance <= radius_rad:
                tag = str(poi[2])
                nearby_counts[tag] = nearby_counts.get(tag, 0) + 1

        results.extend(
            {"id": apt_id, "amenity": tag, "count": count}
            for tag, count in nearby_counts.items()
        )
    return results


# === Run Benchmark ===
start = time.time()
_ = iterative_version()
print(f"Iterative time: {time.time() - start:.2f}s")

start = time.time()
_ = vectorized_version()
print(f"Vectorized time: {time.time() - start:.2f}s")


start = time.time()
_ = brute_force_version()
print(f"Brute Force time: {time.time() - start:.2f}s")
