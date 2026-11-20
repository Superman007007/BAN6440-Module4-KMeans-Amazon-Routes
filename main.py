import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


def load_json(filename):
    """Load a JSON file from the project data directory."""
    filepath = DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


def build_route_features(route_data, package_data):
    """Aggregate Amazon route and package data into route-level features."""

    records = []

    for route_id, route in route_data.items():

        stops = route.get("stops", {})
        packages_by_stop = package_data.get(route_id, {})

        num_stops = len(stops)
        num_packages = 0

        service_times = []
        package_volumes = []

        for stop_packages in packages_by_stop.values():

            for package in stop_packages.values():

                num_packages += 1

                # Planned service time
                service_time = package.get("planned_service_time_seconds")

                if isinstance(service_time, (int, float)):
                    service_times.append(service_time)

                # Package dimensions
                dimensions = package.get("dimensions", {})

                depth = dimensions.get("depth_cm")
                height = dimensions.get("height_cm")
                width = dimensions.get("width_cm")

                if all(
                    isinstance(value, (int, float))
                    for value in [depth, height, width]
                ):
                    volume = depth * height * width
                    package_volumes.append(volume)

        total_service_time = sum(service_times)

        avg_service_time = (
            total_service_time / len(service_times)
            if service_times else 0
        )

        total_package_volume = sum(package_volumes)

        executor_capacity = route.get("executor_capacity_cm3")

        capacity_utilization = (
            total_package_volume / executor_capacity
            if isinstance(executor_capacity, (int, float))
            and executor_capacity > 0
            else 0
        )

        packages_per_stop = (
            num_packages / num_stops
            if num_stops > 0
            else 0
        )

        records.append({
            "route_id": route_id,
            "num_stops": num_stops,
            "num_packages": num_packages,
            "packages_per_stop": packages_per_stop,
            "total_service_time": total_service_time,
            "avg_service_time": avg_service_time,
            "total_package_volume": total_package_volume,
            "capacity_utilization": capacity_utilization
        })

    return pd.DataFrame(records)


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading Amazon Last Mile dataset...")

    route_data = load_json("route_data.json")
    package_data = load_json("package_data.json")

    print("Building route-level features...")

    df = build_route_features(route_data, package_data)

    output_file = OUTPUT_DIR / "route_features.csv"
    df.to_csv(output_file, index=False)

    print("\nRoute-level dataset created successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nFirst five rows:")
    print(df.head())

    print("\nFeature summary:")
    print(df.describe())

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
