import json
from pathlib import Path

DATA_DIR = Path("data")


def inspect_json(filename):
    path = DATA_DIR / filename

    print("\n" + "=" * 70)
    print(f"FILE: {filename}")
    print(f"SIZE: {path.stat().st_size / (1024 * 1024):.2f} MB")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"TOP-LEVEL TYPE: {type(data).__name__}")

    if isinstance(data, dict):
        print(f"NUMBER OF TOP-LEVEL RECORDS: {len(data)}")

        first_key = next(iter(data))
        first_value = data[first_key]

        print(f"\nFIRST KEY: {first_key}")
        print("\nFIRST RECORD:")

        if isinstance(first_value, dict):
            for key, value in first_value.items():
                if isinstance(value, dict):
                    print(f"  {key}: <dictionary with {len(value)} entries>")
                    sample_key = next(iter(value), None)

                    if sample_key:
                        print(f"    Example key: {sample_key}")
                        print(f"    Example value: {value[sample_key]}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(first_value)


inspect_json("route_data.json")
inspect_json("package_data.json")