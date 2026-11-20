from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


OUTPUT_DIR = Path("output")
DATA_FILE = OUTPUT_DIR / "route_features.csv"

FEATURES = [
    "num_stops",
    "num_packages",
    "packages_per_stop",
    "total_service_time",
    "avg_service_time",
    "total_package_volume",
    "capacity_utilization",
]


def load_and_validate_data():
    """Load engineered route features and validate required fields."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    missing_columns = [
        column for column in FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print(f"Routes loaded: {len(df)}")
    print(f"Missing values: {df[FEATURES].isna().sum().sum()}")
    print(f"Duplicate route IDs: {df['route_id'].duplicated().sum()}")

    return df


def scale_features(df):
    """Standardize features so each contributes comparably to K-Means."""

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        df[FEATURES]
    )

    return scaled_data


def evaluate_k_values(scaled_data):
    """Compare K values using inertia and silhouette score."""

    results = []

    print("\nK-Means model comparison")
    print("-" * 45)

    for k in range(2, 9):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(scaled_data)

        inertia = model.inertia_
        silhouette = silhouette_score(
            scaled_data,
            labels
        )

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette_score": silhouette
        })

        print(
            f"K={k}: "
            f"Inertia={inertia:.2f}, "
            f"Silhouette={silhouette:.4f}"
        )

    return pd.DataFrame(results)


def save_charts(results_df):
    """Create Elbow and Silhouette charts."""

    # Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(
        results_df["k"],
        results_df["inertia"],
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia / WCSS")
    plt.title("Elbow Method for Amazon Delivery Routes")
    plt.xticks(results_df["k"])
    plt.grid(alpha=0.3)
    plt.tight_layout()

    elbow_file = OUTPUT_DIR / "elbow_method.png"
    plt.savefig(elbow_file, dpi=300)
    plt.close()

    # Silhouette Score
    plt.figure(figsize=(8, 5))
    plt.plot(
        results_df["k"],
        results_df["silhouette_score"],
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis for Amazon Delivery Routes")
    plt.xticks(results_df["k"])
    plt.grid(alpha=0.3)
    plt.tight_layout()

    silhouette_file = OUTPUT_DIR / "silhouette_scores.png"
    plt.savefig(silhouette_file, dpi=300)
    plt.close()

    print(f"\nSaved: {elbow_file}")
    print(f"Saved: {silhouette_file}")


def main():

    df = load_and_validate_data()

    scaled_data = scale_features(df)

    results_df = evaluate_k_values(
        scaled_data
    )

    results_file = OUTPUT_DIR / "k_selection_results.csv"
    results_df.to_csv(
        results_file,
        index=False
    )

    save_charts(results_df)

    best_row = results_df.loc[
        results_df["silhouette_score"].idxmax()
    ]

    print("\nHighest Silhouette Score:")
    print(
        f"K={int(best_row['k'])}, "
        f"Score={best_row['silhouette_score']:.4f}"
    )

    print(f"\nSaved: {results_file}")


if __name__ == "__main__":
    main()