from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

FINAL_K = 3
RANDOM_STATE = 42


def load_data():
    """Load the engineered Amazon route-level dataset."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Route feature file not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = ["route_id"] + FEATURES
    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


def prepare_features(df):
    """Standardize numeric features before distance-based clustering."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[FEATURES])

    return scaler, scaled_data


def fit_kmeans(scaled_data):
    """Fit the final three-cluster K-Means model."""
    model = KMeans(
        n_clusters=FINAL_K,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = model.fit_predict(scaled_data)

    return model, labels


def create_cluster_profile(df):
    """Calculate mean operational characteristics for each cluster."""
    profile = (
        df.groupby("cluster")[FEATURES]
        .mean()
        .round(3)
    )

    profile["route_count"] = (
        df.groupby("cluster")
        .size()
    )

    return profile


def create_pca_visualization(scaled_data, labels):
    """
    Reduce standardized features to two PCA components
    for visualization only.
    """
    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE
    )

    components = pca.fit_transform(scaled_data)

    plot_df = pd.DataFrame({
        "PC1": components[:, 0],
        "PC2": components[:, 1],
        "cluster": labels
    })

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette="Set1",
        s=35,
        alpha=0.65
    )

    plt.title(
        "K-Means Clusters of Amazon Last-Mile Delivery Routes"
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")

    plt.tight_layout()

    output_file = OUTPUT_DIR / "route_clusters_pca.png"

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    return pca, output_file


def main():

    print("Loading route features...")
    df = load_data()

    print("Standardizing features...")
    scaler, scaled_data = prepare_features(df)

    print(f"Fitting K-Means with K={FINAL_K}...")
    model, labels = fit_kmeans(scaled_data)

    df["cluster"] = labels

    silhouette = silhouette_score(
        scaled_data,
        labels
    )

    print(
        f"\nFinal Silhouette Score: "
        f"{silhouette:.4f}"
    )

    print("\nCluster sizes:")
    print(
        df["cluster"]
        .value_counts()
        .sort_index()
    )

    profile = create_cluster_profile(df)

    print("\nCluster operational profiles:")
    print(profile.to_string())

    clustered_file = (
        OUTPUT_DIR /
        "clustered_routes.csv"
    )

    profile_file = (
        OUTPUT_DIR /
        "cluster_profiles.csv"
    )

    df.to_csv(
        clustered_file,
        index=False
    )

    profile.to_csv(
        profile_file
    )

    pca, chart_file = create_pca_visualization(
        scaled_data,
        labels
    )

    print(
        "\nPCA variance represented by visualization:"
    )

    print(
        f"PC1: "
        f"{pca.explained_variance_ratio_[0]:.2%}"
    )

    print(
        f"PC2: "
        f"{pca.explained_variance_ratio_[1]:.2%}"
    )

    print(f"\nSaved: {clustered_file}")
    print(f"Saved: {profile_file}")
    print(f"Saved: {chart_file}")

    print(
        "\nK-Means clustering completed successfully."
    )


if __name__ == "__main__":
    main()