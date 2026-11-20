import numpy as np
import pandas as pd
import pytest

from main import build_route_features
from final_clustering import (
    FEATURES,
    FINAL_K,
    prepare_features,
    fit_kmeans,
    create_cluster_profile,
)


@pytest.fixture
def sample_route_data():
    """Small route dataset for testing feature engineering."""
    return {
        "Route_1": {
            "executor_capacity_cm3": 100000.0,
            "stops": {
                "A": {
                    "lat": 1.0,
                    "lng": 2.0,
                    "type": "Dropoff"
                },
                "B": {
                    "lat": 1.1,
                    "lng": 2.1,
                    "type": "Dropoff"
                }
            }
        }
    }


@pytest.fixture
def sample_package_data():
    """Package records corresponding to the test route."""
    return {
        "Route_1": {
            "A": {
                "Package_1": {
                    "planned_service_time_seconds": 30.0,
                    "dimensions": {
                        "depth_cm": 10.0,
                        "height_cm": 10.0,
                        "width_cm": 10.0
                    }
                }
            },
            "B": {
                "Package_2": {
                    "planned_service_time_seconds": 60.0,
                    "dimensions": {
                        "depth_cm": 20.0,
                        "height_cm": 10.0,
                        "width_cm": 10.0
                    }
                }
            }
        }
    }


def test_build_route_features(
    sample_route_data,
    sample_package_data
):
    """Test that route-level features are calculated correctly."""

    df = build_route_features(
        sample_route_data,
        sample_package_data
    )

    assert len(df) == 1

    row = df.iloc[0]

    assert row["route_id"] == "Route_1"
    assert row["num_stops"] == 2
    assert row["num_packages"] == 2
    assert row["packages_per_stop"] == pytest.approx(1.0)

    assert row["total_service_time"] == pytest.approx(90.0)
    assert row["avg_service_time"] == pytest.approx(45.0)

    # Package volumes:
    # Package 1 = 10 × 10 × 10 = 1,000 cm3
    # Package 2 = 20 × 10 × 10 = 2,000 cm3
    assert row["total_package_volume"] == pytest.approx(3000.0)

    # Capacity utilization = 3,000 / 100,000
    assert row["capacity_utilization"] == pytest.approx(0.03)


def test_empty_route_is_handled():
    """Test graceful handling of a route with no packages or stops."""

    route_data = {
        "Route_Empty": {
            "executor_capacity_cm3": 50000.0,
            "stops": {}
        }
    }

    df = build_route_features(
        route_data,
        {}
    )

    row = df.iloc[0]

    assert row["num_stops"] == 0
    assert row["num_packages"] == 0
    assert row["packages_per_stop"] == 0
    assert row["total_service_time"] == 0
    assert row["total_package_volume"] == 0
    assert row["capacity_utilization"] == 0


def test_standard_scaling():
    """Test that StandardScaler produces near-zero feature means."""

    df = pd.DataFrame({
        feature: np.arange(1, 11, dtype=float)
        for feature in FEATURES
    })

    _, scaled_data = prepare_features(df)

    assert scaled_data.shape == (10, len(FEATURES))

    assert np.allclose(
        scaled_data.mean(axis=0),
        0.0,
        atol=1e-7
    )


def test_kmeans_creates_three_clusters():
    """Test that the configured K-Means model returns three groups."""

    rng = np.random.default_rng(42)

    cluster_a = rng.normal(
        loc=-5,
        scale=0.3,
        size=(20, len(FEATURES))
    )

    cluster_b = rng.normal(
        loc=0,
        scale=0.3,
        size=(20, len(FEATURES))
    )

    cluster_c = rng.normal(
        loc=5,
        scale=0.3,
        size=(20, len(FEATURES))
    )

    test_data = np.vstack([
        cluster_a,
        cluster_b,
        cluster_c
    ])

    model, labels = fit_kmeans(test_data)

    assert model.n_clusters == FINAL_K
    assert len(labels) == 60
    assert len(set(labels)) == 3


def test_cluster_profile_route_counts():
    """Test that cluster profiles account for every route."""

    df = pd.DataFrame({
        "cluster": [0, 0, 1, 1, 2, 2],
        **{
            feature: [1, 2, 3, 4, 5, 6]
            for feature in FEATURES
        }
    })

    profile = create_cluster_profile(df)

    assert profile["route_count"].sum() == len(df)
    assert set(profile.index) == {0, 1, 2}