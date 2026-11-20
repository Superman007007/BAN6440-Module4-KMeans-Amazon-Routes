# BAN6440 Module 4 - K-Means Python Application

## Project Overview

This project applies K-Means clustering to the 2021 Amazon Last Mile Routing Research Challenge Dataset from the Registry of Open Data on AWS.

The analysis uses historical last-mile delivery route and package information to identify operationally meaningful groups of delivery routes.

## Dataset

Source:
2021 Amazon Last Mile Routing Research Challenge Dataset

Registry of Open Data on AWS:
https://registry.opendata.aws/amazon-last-mile-challenges/

The training dataset contains 6,112 historical delivery routes.

The application uses:

- route_data.json
- package_data.json

The large raw AWS dataset files are not stored in this repository.

## Engineered Features

Each delivery route is represented using seven numerical features:

1. Number of stops
2. Number of packages
3. Packages per stop
4. Total planned service time
5. Average planned service time
6. Total package volume
7. Vehicle capacity utilization

The features are standardized using StandardScaler before K-Means clustering.

## Model Selection

K values from 2 through 8 were evaluated using:

- Within-Cluster Sum of Squares (WCSS / inertia)
- Silhouette Score

K = 3 was selected because it provided a useful balance between cluster separation and operational interpretability.

Final Silhouette Score: 0.2720

## Results

The final model identified three operational route profiles:

- Cluster 0: Moderate-load, lower-utilization routes
- Cluster 1: Large, high-volume, high-utilization routes
- Cluster 2: Compact, package-dense, service-intensive routes

Cluster sizes:

- Cluster 0: 2,298 routes
- Cluster 1: 2,754 routes
- Cluster 2: 1,060 routes

PCA was used only to visualize the seven-dimensional clustering results in two dimensions.

The first two principal components represent approximately 76.58% of total variance.

## Project Files

- main.py - Loads AWS JSON data and creates route-level features
- analyze_clusters.py - Evaluates candidate K values
- final_clustering.py - Runs final K-Means clustering and PCA visualization
- test_kmeans.py - Unit tests
- inspect_aws.py - Downloads/inspects AWS dataset resources
- output/ - Generated analytical results and charts

## Unit Testing

Unit tests are implemented with pytest.

Run:

python -m pytest -v

Five tests validate:

- Route-level feature engineering
- Empty-route handling
- Feature standardization
- Three-cluster K-Means behavior
- Cluster-profile route reconciliation

All five tests passed successfully.

## Python Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pytest
- boto3

## Author

Kyaw Swar

BAN6440 Applied Machine Learning for Business Analytics