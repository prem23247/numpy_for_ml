import numpy as np

def main():
    print("=== ML Foundations: Distance Metrics (Euclidean) ===\n")

    # Distance metrics are the core of many algorithms like K-Nearest Neighbors (KNN),
    # K-Means Clustering, and Recommendation Systems.
    # They measure "similarity": smaller distance = more similar.

    # Example: Two data points in 2D space
    # Point A (Person A): [Age, Income]
    point_a = np.array([30, 50000])
    
    # Point B (Person B): [Age, Income]
    point_b = np.array([35, 55000])
    
    print(f"Point A: {point_a}")
    print(f"Point B: {point_b}\n")

    # 1. Euclidean Distance
    # Formula: sqrt(sum((a - b)^2))
    
    # Step-by-step
    squared_diff = (point_a - point_b) ** 2
    sum_squared_diff = np.sum(squared_diff)
    distance = np.sqrt(sum_squared_diff)
    
    print(f"Squared Differences: {squared_diff}")
    print(f"Sum of Squared Differences: {sum_squared_diff}")
    print(f"Euclidean Distance: {distance}\n")

    # 2. Using np.linalg.norm
    # NumPy provides a linear algebra module for common norms.
    dist_norm = np.linalg.norm(point_a - point_b)
    print(f"Distance using np.linalg.norm: {dist_norm}\n")

    # 3. KNN Concept
    # Let's find which point from a dataset is closest to a "Query Point".
    
    dataset = np.array([
        [25, 40000],
        [50, 90000],
        [32, 52000]  # This looks closest to Point A (30, 50000)
    ])
    query = np.array([30, 50000])
    
    print(f"Dataset:\n{dataset}")
    print(f"Query Point: {query}\n")

    # Calculate distance from query to ALL points in dataset at once (Broadcasting!)
    # (3, 2) - (2,) -> (3, 2)
    differences = dataset - query
    
    # Calculate norm along axis 1 (rows)
    # We want a distance for *each* sample.
    distances = np.linalg.norm(differences, axis=1)
    
    print(f"Distances to Query Point: {distances}")
    
    # Find the index of the minimum distance
    nearest_neighbor_idx = np.argmin(distances)
    print(f"Nearest Neighbor Index: {nearest_neighbor_idx}")
    print(f"Nearest Neighbor Data: {dataset[nearest_neighbor_idx]}")

if __name__ == "__main__":
    main()
