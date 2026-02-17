import numpy as np

def main():
    print("=== ML Foundations: Feature Scaling (Normalization) ===\n")

    # Why Normalize?
    # In ML, features often have different scales (e.g., Age: 20-60, Income: 20000-100000).
    # This confuses models (especially those using Gradient Descent or Distance metrics).
    # Normalization (Standardization) brings them to a similar scale (usually mean=0, std=1).
    
    # Example Dataset: [Age, Income]
    data = np.array([
        [25, 50000],
        [30, 80000],
        [45, 120000],
        [22, 35000],
        [50, 60000]
    ])
    
    print(f"Original Data:\n{data}\n")

    # 1. Calculate Mean and Standard Deviation per Feature (Column)
    # axis=0 means "collapse the rows" -> aggregate over columns.
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    print(f"Mean (Age, Income): {mean}")
    print(f"Std (Age, Income): {std}\n")

    # 2. Standardize (Z-score Normalization)
    # Formula: (x - mean) / std
    # Broadcasting handles the subtraction and division column-wise automatically!
    
    normalized_data = (data - mean) / std
    
    print(f"Normalized Data:\n{normalized_data}\n")
    
    # 3. Verification
    # After normalization, each feature should have mean ≈ 0 and std ≈ 1.
    new_mean = np.mean(normalized_data, axis=0)
    new_std = np.std(normalized_data, axis=0)
    
    # Using np.round to avoid tiny floating point errors in display
    print(f"New Mean: {np.round(new_mean, 2)}") # Should be [0. 0.]
    print(f"New Std:  {np.round(new_std, 2)}")  # Should be [1. 1.]

if __name__ == "__main__":
    main()
