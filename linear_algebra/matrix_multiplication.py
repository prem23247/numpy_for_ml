import numpy as np

def main():
    print("=== Matrix Multiplication ===\n")

    # In Neural Networks, layers are often represented as matrices.
    # Matrix multiplication allows us to propagate inputs through these layers efficiently.
    # Rule: (M x N) . (N x P) = (M x P)
    # The inner dimensions (N) must match!

    # Example: A batch of 3 samples, each with 4 features.
    # Shape: (3, 4)
    inputs = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])
    
    # A Hidden Layer with 4 inputs (matching features) and 2 neurons (outputs).
    # Shape: (4, 2)
    weights = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8]
    ])

    print(f"Inputs Shape: {inputs.shape}")   # (3, 4)
    print(f"Weights Shape: {weights.shape}") # (4, 2)
    
    # 1. Using np.dot()
    # The standard way to multiply matrices.
    output_dot = np.dot(inputs, weights)
    print(f"\nOutput Shape (using np.dot): {output_dot.shape}") # Should be (3, 2)
    print(f"Output:\n{output_dot}\n")

    # 2. Using the @ operator (Python 3.5+)
    # This is syntactic sugar for matrix multiplication and is preferred for readability.
    output_at = inputs @ weights
    print(f"Output Shape (using @ operator): {output_at.shape}")
    print(f"Output:\n{output_at}\n")

    # Common Pitfall: Element-wise multiplication (*)
    # Ideally, this throws an error if shapes don't align for broadcasting.
    try:
        wrong_op = inputs * weights
        print("Element-wise multiplication worked (unexpectedly)!")
    except ValueError as e:
        print("Element-wise multiplication failed as expected (shapes incompatible for broadcasting).")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
