import numpy as np

def main():
    print("=== Dot Product (Vector-Vector Multiplication) ===\n")

    # In ML, the dot product is fundamental.
    # It calculates the similarity between two vectors, or the weighted sum of inputs.
    # Formula: a . b = sum(a[i] * b[i])
    
    # Example: A simple neuron with 3 inputs.
    inputs = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.2, 0.8, -0.5])
    bias = 2.0

    print(f"Inputs: {inputs}")
    print(f"Weights: {weights}")
    print(f"Bias: {bias}\n")

    # 1. Manual Calculation (for understanding)
    # Output = (x1*w1) + (x2*w2) + (x3*w3) + b
    manual_dot = (inputs[0]*weights[0]) + (inputs[1]*weights[1]) + (inputs[2]*weights[2])
    print(f"Manual Dot Product: {manual_dot}")
    print(f"Manual Output (+ Bias): {manual_dot + bias}\n")

    # 2. NumPy Dot Product
    # Using np.dot() is cleaner and faster.
    numpy_dot = np.dot(inputs, weights)
    print(f"NumPy Dot Product (np.dot): {numpy_dot}")
    print(f"NumPy Output (+ Bias): {numpy_dot + bias}\n")

    # Geometric Interpretation
    # If vectors are normalized (length 1), the dot product is the cosine of the angle between them.
    # - 1.0: Vectors point in same direction (similar)
    # - 0.0: Vectors are orthogonal (unrelated)
    # - -1.0: Vectors point in opposite directions (dissimilar)
    
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    print(f"Dot product of orthogonal vectors {v1} and {v2}: {np.dot(v1, v2)}")

if __name__ == "__main__":
    main()
