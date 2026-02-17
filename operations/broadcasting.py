import numpy as np

def main():
    print("=== NumPy Broadcasting ===\n")

    # What is Broadcasting?
    # It's NumPy's mechanism for working with arrays of different shapes during arithmetic operations.
    # The smaller array is "broadcast" across the larger array so that they have compatible shapes.
    
    # Case 1: Scalar and Array
    # We already saw this: array + 5
    # The scalar 5 is "stretched" to match the shape of the array.
    arr = np.array([1, 2, 3])
    print(f"Array: {arr}")
    print(f"Array + 5: {arr + 5}\n")

    # Case 2: Array and Matrix (Adding a vector to each row of a matrix)
    # Common ML use-case: Adding a bias term vector to a batch of inputs.
    
    # Input batch (3 samples, 2 features)
    inputs = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    
    # Bias vector (1 bias for each feature)
    biases = np.array([10.0, 20.0])
    
    print(f"Inputs (Shape {inputs.shape}):\n{inputs}")
    print(f"Biases (Shape {biases.shape}):\n{biases}\n")
    
    # Broadcasting happens here!
    # The 'biases' vector (shape (2,)) is virtually treated as shape (1, 2)
    # and then replicated 3 times to match 'inputs' shape (3, 2).
    
    output = inputs + biases
    print(f"Output (Inputs + Biases):\n{output}\n")
    
    # Case 3: Incompatible Shapes
    # Broadcasting rules are strict. Dimensions must be compatible.
    # Two dimensions are compatible when:
    # 1. They are equal, or
    # 2. One of them is 1.
    
    try:
        incompatible_arr = np.array([1, 2, 3]) # Shape (3,)
        result = inputs + incompatible_arr
    except ValueError as e:
        print("Broadcasting Error Example:")
        print(f"Cannot broadcast shapes {inputs.shape} and {incompatible_arr.shape}")
        print(f"Error message: {e}")

if __name__ == "__main__":
    main()
