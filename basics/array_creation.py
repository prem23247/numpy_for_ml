import numpy as np

def main():
    print("=== NumPy Array Creation Basics ===\n")

    # 1. Scalar (0-D array)
    # A scalar is just a single value.
    scalar = np.array(42)
    print(f"Scalar:\n{scalar}")
    print(f"Shape: {scalar.shape}, Dimension: {scalar.ndim}\n")

    # 2. 1-Dimensional Array (Vector)
    # Think of this as a list of numbers. In ML, this often represents a single feature vector
    # or a sequence of values (like time series data).
    vector = np.array([1, 2, 3, 4, 5])
    print(f"1D Array (Vector):\n{vector}")
    print(f"Shape: {vector.shape}")  # (5,) -> 5 elements
    print(f"Dimension: {vector.ndim}") # 1
    print(f"Data Type: {vector.dtype}\n") # int32 or int64 depending on OS

    # 3. 2-Dimensional Array (Matrix)
    # Think of this as a table or a spreadsheet. In ML, this is the most common format
    # for datasets: rows are samples, columns are features.
    matrix = np.array([
        [1.5, 2.5, 3.5],
        [4.5, 5.5, 6.5]
    ])
    print(f"2D Array (Matrix):\n{matrix}")
    print(f"Shape: {matrix.shape}")  # (2, 3) -> 2 rows, 3 columns
    print(f"Dimension: {matrix.ndim}") # 2
    print(f"Data Type: {matrix.dtype}\n") # float64

    # 4. Creating arrays with zeros and ones
    # Initializing weights (often small random numbers or specialized schemes) or biases (often zeros)
    # is a crucial step in preparing Neural Networks.
    zeros_matrix = np.zeros((3, 3))
    print(f"Zeros Matrix (3x3):\n{zeros_matrix}")
    
    ones_vector = np.ones((4,))
    print(f"Ones Vector (length 4):\n{ones_vector}\n")

    # 5. Specifying Data Types
    # Memory management is key in DL. Using simpler types like 'int8' or 'float32'
    # instead of default 'int64'/'float64' can save massive amounts of RAM.
    small_int_arr = np.array([10, 20, 30], dtype='int8')
    print(f"Small Int Array:\n{small_int_arr}")
    print(f"Data Type: {small_int_arr.dtype}\n")

if __name__ == "__main__":
    main()
