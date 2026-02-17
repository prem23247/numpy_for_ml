import numpy as np
import time

def main():
    print("=== NumPy Arithmetic Operations ===\n")

    # 1. Element-wise Operations
    # In ML, we often perform operations on entire datasets at once (batch processing).
    
    # Example: Normalize pixel values (0-255) to (0-1)
    pixels = np.array([0, 127.5, 255])
    print(f"Original Pixels: {pixels}")
    
    # Division
    normalized_pixels = pixels / 255.0
    print(f"Normalized Pixels (/ 255.0): {normalized_pixels}\n")

    # Other basic ops
    array_a = np.array([1, 2, 3])
    array_b = np.array([4, 5, 6])
    
    print(f"A: {array_a}, B: {array_b}")
    print(f"A + B: {array_a + array_b}")
    print(f"A - B: {array_a - array_b}")
    print(f"A * B: {array_a * array_b}\n") # Element-wise multiplication, NOT matrix multiplication!

    # 2. Vectorized Computation vs Loops
    # This is the "secret sauce" of NumPy (and by extension, libraries like PyTorch/TensorFlow).
    # Operations are pushed to C-level primitives, making them orders of magnitude faster.
    
    size = 1_000_000
    large_array = np.arange(size)
    large_list = list(range(size))

    print(f"Timing operation on {size} elements...")

    # Python List Loop
    start_time = time.time()
    list_result = [x * 2 for x in large_list]
    end_time = time.time()
    list_duration = end_time - start_time
    print(f"Python List Loop: {list_duration:.6f} seconds")

    # NumPy Vectorized Operation
    start_time = time.time()
    numpy_result = large_array * 2
    end_time = time.time()
    numpy_duration = end_time - start_time
    print(f"NumPy Vectorized: {numpy_duration:.6f} seconds")

    speedup = list_duration / numpy_duration
    print(f"NumPy Speedup: {speedup:.2f}x faster!\n")
    print("In Deep Learning, where we process millions of parameters, this speedup is non-negotiable.")

if __name__ == "__main__":
    main()
