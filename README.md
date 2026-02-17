# numpy-for-ml

## Project Overview
**numpy-for-ml** is a professional learning project designed to strip away the complexity of high-level Machine Learning libraries and focus on the fundamental building block of Data Science in Python: **NumPy**.

This repository contains clear, runnable, and heavily commented examples demonstrating how core ML concepts (like normalization, distance metrics, and matrix multiplication) are implemented using pure NumPy.

## Topics Covered
The project is organized into four key modules:

1.  **Basics**
    *   `array_creation.py`: Understanding scalars, vectors, matrices, and data types.
    *   `indexing_slicing.py`: Extracting data, slicing rows/columns, and boolean masking.

2.  **Operations**
    *   `arithmetic_ops.py`: Element-wise comparisons and the power of vectorized computation vs. loops.
    *   `broadcasting.py`: How NumPy handles operations between arrays of different shapes.

3.  **Linear Algebra**
    *   `dot_product.py`: The engine of neural networks (weighted sums).
    *   `matrix_multiplication.py`: Propagating inputs through layers using `np.dot` and `@`.

4.  **ML Foundations**
    *   `normalization.py`: Implementing Feature Scaling (Z-score normalization) from scratch.
    *   `distance_metrics.py`: Calculating Euclidean distance for algorithms like K-Nearest Neighbors (KNN).

## Why NumPy is Important for ML
*   **Performance**: NumPy uses optimized C-level primitives, making it orders of magnitude faster than Python lists.
*   **Foundation**: Libraries like TensorFlow, PyTorch, Scikit-Learn, and Pandas are all built on top of NumPy.
*   **Vectorization**: It allows for mathematical operations on entire datasets without writing slow loops, which is essential for training models on massive data.

## How to Run
Ensure you have Python and NumPy installed.

```bash
pip install numpy
```

Navigate to the project directory and run any script:

```bash
# Example: Run the broadcasting demo
python operations/broadcasting.py

# Example: Run the normalization demo
python ml_foundations/normalization.py
```

## Author
**Prem Patel**
