import numpy as np

def main():
    print("=== NumPy Indexing and Slicing ===\n")

    # Create a 2D matrix (3 rows, 4 columns)
    # This could represent a dataset with 3 samples and 4 features each.
    data = np.array([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120]
    ])
    print(f"Original Data Matrix:\n{data}\n")

    # 1. Accessing specific elements
    # Syntax: [row_index, col_index]
    element = data[0, 2] # Row 0, Column 2 (value 30)
    print(f"Element at [0, 2]: {element}\n")

    # 2. Slicing Rows and Columns
    # In ML, we often need to separate our 'X' (features) from our 'y' (labels).
    # If the last column was the label, we'd want all rows, but only the last column.
    
    # Get the first row
    first_row = data[0, :] 
    print(f"First Row (Sample 1): {first_row}")

    # Get the first column
    first_col = data[:, 0]
    print(f"First Column (Feature 1): {first_col}\n")

    # Slicing a sub-matrix (e.g., first 2 rows, first 3 columns)
    sub_matrix = data[:2, :3]
    print(f"Sub-matrix (First 2 rows, First 3 cols):\n{sub_matrix}\n")

    # 3. Boolean Indexing (Filtering)
    # This is powerful for data cleaning. E.g., remove all outliers or find values matching a criteria.
    # Let's find all values greater than 50.
    mask = data > 50
    print(f"Mask (Values > 50):\n{mask}\n")
    
    filtered_data = data[mask]
    print(f"Filtered Data (Values > 50):\n{filtered_data}")
    # Note: Boolean indexing returns a 1D array of the elements that matched.

if __name__ == "__main__":
    main()
