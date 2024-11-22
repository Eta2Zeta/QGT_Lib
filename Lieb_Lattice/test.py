import numpy as np
def spiral_indices(n):
    indices = []
    left, right = 0, n - 1
    top, bottom = 0, n - 1
    spiral_indices_matrix = np.empty((n, n), dtype=object)


    while left <= right and top <= bottom:
        for i in range(left, right + 1):
            indices.append((top, i))
        top += 1

        for i in range(top, bottom + 1):
            indices.append((i, right))
        right -= 1

        for i in range(right, left - 1, -1):
            indices.append((bottom, i))
        bottom -= 1

        for i in range(bottom, top - 1, -1):
            indices.append((i, left))
        left += 1
    
    i_index = 0
    for i in range(n): 
        for j in range (n):
            spiral_indices_matrix[i,j] = indices[i_index]
            i_index += 1

    return spiral_indices_matrix

# Example usage
n = 5  # Size of the 2D space
spiral_order_matrix = spiral_indices(n)
for row in spiral_order_matrix:
    print(row)
