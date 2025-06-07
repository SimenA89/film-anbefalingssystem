def sort_array(arr):
    """
    Sorts an array in ascending order.
    
    Args:
        arr (list): The array to be sorted.
    
    Returns:
        list: The sorted array.
    """
    return sorted(arr)

# Example usage
if __name__ == "__main__":
    sample_array = [5, 2, 9, 1, 5, 6]
    print("Original array:", sample_array)
    print("Sorted array:", sort_array(sample_array))


# This code defines a function `sort_array` that takes an array as input and returns the sorted array in ascending order using Python's built-in `sorted()` function. The example usage demonstrates how to use the function.
# The function is designed to be reusable and can handle any list of numbers.