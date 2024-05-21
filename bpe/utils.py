from icecream import ic
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from icecream import ic as logger
from contextlib import contextmanager


@contextmanager
def open_file(file_name, mode):
    """
    Context manager for opening and closing a file.

    Parameters:
    file_name (str): The name of the file to be opened.
    mode (str): The mode in which the file is to be opened.

    Yields:
    file: The file object.
    """
    try:
        file = open(file_name, mode)
        yield file
    except Exception as e:
        logger(f"An error occurred: {e}")
        raise
    finally:
        file.close()


# Create logs directory if it doesn't exist
if not os.path.exists(f'{os.getcwd()}/logs'):
    os.makedirs(f'{os.getcwd()}/logs')

# Define a file to log IceCream output
log_file_path = os.path.join(
    f'{os.getcwd()}/logs', 'logs.log')


def log_to_file(message):
    with open_file(log_file_path, 'a') as f:
        f.write(message + '\n')


# Replace logging configuration with IceCream configuration
ic.configureOutput(prefix=' - ', outputFunction=log_to_file)


def parallel_concat(arrs):
    lens = [len(arr) for arr in arrs]
    start_idcs = [0] + [sum(lens[:i]) for i in range(1, len(lens))]
    total_len = sum(lens)
    result = [None] * total_len

    def copy_elements(arr, start_idx):
        result[start_idx:start_idx + len(arr)] = arr

    with ThreadPoolExecutor() as executor:
        executor.map(copy_elements, arrs, start_idcs)

    return result


# Example usage
if __name__ == "__main__":
    arrs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    result = parallel_concat(arrs)
    print(result)
