from typing import NamedTuple

import numpy as np


class DCTBasis(NamedTuple):
    basis_functions: np.ndarray
    spatial_frequencies_components: np.ndarray
    spatial_frequencies_magnitude: np.ndarray
    multiplication_factor_matrix: np.ndarray
    multiplication_factor_scalar: float
    block_size: int


def get_dct_basis(block_size: int = 8) -> DCTBasis:
    """Generate the DCT basis variables for a given block size.

    Args:
        block_size (int, optional): The block size. Defaults to 8.

    Returns:
        DCTBasis: The DCT basis variables.
    """
    frequencies = np.arange(block_size)
    x = np.arange(block_size)
    y = np.arange(block_size)
    x, y = np.meshgrid(x, y, indexing="xy")
    basis_functions = np.zeros(
        (block_size, block_size, block_size, block_size), dtype=np.float32
    )
    spatial_frequencies = np.zeros((block_size, block_size, 2), dtype=np.int64)
    multiplication_factor_matrix = np.zeros((block_size, block_size), dtype=np.float32)
    for v in frequencies:
        for u in frequencies:
            # spatial frequencies
            spatial_frequencies[v, u] = (v, u)
            # basis functions
            x_ref_patch = np.cos(((2 * x + 1) * u * np.pi) / (2 * block_size))
            y_ref_patch = np.cos(((2 * y + 1) * v * np.pi) / (2 * block_size))
            basis_functions[v, u] = x_ref_patch * y_ref_patch
            # constants
            c_v = 1 / np.sqrt(2) if v == 0 else 1
            c_u = 1 / np.sqrt(2) if u == 0 else 1
            multiplication_factor_matrix[v, u] = c_u * c_v

    spatial_frequencies_magnitude = np.linalg.norm(spatial_frequencies, axis=2)
    multiplication_factor_scalar = 2 / block_size

    return DCTBasis(
        basis_functions=basis_functions,
        spatial_frequencies_components=spatial_frequencies,
        spatial_frequencies_magnitude=spatial_frequencies_magnitude,
        multiplication_factor_matrix=multiplication_factor_matrix,
        multiplication_factor_scalar=multiplication_factor_scalar,
        block_size=block_size,
    )
