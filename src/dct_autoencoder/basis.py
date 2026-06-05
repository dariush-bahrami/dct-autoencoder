import math
from typing import NamedTuple

import torch


class DCTBasis(NamedTuple):
    basis_functions: torch.Tensor
    spatial_frequencies_components: torch.Tensor
    spatial_frequencies_magnitude: torch.Tensor
    multiplication_factor_matrix: torch.Tensor
    multiplication_factor_scalar: float
    block_size: int


def get_dct_basis(block_size: int = 8, device: str = "cpu") -> DCTBasis:
    """Generate the DCT basis variables for a given block size.

    Args:
        block_size (int, optional): The block size. Defaults to 8.
        device (str, optional): The device to create tensors on ("cpu", "cuda", "mps"). Defaults to "cpu".

    Returns:
        DCTBasis: The DCT basis variables.
    """
    # Create 1D coordinate tensors reshaped to distinct dimensions for 4D broadcasting
    v = torch.arange(block_size, dtype=torch.float32, device=device).reshape(
        block_size, 1, 1, 1
    )
    u = torch.arange(block_size, dtype=torch.float32, device=device).reshape(
        1, block_size, 1, 1
    )
    y = torch.arange(block_size, dtype=torch.float32, device=device).reshape(
        1, 1, block_size, 1
    )
    x = torch.arange(block_size, dtype=torch.float32, device=device).reshape(
        1, 1, 1, block_size
    )

    # Compute basis functions using 4D broadcasting (Shape: [block_size, block_size, block_size, block_size])
    x_ref_patch = torch.cos(((2 * x + 1) * u * math.pi) / (2 * block_size))
    y_ref_patch = torch.cos(((2 * y + 1) * v * math.pi) / (2 * block_size))
    basis_functions = x_ref_patch * y_ref_patch

    # Generate spatial frequencies grid (Shape: [block_size, block_size, 2])
    v_grid, u_grid = torch.meshgrid(
        torch.arange(block_size, dtype=torch.long, device=device),
        torch.arange(block_size, dtype=torch.long, device=device),
        indexing="ij",
    )
    spatial_frequencies = torch.stack([v_grid, u_grid], dim=-1)

    # Compute Euclidean norm along the last dimension
    spatial_frequencies_magnitude = torch.linalg.norm(
        spatial_frequencies.to(torch.float32), dim=-1
    )

    # Compute multiplication factor matrix via outer product
    c = torch.ones(block_size, dtype=torch.float32, device=device)
    c[0] = 1 / math.sqrt(2)
    multiplication_factor_matrix = c.reshape(-1, 1) * c.reshape(1, -1)

    multiplication_factor_scalar = 2.0 / block_size

    return DCTBasis(
        basis_functions=basis_functions,
        spatial_frequencies_components=spatial_frequencies,
        spatial_frequencies_magnitude=spatial_frequencies_magnitude,
        multiplication_factor_matrix=multiplication_factor_matrix,
        multiplication_factor_scalar=multiplication_factor_scalar,
        block_size=block_size,
    )
