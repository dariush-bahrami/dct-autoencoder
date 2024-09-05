import matplotlib.pyplot as plt
import numpy as np

from .basis import DCTBasis


def visualize_dct_basis_functions(
    dct_constants: DCTBasis,
    figsize: int = 8,
    fig_facecolor: str = "#fb6a2c",
    title_color: str = "k",
    title_fontsize: int = 20,
    cmap: str = "gray",
):
    block_size = dct_constants.block_size
    basis_functions = dct_constants.basis_functions
    basis_functions_image = np.zeros((block_size * block_size, block_size * block_size))
    for v in range(block_size):
        for u in range(block_size):
            basis_functions_image[
                v * block_size : (v + 1) * block_size,
                u * block_size : (u + 1) * block_size,
            ] = basis_functions[v, u]
    plt.figure(figsize=(figsize, figsize), facecolor=fig_facecolor)
    plt.title(
        f"DCT Basis functions (block size: {block_size}x{block_size})",
        color=title_color,
        fontsize=title_fontsize,
        fontweight="bold",
    )
    plt.imshow(basis_functions_image, cmap=cmap)
    plt.axis("off")
    for i in range(block_size):
        plt.axhline(i * block_size - 0.5, color=fig_facecolor)
        plt.axvline(i * block_size - 0.5, color=fig_facecolor)
    plt.tight_layout()
    fig = plt.gcf()
    ax = plt.gca()
    return fig, ax
