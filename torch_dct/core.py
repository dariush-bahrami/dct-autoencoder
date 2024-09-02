from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


class DCTConstants(NamedTuple):
    basis_functions: np.ndarray
    spatial_frequencies_components: np.ndarray
    spatial_frequencies_magnitude: np.ndarray
    multiplication_factor_matrix: np.ndarray
    multiplication_factor_scalar: float
    block_size: int


def get_dct_constants(block_size: int = 8) -> DCTConstants:
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

    return DCTConstants(
        basis_functions=basis_functions,
        spatial_frequencies_components=spatial_frequencies,
        spatial_frequencies_magnitude=spatial_frequencies_magnitude,
        multiplication_factor_matrix=multiplication_factor_matrix,
        multiplication_factor_scalar=multiplication_factor_scalar,
        block_size=block_size,
    )


def visualize_dct_basis_functions(
    dct_constants: DCTConstants,
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


class DCTEncodeResult(NamedTuple):
    encodings: torch.Tensor
    padding: tuple


class DCTModule(nn.Module):
    def __init__(self, block_size: int = 8):
        super().__init__()
        dct_constants = get_dct_constants(block_size)
        basis_functions = dct_constants.basis_functions
        kernels = basis_functions.reshape(-1, block_size, block_size)
        spatial_frequencies_magnitude = (
            dct_constants.spatial_frequencies_magnitude.reshape(-1)
        )
        sort_indices = np.argsort(spatial_frequencies_magnitude)
        kernels = kernels[sort_indices]
        spatial_frequencies_magnitude = spatial_frequencies_magnitude[sort_indices]
        kernels = kernels[:, np.newaxis, :, :]
        multiplication_factor_scalar = dct_constants.multiplication_factor_scalar
        multiplication_factor_matrix = dct_constants.multiplication_factor_matrix
        multiplication_factor_matrix = multiplication_factor_matrix.reshape(-1)
        multiplication_factor_matrix = multiplication_factor_matrix[sort_indices]
        multiplication_factor_matrix = multiplication_factor_matrix[
            np.newaxis, :, np.newaxis, np.newaxis
        ]
        self.register_buffer("kernels", torch.from_numpy(kernels))
        self.register_buffer(
            "spatial_frequencies_magnitude",
            torch.from_numpy(spatial_frequencies_magnitude),
        )
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer(
            "multiplication_factor_scalar", torch.tensor(multiplication_factor_scalar)
        )
        self.register_buffer(
            "multiplication_factor_matrix",
            torch.from_numpy(multiplication_factor_matrix),
        )

    def _preprocess(self, image: np.array) -> tuple[torch.Tensor, tuple[int, int]]:
        h, w = image.shape
        h_pad = self.block_size - h % self.block_size if h % self.block_size != 0 else 0
        w_pad = self.block_size - w % self.block_size if w % self.block_size != 0 else 0
        image = np.pad(image, ((0, h_pad), (0, w_pad)), mode="reflect")
        # normalize image to [-1, 1]
        image = ((image / 255.0) - 0.5) / 0.5
        # add channel dimension
        image = image[np.newaxis, :, :]
        image = torch.from_numpy(image).unsqueeze(0).float()
        return image, (h_pad, w_pad)

    def encode(self, image: np.ndarray) -> torch.Tensor:
        assert image.ndim == 2, "Input image must be grayscale"
        images, padding = self._preprocess(image)
        c1 = self.multiplication_factor_scalar
        c2 = self.multiplication_factor_matrix
        encodings = (
            c1 * c2 * F.conv2d(images, self.kernels, stride=self.block_size.item())
        )
        return DCTEncodeResult(encodings, padding)

    def _postprocess(
        self, decodings: torch.Tensor, padding: tuple[int, int]
    ) -> Image.Image:
        decodings = decodings
        decodings = decodings.squeeze(0).squeeze(0).detach().cpu().numpy()
        h, w = decodings.shape[-2:]
        h_pad, w_pad = padding
        h -= h_pad
        w -= w_pad
        decodings = decodings[:h, :w]
        decodings = (decodings * 0.5 + 0.5) * 255
        decodings = np.clip(decodings, 0, 255).astype(np.uint8)
        return decodings

    def decode(self, encoded_result: DCTEncodeResult) -> Image.Image:
        encodings = encoded_result.encodings
        c1 = self.multiplication_factor_scalar
        c2 = self.multiplication_factor_matrix
        decodings = c1 * F.conv_transpose2d(
            encodings * c2, self.kernels, stride=self.block_size.item()
        )
        return self._postprocess(decodings, encoded_result.padding)
