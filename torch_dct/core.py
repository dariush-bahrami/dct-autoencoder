import math
from typing import NamedTuple

import cv2
import kornia as K
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


class DCTGrayImageEncodeResult(NamedTuple):
    encodings: torch.Tensor
    padding: tuple


class DCTColorImageEncodeResult(NamedTuple):
    luminance_encodings: torch.Tensor
    chrominance_blue_encodings: torch.Tensor
    chrominance_red_encodings: torch.Tensor
    padding: tuple


class DCTCompressionResult(NamedTuple):
    encodings: torch.Tensor
    padding: tuple
    num_luminance_channels: int
    num_chrominance_channels: int


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

    def _preprocess_gray_image(
        self, image: np.array
    ) -> tuple[torch.Tensor, tuple[int, int]]:
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

    def encode_gray_image(self, image: np.ndarray) -> torch.Tensor:
        assert image.ndim == 2, "Input image must be grayscale"
        images, padding = self._preprocess_gray_image(image)
        c1 = self.multiplication_factor_scalar
        c2 = self.multiplication_factor_matrix
        encodings = (
            c1 * c2 * F.conv2d(images, self.kernels, stride=self.block_size.item())
        )
        return DCTGrayImageEncodeResult(encodings, padding)

    def _postprocess_gray_image(
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

    def decode_gray_image(self, encoded_result: DCTGrayImageEncodeResult) -> np.ndarray:
        encodings = encoded_result.encodings
        c1 = self.multiplication_factor_scalar
        c2 = self.multiplication_factor_matrix
        decodings = c1 * F.conv_transpose2d(
            encodings * c2, self.kernels, stride=self.block_size.item()
        )
        return self._postprocess_gray_image(decodings, encoded_result.padding)

    def encode_rgb_image(self, image: np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        luminance = image[:, :, 0]
        chrominance_blue = image[:, :, 1]
        chrominance_red = image[:, :, 2]

        # DCT encode
        luminance = self.encode_gray_image(luminance)
        chrominance_blue = self.encode_gray_image(chrominance_blue)
        chrominance_red = self.encode_gray_image(chrominance_red)
        padding = luminance.padding

        return DCTColorImageEncodeResult(
            luminance.encodings,
            chrominance_blue.encodings,
            chrominance_red.encodings,
            padding,
        )

    def decode_rgb_image(self, encoded_result: DCTColorImageEncodeResult) -> np.ndarray:
        luminance = encoded_result.luminance_encodings
        chrominance_blue = encoded_result.chrominance_blue_encodings
        chrominance_red = encoded_result.chrominance_red_encodings
        padding = encoded_result.padding
        luminance = self.decode_gray_image(DCTGrayImageEncodeResult(luminance, padding))
        chrominance_blue = self.decode_gray_image(
            DCTGrayImageEncodeResult(chrominance_blue, padding)
        )
        chrominance_red = self.decode_gray_image(
            DCTGrayImageEncodeResult(chrominance_red, padding)
        )
        image = np.stack([luminance, chrominance_blue, chrominance_red], axis=-1)
        image = cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
        return image

    def compress_rgb_image(
        self,
        image: np.ndarray,
        luminance_compresion_ratio: float = 0.5,
        chrominance_compresion_ratio: float = 0.75,
    ) -> DCTCompressionResult:
        dct_encode_resul: DCTColorImageEncodeResult = self.encode_rgb_image(image)

        luminance_encodings = dct_encode_resul.luminance_encodings
        chrominance_red_encodings = dct_encode_resul.chrominance_red_encodings
        chrominance_blue_encodings = dct_encode_resul.chrominance_blue_encodings
        padding = dct_encode_resul.padding
        num_luminance_channels = math.floor(
            luminance_encodings.shape[1] * (1 - luminance_compresion_ratio)
        )
        num_chrominance_channels = math.floor(
            chrominance_red_encodings.shape[1] * (1 - chrominance_compresion_ratio)
        )
        compressed_image_encodings = torch.cat(
            [
                luminance_encodings[:, :num_luminance_channels, :, :],
                chrominance_blue_encodings[:, :num_chrominance_channels, :, :],
                chrominance_red_encodings[:, :num_chrominance_channels, :, :],
            ],
            dim=1,
        )

        return DCTCompressionResult(
            compressed_image_encodings,
            padding,
            num_luminance_channels,
            num_chrominance_channels,
        )

    def decompress_rgb_image(
        self, compression_result: DCTCompressionResult
    ) -> np.ndarray:
        encodings = compression_result.encodings
        batch_size, _, h, w = encodings.shape
        num_decompressed_channels = self.block_size**2
        luminance_encodings = torch.zeros((batch_size, num_decompressed_channels, h, w))
        chrominance_blue_encodings = torch.zeros(
            (batch_size, num_decompressed_channels, h, w)
        )
        chrominance_red_encodings = torch.zeros(
            (batch_size, num_decompressed_channels, h, w)
        )
        luminance_encodings[:, : compression_result.num_luminance_channels, :, :] = (
            encodings[:, : compression_result.num_luminance_channels, :, :]
        )
        chrominance_blue_encodings[
            :, : compression_result.num_chrominance_channels, :, :
        ] = encodings[
            :,
            compression_result.num_luminance_channels : compression_result.num_luminance_channels
            + compression_result.num_chrominance_channels,
            :,
            :,
        ]
        chrominance_red_encodings[
            :, : compression_result.num_chrominance_channels, :, :
        ] = encodings[
            :,
            compression_result.num_luminance_channels
            + compression_result.num_chrominance_channels :,
            :,
            :,
        ]

        return self.decode_rgb_image(
            DCTColorImageEncodeResult(
                luminance_encodings,
                chrominance_blue_encodings,
                chrominance_red_encodings,
                compression_result.padding,
            )
        )


class DCTEncoderLayer(nn.Module):
    def __init__(self, block_size: int = 32):
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

        self.out_channels = (block_size**2) * 3

    def forward(self, rgb_images_batch: torch.Tensor) -> torch.Tensor:
        # check input
        b, c, h, w = rgb_images_batch.shape
        assert c == 3, "Input images must be RGB"
        assert (
            h % self.block_size == 0 and w % self.block_size == 0
        ), "Image dimensions must be divisible by the block size"
        # convert to YCbCr
        ycbcr_tsr = K.color.rgb_to_ycbcr(rgb_images_batch)
        # normalize to -1, 1
        ycbcr_tsr = 2 * ycbcr_tsr - 1
        y = ycbcr_tsr[:, [0], :, :]
        cb = ycbcr_tsr[:, [1], :, :]
        cr = ycbcr_tsr[:, [2], :, :]

        # DCT encode
        c1 = self.multiplication_factor_scalar
        c2 = self.multiplication_factor_matrix
        y = c1 * c2 * F.conv2d(y, self.kernels, stride=self.block_size.item())
        cb = c1 * c2 * F.conv2d(cb, self.kernels, stride=self.block_size.item())
        cr = c1 * c2 * F.conv2d(cr, self.kernels, stride=self.block_size.item())

        return torch.cat([y, cb, cr], dim=1)


class DCTDecoderLayer(nn.Module):
    def __init__(self, encoder: DCTEncoderLayer):
        super().__init__()
        self.compressor = encoder

    def forward(self, encodings_batch: torch.Tensor) -> torch.Tensor:
        # check input
        b, c, h, w = encodings_batch.shape
        org_ch = self.compressor.block_size**2
        y = encodings_batch[:, :org_ch, :, :]
        cb = encodings_batch[:, org_ch : org_ch * 2, :, :]
        cr = encodings_batch[:, org_ch * 2 :, :, :]

        # DCT Decode
        c1 = self.compressor.multiplication_factor_scalar
        c2 = self.compressor.multiplication_factor_matrix
        y = c1 * F.conv_transpose2d(
            y * c2, self.compressor.kernels, stride=self.compressor.block_size.item()
        )
        cb = c1 * F.conv_transpose2d(
            cb * c2, self.compressor.kernels, stride=self.compressor.block_size.item()
        )
        cr = c1 * F.conv_transpose2d(
            cr * c2, self.compressor.kernels, stride=self.compressor.block_size.item()
        )

        # convert to RGB
        ycbcr_tsr = torch.cat([y, cb, cr], dim=1)
        ycbcr_tsr = ycbcr_tsr / 2 + 0.5
        rgb_images_batch = K.color.ycbcr_to_rgb(ycbcr_tsr)
        return rgb_images_batch
