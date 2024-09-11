import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .basis import get_dct_basis
from .utils import rgb_to_ycbcr, ycbcr_to_rgb


class DCTAutoencoder(nn.Module):
    """DCT Autoencoder.

    Args:
        block_size (int, optional): The block size. Defaults to 8.
    """

    def __init__(self, block_size: int = 8) -> None:
        super().__init__()
        dct_basis = get_dct_basis(block_size)
        basis_functions = dct_basis.basis_functions
        kernels = basis_functions.reshape(-1, block_size, block_size)
        spatial_frequencies_magnitude = dct_basis.spatial_frequencies_magnitude.reshape(
            -1
        )
        spatial_frequencies_components = (
            dct_basis.spatial_frequencies_components.reshape(-1, 2)
        )
        sort_indices = np.argsort(spatial_frequencies_magnitude)
        kernels = kernels[sort_indices]
        spatial_frequencies_magnitude = spatial_frequencies_magnitude[sort_indices]
        spatial_frequencies_components = spatial_frequencies_components[sort_indices]
        kernels = kernels[:, np.newaxis, :, :]
        multiplication_factor_scalar = dct_basis.multiplication_factor_scalar
        multiplication_factor_matrix = dct_basis.multiplication_factor_matrix
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
        self.register_buffer(
            "spatial_frequencies_components",
            torch.from_numpy(spatial_frequencies_components),
        )
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer(
            "multiplication_factor_scalar", torch.tensor(multiplication_factor_scalar)
        )
        self.register_buffer(
            "multiplication_factor_matrix",
            torch.from_numpy(multiplication_factor_matrix),
        )

        self.embedding_dimension = (block_size**2) * 3

    def encode(self, rgb_images_batch: torch.Tensor) -> torch.Tensor:
        """Encodes the input RGB images.

        Args:
            rgb_images_batch (torch.Tensor): The input RGB images. The images should
                have shape (*, 3, height, width). Image values should be in the range [0, 1].

        Returns:
            torch.Tensor: The encoded images.
        """
        # check input
        b, c, h, w = rgb_images_batch.shape
        if c != 3:
            raise ValueError("Input images must be RGB")
        if h % self.block_size != 0 or w % self.block_size != 0:
            raise ValueError("Image dimensions must be divisible by the block size")
        # convert to YCbCr
        ycbcr_tsr = rgb_to_ycbcr(rgb_images_batch)
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
        encodings_batch = torch.cat([y, cb, cr], dim=1)
        # scale down
        encodings_batch = encodings_batch / self.block_size
        return encodings_batch

    def decode(self, encodings_batch: torch.Tensor) -> torch.Tensor:
        """Decodes the input encoded images.

        Args:
            encodings_batch (torch.Tensor): The input encoded images.

        Returns:
            torch.Tensor: The decoded images.
        """
        # scale up
        encodings_batch = encodings_batch * self.block_size
        org_ch = self.block_size**2
        y = encodings_batch[:, :org_ch, :, :]
        cb = encodings_batch[:, org_ch : org_ch * 2, :, :]
        cr = encodings_batch[:, org_ch * 2 :, :, :]

        # DCT Decode
        c1 = self.multiplication_factor_scalar
        c2 = self.multiplication_factor_matrix
        y = c1 * F.conv_transpose2d(y * c2, self.kernels, stride=self.block_size.item())
        cb = c1 * F.conv_transpose2d(
            cb * c2, self.kernels, stride=self.block_size.item()
        )
        cr = c1 * F.conv_transpose2d(
            cr * c2, self.kernels, stride=self.block_size.item()
        )

        # convert to RGB
        ycbcr_tsr = torch.cat([y, cb, cr], dim=1)
        ycbcr_tsr = ycbcr_tsr / 2 + 0.5
        rgb_images_batch = ycbcr_to_rgb(ycbcr_tsr)
        return rgb_images_batch

    def get_num_compressed_channels(
        self,
        luminance_compression_ratio: float = 1 / 2,
        chrominance_compression_ratio: float = 1 / 4,
    ) -> int:
        """Get the number of compressed channels.

        Args:
            luminance_compression_ratio (float, optional): The luminance compression
                ratio. Defaults to 1/2.
            chrominance_compression_ratio (float, optional): The chrominance compression
                ratio. Defaults to 1/4.

        Returns:
            int: The number of compressed channels.
        """
        num_per_channel_encodings = self.block_size**2
        num_luminance_encodings = torch.round(
            num_per_channel_encodings * luminance_compression_ratio
        ).int()
        num_chrominance_encodings = torch.round(
            num_per_channel_encodings * chrominance_compression_ratio
        ).int()
        return (num_luminance_encodings + 2 * num_chrominance_encodings).item()

    def compress(
        self,
        encodings_batch: torch.Tensor,
        luminance_compression_ratio: float = 1 / 2,
        chrominance_compression_ratio: float = 1 / 4,
    ) -> torch.Tensor:
        """Compresses the input encodings.

        Args:
            encodings_batch (torch.Tensor): The input encodings.
            luminance_compression_ratio (float, optional): The luminance compression
                ratio. Defaults to 1/2.
            chrominance_compression_ratio (float, optional): The chrominance compression
                ratio. Defaults to 1/4.

        Returns:
            torch.Tensor: The compressed encodings.
        """

        num_per_channel_encodings = self.block_size**2
        num_luminance_encodings = torch.round(
            num_per_channel_encodings * luminance_compression_ratio
        ).int()
        num_chrominance_encodings = torch.round(
            num_per_channel_encodings * chrominance_compression_ratio
        ).int()

        luminance_encodings = encodings_batch[:, :num_per_channel_encodings]
        chrominance_blue_encodings = encodings_batch[
            :, num_per_channel_encodings : 2 * num_per_channel_encodings
        ]
        chrominance_red_encodings = encodings_batch[:, 2 * num_per_channel_encodings :]

        luminance_encodings = luminance_encodings[:, :num_luminance_encodings]
        chrominance_blue_encodings = chrominance_blue_encodings[
            :, :num_chrominance_encodings
        ]
        chrominance_red_encodings = chrominance_red_encodings[
            :, :num_chrominance_encodings
        ]
        compressed_dct_encodings = torch.cat(
            [
                luminance_encodings,
                chrominance_blue_encodings,
                chrominance_red_encodings,
            ],
            dim=1,
        )
        return compressed_dct_encodings

    def decompress(
        self,
        compressed_encodings_batch: torch.Tensor,
        luminance_compression_ratio: float = 1 / 2,
        chrominance_compression_ratio: float = 1 / 4,
    ) -> torch.Tensor:
        """Decompresses the input compressed encodings.

        Args:
            compressed_encodings_batch (torch.Tensor): The input compressed encodings.
            luminance_compression_ratio (float, optional): The luminance compression
                ratio. Defaults to 1/2.
            chrominance_compression_ratio (float, optional): The chrominance compression
                ratio. Defaults to 1/4.

        Returns:
            torch.Tensor: The decompressed encodings.
        """

        b, _, h, w = compressed_encodings_batch.shape
        dtype = compressed_encodings_batch.dtype
        device = compressed_encodings_batch.device

        num_per_channel_encodings = self.block_size**2
        num_luminance_encodings = torch.floor(
            num_per_channel_encodings * luminance_compression_ratio
        ).int()
        num_chrominance_encodings = torch.floor(
            num_per_channel_encodings * chrominance_compression_ratio
        ).int()
        compressed_luminance_encodings = compressed_encodings_batch[
            :, :num_luminance_encodings
        ]
        compressed_chrominance_blue_encodings = compressed_encodings_batch[
            :,
            num_luminance_encodings : num_luminance_encodings
            + num_chrominance_encodings,
        ]
        compressed_chrominance_red_encodings = compressed_encodings_batch[
            :, num_luminance_encodings + num_chrominance_encodings :
        ]

        luminance_encodings = torch.zeros(
            b, num_per_channel_encodings, h, w, dtype=dtype, device=device
        )
        luminance_encodings[:, :num_luminance_encodings, :, :] = (
            compressed_luminance_encodings
        )
        chrominance_blue_encodings = torch.zeros(
            b, num_per_channel_encodings, h, w, dtype=dtype, device=device
        )
        chrominance_blue_encodings[:, :num_chrominance_encodings, :, :] = (
            compressed_chrominance_blue_encodings
        )
        chrominance_red_encodings = torch.zeros(
            b, num_per_channel_encodings, h, w, dtype=dtype, device=device
        )
        chrominance_red_encodings[:, :num_chrominance_encodings, :, :] = (
            compressed_chrominance_red_encodings
        )
        decompressed_dct_encodings = torch.cat(
            [
                luminance_encodings,
                chrominance_blue_encodings,
                chrominance_red_encodings,
            ],
            dim=1,
        )

        return decompressed_dct_encodings
