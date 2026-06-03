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
        num_luminance_compressed_channels (int | None, optional): Number of lowest-frequency
            luminance channels to retain after compression. ``None`` keeps all channels
            (no compression). Defaults to ``None``.
        num_chrominance_compressed_channels (int | None, optional): Number of lowest-frequency
            chrominance channels to retain after compression. ``None`` keeps all channels
            (no compression). Defaults to ``None``.
    """

    def __init__(
        self,
        block_size: int = 8,
        num_luminance_compressed_channels: int | None = None,
        num_chrominance_compressed_channels: int | None = None,
    ) -> None:
        super().__init__()
        total_channels = block_size**2

        if num_luminance_compressed_channels is not None and not (
            1 <= num_luminance_compressed_channels <= total_channels
        ):
            raise ValueError(
                f"num_luminance_compressed_channels must be between 1 and {total_channels}, "
                f"got {num_luminance_compressed_channels}"
            )
        if num_chrominance_compressed_channels is not None and not (
            1 <= num_chrominance_compressed_channels <= total_channels
        ):
            raise ValueError(
                f"num_chrominance_compressed_channels must be between 1 and {total_channels}, "
                f"got {num_chrominance_compressed_channels}"
            )

        dct_basis = get_dct_basis(block_size)
        basis_functions = dct_basis.basis_functions
        kernels = basis_functions.reshape(-1, block_size, block_size)
        spatial_frequencies_magnitude = dct_basis.spatial_frequencies_magnitude.reshape(
            -1
        )
        spatial_frequencies_components = (
            dct_basis.spatial_frequencies_components.reshape(-1, 2)
        )
        sort_indices = np.argsort(spatial_frequencies_magnitude, kind="stable")
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

        self.embedding_dimension = total_channels * 3

        # compressor initialization — kernels are already sorted by ascending frequency,
        # so keeping the first N channels retains the N lowest frequencies exactly.
        lum_n = num_luminance_compressed_channels
        chr_n = num_chrominance_compressed_channels
        self.do_compression = (lum_n is not None and lum_n < total_channels) or (
            chr_n is not None and chr_n < total_channels
        )

        lum_passband = lum_n if lum_n is not None else total_channels
        chr_passband = chr_n if chr_n is not None else total_channels

        lum_mask = torch.zeros(total_channels, dtype=torch.bool)
        lum_mask[:lum_passband] = True
        chr_mask = torch.zeros(total_channels, dtype=torch.bool)
        chr_mask[:chr_passband] = True

        self.register_buffer("compression_luminance_mask", lum_mask)
        self.register_buffer("compression_chrominance_mask", chr_mask)
        self.register_buffer(
            "compression_luminance_passband", torch.tensor(lum_passband)
        )
        self.register_buffer(
            "compression_chrominance_passband", torch.tensor(chr_passband)
        )

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

    def decode(
        self, encodings_batch: torch.Tensor, clamp_output: bool = True
    ) -> torch.Tensor:
        """Decodes the input encoded images.

        Args:
            encodings_batch (torch.Tensor): The input encoded images.
            clamp_output (bool, optional): Whether to clamp the output to the range [0, 1]. Defaults to True.
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
        if clamp_output:
            # clamp is expected for display; note it makes decode non-linear for out-of-range values
            rgb_images_batch = rgb_images_batch.clamp(0, 1)
        return rgb_images_batch

    def get_num_compressed_channels(self) -> int:
        if not self.do_compression:
            return self.block_size**2 * 3
        else:
            return (
                self.compression_luminance_passband.item()
                + 2 * self.compression_chrominance_passband.item()
            )

    def compress(self, encodings: torch.Tensor) -> torch.Tensor:
        if not self.do_compression:
            return encodings
        else:
            l, c1, c2 = encodings.chunk(3, dim=1)
            luminance_mask = self.compression_luminance_mask
            chrominance_mask = self.compression_chrominance_mask
            l = l[:, luminance_mask, :, :]
            c1 = c1[:, chrominance_mask, :, :]
            c2 = c2[:, chrominance_mask, :, :]
            compressed_encoding = torch.cat([l, c1, c2], dim=1)
            return compressed_encoding

    def decompress(self, compressed_encoding: torch.Tensor) -> torch.Tensor:
        if not self.do_compression:
            return compressed_encoding
        else:
            batch_size, _, height, width = compressed_encoding.shape
            device = compressed_encoding.device
            dtype = compressed_encoding.dtype
            luminance_mask = self.compression_luminance_mask
            chrominance_mask = self.compression_chrominance_mask
            luminance_passband = self.compression_luminance_passband.item()
            chrominance_passband = self.compression_chrominance_passband.item()
            l_comp, c1_comp, c2_comp = compressed_encoding.split(
                [luminance_passband, chrominance_passband, chrominance_passband],
                dim=1,
            )
            l = torch.zeros(
                batch_size,
                self.block_size**2,
                height,
                width,
                device=device,
                dtype=dtype,
            )
            l[:, luminance_mask, :, :] = l_comp
            c1 = torch.zeros(
                batch_size,
                self.block_size**2,
                height,
                width,
                device=device,
                dtype=dtype,
            )
            c1[:, chrominance_mask, :, :] = c1_comp
            c2 = torch.zeros(
                batch_size,
                self.block_size**2,
                height,
                width,
                device=device,
                dtype=dtype,
            )
            c2[:, chrominance_mask, :, :] = c2_comp
            decompressed_encoding = torch.cat([l, c1, c2], dim=1)
            return decompressed_encoding
