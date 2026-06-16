import torch

# Full-range BT.601 / JFIF (ITU-T T.871) constants.
#
# The luma weights below *define* the transform exactly. Everything else is
# derived from them so the forward and inverse stay a perfectly matched pair:
#   * chroma denominators: 2*(1 - KB) = 1.772 and 2*(1 - KR) = 1.402
#   * inverse green coefficients: KR*1.402/KG and KB*1.772/KG
#
# Expressing the chroma terms as divisions by these exact denominators (rather
# than multiplying by pre-rounded reciprocals like 0.564 / 0.713, or by the
# 6-digit JFIF constants 0.168736 / 0.344136 / 0.714136) keeps the conversion
# as numerically precise as the standard allows.
_KR: float = 0.299
_KG: float = 0.587
_KB: float = 0.114
_CB_DENOM: float = 2.0 * (1.0 - _KB)  # 1.772
_CR_DENOM: float = 2.0 * (1.0 - _KR)  # 1.402


def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """Converts an image from YCbCr to RGB color space.

    Args:
        image (torch.Tensor): The input image. The image should have shape
            (*, 3, height, width). Image values should be in the range [0, 1].

    Returns:
        torch.Tensor: The output image in RGB color space.
    """
    y = image[..., 0, :, :]
    cb = image[..., 1, :, :]
    cr = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted = cb - delta
    cr_shifted = cr - delta

    # Exact algebraic inverse of the forward transform below.
    r = y + _CR_DENOM * cr_shifted
    b = y + _CB_DENOM * cb_shifted
    g = y - (_KR * _CR_DENOM * cr_shifted + _KB * _CB_DENOM * cb_shifted) / _KG
    return torch.stack([r, g, b], -3)


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    """Converts an image from RGB to YCbCr color space.

    Args:
        image (torch.Tensor): The input image. The image should have shape
            (*, 3, height, width). Image values should be in the range [0, 1].

    Returns:
        torch.Tensor: The output image in YCbCr color space.
    """
    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]

    delta: float = 0.5
    y = _KR * r + _KG * g + _KB * b
    cb = (b - y) / _CB_DENOM + delta
    cr = (r - y) / _CR_DENOM + delta
    return torch.stack([y, cb, cr], -3)
