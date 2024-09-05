import torch


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

    r = y + 1.403 * cr_shifted
    g = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3).clamp(0, 1)


def rgb_to_ycbcr(image) -> torch.Tensor:
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
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + delta
    cr = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], -3)
