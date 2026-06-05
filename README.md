# DCT-Autoencoder

A PyTorch-based implementation of a 2D Discrete Cosine Transform (DCT) autoencoder for RGB images.

[![PyPI version](https://badge.fury.io/py/dct-autoencoder.svg)](https://badge.fury.io/py/dct-autoencoder)

## Overview

`dct-autoencoder` is a **non-trainable, analytically defined autoencoder** that reproduces the core ideas behind JPEG compression:

1. Convert RGB to **YCbCr** to separate luminance from chrominance.
2. Partition each channel into non-overlapping blocks and apply the **2-D DCT** via strided convolution.
3. Optionally **zero high-frequency coefficients** for lossy compression.

The transform is fully differentiable and integrates cleanly into PyTorch pipelines. Encoding is **lossless by default**; compression is opt-in at construction time.

## Project structure

```
dct-autoencoder/
├── src/dct_autoencoder/       # Package source
│   ├── __init__.py            # Public API exports
│   ├── basis.py               # DCTBasis, get_dct_basis()
│   ├── core.py                # DCTAutoencoder (nn.Module)
│   └── utils.py               # RGB ↔ YCbCr conversions
├── notebooks/
│   ├── quick_start.ipynb      # Minimal getting-started example
│   ├── full_tutorial.ipynb    # In-depth walkthrough
│   ├── visualization.py       # DCT basis-function plotting helpers
│   └── test_images/           # Sample images for notebooks
├── assets/figures/            # Diagrams and figures
├── pyproject.toml
└── README.md
```

## Installation

Requires **Python 3.12+**.

### 1. Install PyTorch

Install PyTorch and torchvision first, following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/). Choose the build that matches your hardware (CPU, CUDA, ROCm, etc.).

### 2. Install this package

Then install `dct-autoencoder` from PyPI:

```bash
pip install dct-autoencoder
```

PyTorch is required to use `DCTAutoencoder` but is not bundled so you can pick the correct build for your system.

### Development setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
git clone https://github.com/dariush-bahrami/dct-autoencoder.git
cd dct-autoencoder
uv sync --group dev
```

## Quick start

```python
import torch
from dct_autoencoder import DCTAutoencoder

model = DCTAutoencoder(
    block_size=8,
    num_luminance_compressed_channels=10,   # optional low-pass on Y
    num_chrominance_compressed_channels=5,  # optional low-pass on Cb/Cr
)

# Input: (batch, 3, H, W), values in [0, 1]; H and W must be divisible by block_size
images = torch.rand(1, 3, 256, 256)

encoded = model.encode(images)
compressed = model.compress(encoded)
reconstructed = model.decode(model.decompress(compressed))
```

### Public API

| Symbol | Module | Description |
|--------|--------|-------------|
| `DCTAutoencoder` | `core` | Encode/decode RGB images; optional frequency-domain compression |
| `DCTBasis` | `basis` | Named tuple of precomputed DCT basis data for a block size |
| `get_dct_basis` | `basis` | Build a `DCTBasis` for a given block size |

## Notebooks

| Notebook | Description |
|----------|-------------|
| [notebooks/quick_start.ipynb](https://github.com/dariush-bahrami/dct-autoencoder/blob/main/notebooks/quick_start.ipynb) | Minimal example: load an image, encode, compress, decode |
| [notebooks/full_tutorial.ipynb](https://github.com/dariush-bahrami/dct-autoencoder/blob/main/notebooks/full_tutorial.ipynb) | Full tutorial covering math, API details, and interactive exploration |

## Visualizations

### Computation graph

![Computation Graph](https://raw.githubusercontent.com/dariush-bahrami/dct-autoencoder/main/assets/figures/DCT%20Autoencoder.drawio.png)

### DCT basis functions

DCT basis functions for a block size of 16:

![DCT Basis Functions](https://raw.githubusercontent.com/dariush-bahrami/dct-autoencoder/main/assets/figures/dct_basis_functions_block_size_16.png)

You can regenerate basis-function plots locally with `notebooks/visualization.py` and `get_dct_basis()`.

## License

MIT — see [LICENSE](LICENSE).
