# DCT-Autoencoder

A PyTorch-based implementation of 2D Discrete Cosine Transform (DCT) autoencoder.

[![PyPI version](https://badge.fury.io/py/dct-autoencoder.svg)](https://badge.fury.io/py/dct-autoencoder)

## Overview

The `dct-autoencoder` package offers a PyTorch implementation of the 2D Discrete Cosine Transform (DCT), which is fully differentiable and can be integrated into deep learning models. It is particularly useful for reducing the spatial dimensions of images by transforming them into the frequency domain via DCT. Inspired by the JPEG algorithm, the package also includes a compression method using low-pass filtering, which reduces the number of frequency domain coefficients while retaining most of the image information.


## Installation

Install the package via pip:

```bash
pip install dct-autoencoder
```

## Usage

For detailed usage examples, refer to the [Usage Notebook](https://github.com/dariush-bahrami/dct-autoencoder/blob/main/usage.ipynb). It provides code snippets and demonstrations of the DCT autoencoder in action.

## Visualizations

### Computation Graph

The following figure illustrates the computation graph of the DCT autoencoder:

![Computation Graph](https://raw.githubusercontent.com/dariush-bahrami/dct-autoencoder/main/assets/figures/DCT%20Autoencoder.drawio.png)

### DCT Basis Functions

DCT basis functions for a block size of 16:

![DCT Basis Functions](https://raw.githubusercontent.com/dariush-bahrami/dct-autoencoder/main/assets/figures/dct_basis_functions_block_size_16.png)

## TODO

- [x] Add support for color images
- [x] Improve documentation
- [ ] Add unit tests
- [x] Distribute package on PyPI
