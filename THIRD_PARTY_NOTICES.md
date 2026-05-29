# Third-Party Dependencies and External Assets

This repository depends on third-party open-source packages, models, and
datasets that users install or obtain separately. Third-party source code,
model weights, checkpoints, generated images, precomputed caches, and image
datasets are not intentionally vendored in this repository.

Users are responsible for complying with the licenses, terms, and access
policies of the external projects and assets they install, download, or use.

Key external dependencies and assets include:

- PyTorch and torchvision
- Hugging Face diffusers, transformers, datasets, accelerate, and related
  Python packages
- Stable Diffusion 1.5 model weights, which are not distributed here
- CLIP / OpenCLIP dependencies used by prompt and attribution utilities
- CIFAR-10, downloaded by torchvision or provided by the user
- UnlearnCanvas image data, which is not distributed here

The UnlearnCanvas prompt and descriptor files included in this repository are
metadata used to reproduce the paper experiments. Confirm that publishing this
metadata is covered by the applicable dataset and AI ethics review processes
before public release.
