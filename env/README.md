# Environment Setup

The files in this directory are minimal dependency lists for the public GUDA
reproduction code. They are intentionally not full environment lock files.
Install PyTorch first with the CUDA wheel that matches your system, then install
the remaining dependencies from the relevant requirements file.

## Known-Working References

The CIFAR-10 pipeline has been tested with the following reference setup:

- Python 3.10
- CUDA 12.3 runtime/development image
- PyTorch and torchvision installed from the CUDA wheel index
- `diffusers[torch]`
- `accelerate==0.29.*`
- `datasets==2.*`
- `transformers==4.*`

A broader UnlearnCanvas benchmark environment used Python 3.8.5, CUDA 11.3,
PyTorch 1.11.0, and torchvision 0.12.0. That setup includes dependencies for
tasks outside the GUDA reproduction scope, so this repository provides a smaller
dependency list for the GUDA pipeline.

## CIFAR-10

Create a separate environment for the CIFAR-10 pipeline:

```bash
python3 -m venv .venv-cifar10
source .venv-cifar10/bin/activate
python3 -m pip install --upgrade pip

# Example for CUDA 12.3. Replace the index URL if your CUDA setup differs.
python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu123 \
  torch torchvision

python3 -m pip install -r env/cifar10.requirements.txt
```

Use the PyTorch installation command recommended for your platform if you are
not using CUDA 12.3. GPU reproduction requires a CUDA-enabled PyTorch build; a
CPU-only wheel is not sufficient for the training and attribution runs.

## UnlearnCanvas

Create a separate environment for the UnlearnCanvas pipeline:

```bash
python3 -m venv .venv-uc
source .venv-uc/bin/activate
python3 -m pip install --upgrade pip

# Example for CUDA 12.3. Replace the index URL if your CUDA setup differs.
python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu123 \
  torch torchvision

python3 -m pip install -r env/unlearncanvas.requirements.txt
```

Stable Diffusion 1.5 weights, CIFAR-10, UnlearnCanvas images, checkpoints,
generated images, and precomputed caches are not included in this repository.
They must be obtained or generated separately under the applicable licenses,
terms, and access policies.

## Optional Services

Weights & Biases is optional at runtime, but some training scripts import
`wandb`, so it remains listed in the requirements files. Disable or avoid W&B
logging flags if you do not want to log runs.
