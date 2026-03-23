# TomoDpDt

`TomoDpDt` is a research-oriented repository for tomographic reconstruction with jointly unknown object structure and rotations. The codebase is organized as a modular workflow for simulation, forward imaging, latent-space initialization, and optimization.

The repository currently has two main faces:

- `tomodpdt/` contains the reusable package code.
- `Notebooks/` contains tutorial-style examples for synthetic and experimental reconstruction cases.

## What is in the repo?

```text
tomodpdt/
  application.py                 Main optimization / reconstruction application
  estimate_rotations_from_latent.py
                                 Rotation initialization from latent space and correlations
  image_modalities_dt.py         DeepTrack-based image formation models
  imaging_modality_torch.py      Torch-based imaging models
  rotations.py                   Quaternion generation and utilities
  simulate.py                    Synthetic dataset generation
  volumes.py                     Procedural and precomputed 3D test volumes
  plotting.py                    Plotting and visualization helpers
  forward_module.py              Simple forward projector
  fft_loader.py                  FFT/field conversion helpers
  helpers.py                     Tracking and segmentation helpers
  vaemod.py                      Convolutional VAE components
Notebooks/                       Tutorial notebooks
test_data/                       Precomputed synthetic and experimental data
run_notebooks.py                 Batch notebook execution helper
```

## Main dependencies

- `deeplay`
- `deeptrack`
- `torch`
- `lightning`
- `numpy`, `scipy`, `matplotlib`

## Getting started

1. Create a Python environment.
2. Install the dependencies from `requirements.txt`.
3. Start with the notebooks in `Notebooks/` to see the intended workflows.

Minimal example:

```python
from tomodpdt.application import Tomography
from tomodpdt.simulate import create_data

volume, quaternions, projections, imaging_model = create_data(
    image_modality="brightfield",
    rotation_case="random_sinusoidal",
    samples=100,
)

tomo = Tomography(volume_size=volume.shape)
```

## Repository goals

This codebase is well suited to a tutorial paper companion repository: it contains the scientific workflow, example datasets, and notebooks that show how the pieces fit together. The most valuable polish areas are documentation, dependency clarity, path robustness, and a cleaner separation between library code and demo code.
