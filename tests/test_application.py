import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader


application = pytest.importorskip("tomodpdt.application")
simulate = pytest.importorskip("tomodpdt.simulate")
dl = pytest.importorskip("deeplay")


def _dummy_rotation_initialization(n_frames, rotation_optim_case="quaternion"):
    quaternions = torch.zeros(n_frames, 4, dtype=torch.float32)
    quaternions[:, 0] = 1.0
    basis = torch.eye(n_frames, dtype=torch.float32)
    coeffs = quaternions.clone()
    peaks = torch.tensor([0, n_frames], dtype=torch.long)

    result = {
        "quaternions": quaternions,
        "basis": basis,
        "coeffs": coeffs,
        "peaks": peaks,
    }

    if rotation_optim_case == "basis":
        result["coeffs"] = quaternions.clone()

    return result


def _make_small_tomography(rotation_optim_case="quaternion", samples=8):
    volume_np = np.zeros((16, 16, 16), dtype=np.float32)
    volume_np[5:11, 6:10, 4:12] = 0.05

    _, _, projections, imaging_model = simulate.create_data(
        volume=volume_np,
        image_modality="sum_projection",
        rotation_case="sinusoidal",
        samples=samples,
        duration=0.2,
    )

    tomo = application.Tomography(
        volume_size=volume_np.shape,
        imaging_model=imaging_model,
        initial_volume="zeros",
        rotation_optim_case=rotation_optim_case,
        verbose=False,
    )
    return tomo, projections, volume_np


def test_tomography_initialization_creates_core_state(monkeypatch):
    tomo, projections, volume_np = _make_small_tomography()
    monkeypatch.setattr(
        application.erfl,
        "process_latent_space",
        lambda z, frames, **kwargs: _dummy_rotation_initialization(len(frames), "quaternion"),
    )

    tomo.initialize_parameters(
        projections,
        normalize=False,
        max_epochs=1,
        axis_sweep=False,
        peaks_period_range=[2, 6],
    )

    assert tomo.frames.shape[0] > 0
    assert tomo.frames.shape[1:] == projections.shape[1:]
    assert tomo.volume.shape == volume_np.shape
    assert tomo.rotation_params.shape[0] == tomo.frames.shape[0]
    assert tomo.translation_params.shape[0] == tomo.frames.shape[0]
    assert tomo.CH == projections.shape[1]


def test_tomography_forward_returns_projection_batch(monkeypatch):
    tomo, projections, _ = _make_small_tomography()
    monkeypatch.setattr(
        application.erfl,
        "process_latent_space",
        lambda z, frames, **kwargs: _dummy_rotation_initialization(len(frames), "quaternion"),
    )

    tomo.initialize_parameters(
        projections,
        normalize=False,
        max_epochs=1,
        axis_sweep=False,
        peaks_period_range=[2, 6],
    )

    idx = torch.arange(min(3, len(tomo.frames)), device=tomo.frames.device)
    yhat = tomo.forward(idx)

    assert yhat.ndim == 4
    assert yhat.shape[0] == len(idx)
    assert yhat.shape[1] == tomo.CH
    assert torch.isfinite(yhat).all()


def test_tomography_training_step_returns_finite_loss(monkeypatch):
    tomo, projections, _ = _make_small_tomography()
    monkeypatch.setattr(
        application.erfl,
        "process_latent_space",
        lambda z, frames, **kwargs: _dummy_rotation_initialization(len(frames), "quaternion"),
    )

    tomo.initialize_parameters(
        projections,
        normalize=False,
        max_epochs=1,
        axis_sweep=False,
        peaks_period_range=[2, 6],
    )

    idx = torch.arange(min(4, len(tomo.frames)), device=tomo.frames.device)
    monkeypatch.setattr(tomo, "log", lambda *args, **kwargs: None)
    loss = tomo.training_step(idx, 0)

    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_tomography_smoke_optimization_runs_without_error(monkeypatch):
    tomo, projections, _ = _make_small_tomography()
    monkeypatch.setattr(
        application.erfl,
        "process_latent_space",
        lambda z, frames, **kwargs: _dummy_rotation_initialization(len(frames), "quaternion"),
    )

    tomo.initialize_parameters(
        projections,
        normalize=False,
        max_epochs=1,
        axis_sweep=False,
        peaks_period_range=[2, 6],
    )

    idx = torch.arange(len(tomo.frames))
    trainer = dl.Trainer(
        max_epochs=1,
        accelerator="auto",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        log_every_n_steps=999999,
        limit_train_batches=1,
    )
    trainer.fit(tomo, DataLoader(idx, batch_size=len(idx), shuffle=False))

    volume = tomo.get_volume().detach().cpu()
    assert torch.isfinite(volume).all()
