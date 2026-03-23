import numpy as np
import pytest
import torch


simulate = pytest.importorskip("tomodpdt.simulate")


def test_create_data_sum_projection_returns_consistent_shapes():
    volume, quaternions, projections, imaging_model = simulate.create_data(
        volume_case="gaussian",
        image_modality="sum_projection",
        rotation_case="sinusoidal",
        samples=5,
        duration=0.1,
    )

    assert isinstance(volume, torch.Tensor)
    assert isinstance(quaternions, torch.Tensor)
    assert isinstance(projections, torch.Tensor)
    assert volume.ndim == 3
    assert quaternions.shape == (5, 4)
    assert projections.shape == (5, 1, volume.shape[0], volume.shape[1])
    assert imaging_model.microscopy_regime == "sum_projection"


def test_create_data_accepts_explicit_rotation_and_translation_inputs():
    samples = 4
    quaternions = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (samples, 1))
    translations = np.zeros((samples, 3), dtype=np.float32)

    volume, returned_quaternions, projections, _ = simulate.create_data(
        volume_case="gaussian",
        image_modality="sum_projection",
        rotation_case=quaternions,
        translations=translations,
    )

    assert returned_quaternions.shape == (samples, 4)
    assert projections.shape == (samples, 1, volume.shape[0], volume.shape[1])


def test_create_data_rejects_unknown_modes():
    with pytest.raises(ValueError, match="Unknown volume case"):
        simulate.create_data(volume_case="does_not_exist", image_modality="sum_projection", samples=2)

    with pytest.raises(ValueError, match="Unknown imaging modality"):
        simulate.create_data(volume_case="gaussian", image_modality="not_a_mode", samples=2)

    with pytest.raises(ValueError, match="Unknown rotation case"):
        simulate.create_data(
            volume_case="gaussian",
            image_modality="sum_projection",
            rotation_case="not_a_rotation",
            samples=2,
        )
