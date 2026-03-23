import numpy as np

import tomodpdt.volumes as volumes


def test_precomputed_volumes_are_loaded_with_expected_shape():
    loaded = [
        volumes.VOL_GAUSS,
        volumes.VOL_SHELL,
        volumes.VOL_FLUO,
        volumes.VOL_GAUSS_MULT,
        volumes.VOL_RANDOM,
    ]

    for volume in loaded:
        assert isinstance(volume, np.ndarray)
        assert volume.shape == (volumes.SIZE, volumes.SIZE, volumes.SIZE)
        assert np.isfinite(volume).all()


def test_generate_3d_volume_validates_layer_count():
    try:
        volumes.generate_3d_volume(size=16, num_layers=2, layer_densities=[1.0])
    except ValueError as exc:
        assert "number of densities" in str(exc)
    else:
        raise AssertionError("generate_3d_volume should validate the layer count")


def test_sample_positions_3d_respects_requested_count():
    positions = volumes.sample_positions_3D(
        num_points=5,
        area_size=(32, 32, 32),
        min_distance=3,
        edge_margin=4,
    )

    assert positions.shape == (5, 3)
    assert np.all(positions >= 4)
    assert np.all(positions < 28)
