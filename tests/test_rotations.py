import numpy as np
import pytest

import tomodpdt.rotations as rotations


@pytest.mark.parametrize(
    "generator,kwargs",
    [
        (rotations.generate_sinusoidal_quaternion, {"samples": 32, "duration": 0.5}),
        (rotations.generate_random_sinusoidal_quaternion, {"samples": 32, "duration": 0.5}),
        (rotations.generate_noisy_sinusoidal_quaternion, {"samples": 32, "duration": 0.5}),
        (rotations.generate_smooth_varying_quaternion, {"samples": 32, "duration": 0.5}),
        (rotations.generate_random_varying_quaternion, {"samples": 32, "duration": 0.5}),
        (rotations.generate_integrated_angular_velocity_quaternion, {"samples": 32, "duration": 0.5}),
        (rotations.generate_axis_switching_quaternion, {"samples": 32, "duration": 0.5}),
        (rotations.generate_ou_quaternion, {"samples": 32, "duration": 0.5, "seed": 123}),
    ],
)
def test_rotation_generators_return_unit_quaternions(generator, kwargs):
    quaternions = generator(**kwargs)

    assert quaternions.ndim == 2
    assert quaternions.shape[1] == 4
    norms = np.linalg.norm(quaternions, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_normalize_quaternions_to_identity_sets_first_pose_to_identity():
    quaternions = rotations.generate_random_varying_quaternion(samples=16, duration=0.5)

    normalized = rotations.normalize_quaternions_to_identity(quaternions)

    assert normalized.shape == quaternions.shape
    assert np.allclose(normalized[0], np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-5)
