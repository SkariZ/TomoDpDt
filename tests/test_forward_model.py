import torch

from tomodpdt.forward_module import ForwardModelSimple


def test_quaternion_to_rotation_matrix_identity():
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    rotation = ForwardModelSimple.quaternion_to_rotation_matrix(q)

    assert rotation.shape == (3, 3)
    assert torch.allclose(rotation, torch.eye(3), atol=1e-6)


def test_forward_model_identity_projection_matches_direct_sum():
    model = ForwardModelSimple(nx=4, ny=4, nz=4, dim=2, device=torch.device("cpu"))
    volume = torch.arange(64, dtype=torch.float32).reshape(4, 4, 4)
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    projection = model(volume, quaternions)
    # ForwardModelSimple builds its sampling grid in (z, y, x) order, so the
    # identity pose preserves the module's internal axis convention rather than
    # the raw tensor's last-axis projection.
    expected = volume.sum(dim=0).transpose(0, 1).unsqueeze(0)

    assert projection.shape == (1, 4, 4)
    assert torch.allclose(projection, expected, atol=1e-4)
