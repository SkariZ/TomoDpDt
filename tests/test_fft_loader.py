import torch

from tomodpdt.fft_loader import (
    create_circular_mask,
    create_ellipse_mask,
    data_to_real,
    field_to_vec,
    field_to_vec_multi,
    real_to_imag,
    vec_to_field,
    vec_to_field_multi,
)


def test_mask_generators_return_expected_shape_and_dtype():
    circle = create_circular_mask(16, 12, radius=4)
    ellipse = create_ellipse_mask(16, 12, percent=0.25)

    assert circle.shape == (16, 12)
    assert ellipse.shape == (16, 12)
    assert circle.dtype == torch.bool
    assert ellipse.dtype == torch.bool
    assert circle.any()
    assert ellipse.any()


def test_real_complex_conversion_roundtrip():
    field = torch.randn(8, 8, dtype=torch.complex64)

    as_real = data_to_real(field)
    reconstructed = real_to_imag(as_real)

    assert as_real.shape == (8, 8, 2)
    assert torch.allclose(reconstructed, field)


def test_field_vector_roundtrip_with_explicit_mask():
    field = torch.randn(8, 8, dtype=torch.complex64)
    mask = create_circular_mask(8, 8, radius=2)

    vec = field_to_vec(field, pupil_radius=2, mask=mask)
    reconstructed = vec_to_field(vec, pupil_radius=2, shape=(8, 8), mask=mask)
    vec_reconstructed = field_to_vec(reconstructed, pupil_radius=2, mask=mask)

    assert vec.ndim == 1
    assert reconstructed.shape == (8, 8)
    assert torch.allclose(vec_reconstructed, vec, atol=1e-5, rtol=1e-5)


def test_multi_field_vector_roundtrip_with_explicit_mask():
    fields = torch.randn(3, 8, 8, dtype=torch.complex64)
    mask = create_ellipse_mask(8, 8, percent=0.25)

    vecs = field_to_vec_multi(fields, pupil_radius=2, mask=mask)
    reconstructed = vec_to_field_multi(vecs, pupil_radius=2, shape=(8, 8), mask=mask)
    vecs_reconstructed = field_to_vec_multi(reconstructed, pupil_radius=2, mask=mask)

    assert vecs.shape[0] == 3
    assert reconstructed.shape == fields.shape
    assert torch.allclose(vecs_reconstructed, vecs, atol=1e-5, rtol=1e-5)
