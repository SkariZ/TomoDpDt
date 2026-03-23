import torch
import torch.fft as fft


def create_circular_mask(h, w, center=None, radius=None):
    """
    Creates a circular mask.

    Input:
        h : height
        w : width
        center : Define center of image. If None -> middle is used.
        radius : radius of circle.
    Output:
        Circular mask.

    """
    
    if center is None:  # use the middle of the image
        center = (int(h/2), int(w/2))
        
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    
    X, Y = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    dist_from_center = torch.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    mask = dist_from_center <= radius
    return mask


def create_ellipse_mask(h, w, center=None, radius_h=None, radius_w=None, percent=0.05):
    """
    Creates an ellipsoid mask.

    Input:
        h : height
        w : width
        center : Define center of the image. If None, the middle is used.
        radius_h : Radius in height
        radius_w : Radius in width
        percent : If radius_h or radius_w is not defined, use this percentage factor instead.
    Output:
        Ellipsoid mask.
    """

    if center is None:
        center_w, center_h = int(w/2), int(h/2)
    else:
        center_w, center_h = center[0], center[1]

    if radius_h is None and radius_w is None:
        if percent is not None:
            radius_w, radius_h = int(percent*w), int(percent*h)
        else:
            radius_w, radius_h = int(0.25*w/2), int(0.25*h/2)  # Ellipsoid of this size. To get some output

    x_indices, y_indices = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    
    # Calculate the equation of the ellipse
    ellipse_equation = (
        ((x_indices - center_h) / radius_h) ** 2 +
        ((y_indices - center_w) / radius_w) ** 2
    )

    # Set pixels within the ellipse to 1
    mask = ellipse_equation <= 1
    return mask


def data_to_real(img):
    """
    Transforms a complex image to a real image.

    Input:
        img : complex image (torch complex tensor)
    Output:
        Real image (torch float tensor with 2 channels).
    """
    image = torch.zeros((*img.shape, 2), dtype=torch.float32).to(img.device)
    image[..., 0] = img.real
    image[..., 1] = img.imag
    return image


def real_to_imag(img):
    """
    Transforms a real image with 2 channels to a complex image.

    Input:
        img : real image (torch float tensor with 2 channels)
    Output:
        Complex image (torch complex tensor).
    """
    return img[..., 0] + 1j * img[..., 1]


def field_to_vec(field, pupil_radius, mask=None, mask_shape='ellipse'):
    """
    Transforms a field to a vector given a pupil radius.

    Input:
        field : complex tensor representing the field
        pupil_radius : radius of the pupil
        mask : optional circular mask
    Output:
        Vector of complex numbers (torch complex tensor).
    """
    # if field is not a tensor, convert to tensor
    if not torch.is_tensor(field):
        field = torch.tensor(field, dtype=torch.complex64)

    if not torch.is_complex(field):
        field = field[..., 0] + 1j * field[..., 1]

    h, w = field.shape
    fft_image = fft.fftshift(fft.fft2(field))

    if mask is None:
        if mask_shape == 'circle':
            mask = create_circular_mask(h, w, radius=pupil_radius)
        elif mask_shape == 'ellipse':
            mask = create_ellipse_mask(h, w, percent=pupil_radius/h)

    return fft_image[mask]


def field_to_vec_multi(fields, pupil_radius, mask=None, mask_shape='ellipse'):
    """
    Transforms multiple fields to vectors given a pupil radius.

    Input:
        fields : complex tensor representing multiple fields
        pupil_radius : radius of the pupil
        mask : optional circular mask
    Output:
        List of vectors of complex numbers (list of torch complex tensors).
    """

    # if fields is not a tensor, convert to tensor
    if not torch.is_tensor(fields):
        fields = torch.tensor(fields, dtype=torch.complex64)

    if not torch.is_complex(fields):
        fields = fields[..., 0] + 1j * fields[..., 1]

    _, h, w = fields.shape

    if mask is None:
        if mask_shape == 'circle':
            mask = create_circular_mask(h, w, radius=pupil_radius)
        elif mask_shape == 'ellipse':
            mask = create_ellipse_mask(h, w, percent=pupil_radius/h)
    mask = mask.type(torch.bool).to(fields.device)

    vectors = []
    for field in fields:
        fft_image = fft.fftshift(fft.fft2(field))
        vectors.append(fft_image[mask])

    return torch.stack(vectors, dim=0)


def vec_to_field(vec, pupil_radius, shape, mask=None, mask_shape='ellipse', to_real=False):
    """
    Transforms a vector to a field given pupil radius and shape.

    Input:
        vec : vector of complex numbers
        pupil_radius : radius of the pupil
        shape : shape of the resulting field
        mask : optional circular mask
        to_real : flag indicating whether to convert to real image
    Output:
        Complex tensor representing the field.
    """

    # if vecs is not a tensor, convert to tensor
    if not torch.is_tensor(vec):
        vec = torch.tensor(vec, dtype=torch.complex64)

    if mask is None:
        if mask_shape == 'circle':
            mask = create_circular_mask(shape[0], shape[1], radius=pupil_radius)
        elif mask_shape == 'ellipse':
            mask = create_ellipse_mask(shape[0], shape[1], percent=pupil_radius/shape[0])
    mask = mask.type(torch.complex64).to(vec.device)

    mask[mask == 1] = vec

    field = fft.ifft2(fft.ifftshift(mask))
    if to_real:
        field = data_to_real(field)

    return field


def vec_to_field_multi(vecs, pupil_radius, shape, mask=None, mask_shape='ellipse', to_real=False):
    """
    Transforms multiple vectors to fields given pupil radius and shape.

    Input:
        vecs : list of vectors of complex numbers
        pupil_radius : radius of the pupil
        shape : shape of the resulting fields
        mask : optional circular mask
        to_real : flag indicating whether to convert to real images
    Output:
        List of complex tensors representing the fields.
    """
    
    # if vecs is not a tensor, convert to tensor
    if not torch.is_tensor(vecs):
        vecs = torch.tensor(vecs, dtype=torch.complex64)

    if mask is None:
        if mask_shape == 'circle':
            mask = create_circular_mask(shape[0], shape[1], radius=pupil_radius)
        elif mask_shape == 'ellipse':
            mask = create_ellipse_mask(shape[0], shape[1], percent=pupil_radius/shape[0])

    mask = mask.type(torch.complex64).to(vecs.device)

    fields = []
    for vec in vecs:
        mm = mask.clone()
        mm[mm == 1] = vec
        fields.append(fft.ifft2(fft.ifftshift(mm)))

    fields = torch.stack(fields, dim=0)

    if to_real:
        fields = torch.stack([data_to_real(f) for f in fields], dim=0)

    return fields
