import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tomodpdt.image_modalities_dt as dt
except:
    import image_modalities_dt as dt

import deeptrack
from deeptrack.backend.units import create_context
from deeptrack.aberrations import SphericalAberration


def setup_optics(
        shape=None,
        nsize=None,
        padding_xy=64,
        microscopy_regime='Brightfield',
        NA=0.7,
        wavelength=532e-9,
        resolution=100e-9,
        magnification=1,
        return_field=True):
    """
    Set up optical system for arbitrary 3D volume with metadata using (nx, ny, nz) order.
    """
    microscopy_regime = microscopy_regime.lower()

    # Determine shape
    if shape is not None:
        if len(shape) != 3:
            raise ValueError("`shape` must be (nx, ny, nz)")
        nx, ny, nz = shape
    elif nsize is not None:
        nx = ny = nz = int(nsize)
    else:
        raise ValueError("Provide either `shape` or `nsize`")

    # Padded sizes
    padded_nx = nx + 2 * padding_xy
    padded_ny = ny + 2 * padding_xy
    padded_nz = nz  # do not pad z

    # Define optics
    if microscopy_regime == 'brightfield':
        optics = dt.Brightfield(
            NA=NA,
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, padded_nx, padded_ny),
            return_field=return_field
        )
    elif microscopy_regime == 'fluorescence':
        optics = dt.Fluorescence(
            NA=NA,
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, padded_nx, padded_ny),
            pupil=SphericalAberration()
        )
        return_field = False
    elif microscopy_regime == 'darkfield':
        optics = dt.Darkfield(
            NA=NA,
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, padded_nx, padded_ny)
        )
        return_field = False
    elif microscopy_regime == 'iscat':
        optics = dt.ISCAT(
            NA=NA,
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, padded_nx, padded_ny),
            return_field=return_field
        )
    else:
        raise ValueError(f"Unknown microscopy_regime: {microscopy_regime}")

    # Limits in (x, y, z) order
    limits = torch.tensor([
        [0, padded_nx],
        [0, padded_ny],
        [-nz / 2, nz / 2]
    ], dtype=torch.float32)

    # Precompute fields
    padded_xy_for_fft = 2 * ((max(padded_nx, padded_ny) + 31) // 32) * 32
    fields = torch.ones((padded_xy_for_fft, padded_xy_for_fft), dtype=torch.complex64)

    # Filtered properties
    properties = optics.properties()
    filtered_properties = {
        k: v for k, v in properties.items()
        if k in {'padding', 'output_region', 'NA', 'wavelength', 'refractive_index_medium', 'return_field'}
    }

    return {
        'microscopy_regime': microscopy_regime,
        'optics': optics,
        'limits': limits,
        'fields': fields,
        'filtered_properties': filtered_properties,
        'padding_xy': padding_xy,
        'resolution': resolution,
        'shape': (nx, ny, nz),  # consistent with limits
    }


class imaging_model(nn.Module):
    """
    Imaging model using DeepTrack, fully consistent with (nx, ny, nz) metadata.
    Input volumes are still (nz, ny, nx) internally.
    """

    def __init__(self, optics_setup):
        super().__init__()
        self.microscopy_regime = optics_setup['microscopy_regime'].lower()
        self.optics = optics_setup['optics']
        self.limits = optics_setup['limits']
        self.fields = optics_setup['fields']
        self.filtered_properties = optics_setup['filtered_properties']
        self.padding_xy = int(optics_setup['padding_xy'])
        self.resolution = optics_setup['resolution']
        self.nx, self.ny, self.nz = optics_setup['shape']  # metadata

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.padding_value = 0.0
        self.forward_case = 'vmap' if self.microscopy_regime != 'fluorescence' else 'loop'

    def forward(self, obj, vmap=True):
        self.limits = self.limits.to(obj.device)
        self.fields = self.fields.to(obj.device)

        # single volume
        if obj.dim() == 3:
            return self.imaging_step(obj)

        # single volume with batch dim 1
        if obj.dim() == 4 and obj.size(0) == 1:
            return self.imaging_step(obj.squeeze(0)).unsqueeze(0)

        # batch processing
        if self.forward_case == 'vmap' and vmap:
            imaging_vmap = torch.vmap(self.imaging_step, in_dims=0)
            return imaging_vmap(obj)
        else:
            return torch.stack([self.imaging_step(sample) for sample in obj])

    def imaging_step(self, obj):
        obj = obj.to(self.device)

        with deeptrack.units.context(
            create_context(
                xpixel=self.resolution,
                ypixel=self.resolution,
                zpixel=self.resolution,
                xscale=1,
                yscale=1,
                zscale=1,
            )
        ):

            # pad x/y if needed
            if self.padding_xy > 0:
                obj = F.pad(
                    obj.permute(2, 1, 0),  # (nx, ny, nz)
                    (0, 0, self.padding_xy, self.padding_xy, self.padding_xy, self.padding_xy),
                    mode='constant', value=self.padding_value
                ).permute(2, 1, 0)

            # brightfield, darkfield, iscat
            if self.microscopy_regime in {'brightfield', 'darkfield', 'iscat'}:
                image = self.optics.get(obj, self.limits, self.fields, **self.filtered_properties)

            # fluorescence
            elif self.microscopy_regime == 'fluorescence':
                if obj.sum() == 0:
                    c1, c2, c3 = obj.shape
                    # center voxel (compute dynamically)
                    cz = c1 // 2
                    cy = c2 // 2
                    cx = c3 // 2
                    obj[cz, cy, cx] = 1e-7
                image = self.optics.get(obj, self.limits, **self.filtered_properties)
            else:
                raise ValueError('Unknown microscopy regime')

        # remove padding
        if self.padding_xy > 0:
            image = image[self.padding_xy:-self.padding_xy, self.padding_xy:-self.padding_xy]

        return image._value


class Sum3d2d(nn.Module):
    def __init__(self, dim=-1):
        self.dim = dim
        self.microscopy_regime = 'sum_projection'
        super(Sum3d2d, self).__init__()

    def forward(self, x):
        return x.sum(dim=self.dim, keepdim=True)


class SumAvgWeighted3d2d(nn.Module):
    def __init__(self, dim=-1):
        self.dim = dim
        self.microscopy_regime = 'sum_projection_avg_weighted'
        super(SumAvgWeighted3d2d, self).__init__()

    def forward(self, x):
        self.weight_along_dim = torch.linspace(1, 0, x.size()[self.dim]).to(x.device)
        w_object = x * self.weight_along_dim
        return w_object.sum(dim=self.dim, keepdim=True)


class PropagationImagingModel(nn.Module):
    """
    Simple scalar diffraction forward model for tomography-style imaging.
    No rotation or translation of the object — just direct field propagation.
    """

    def __init__(self, 
                 nx, 
                 ny, 
                 nz, 
                 dx=100e-9, 
                 dy=100e-9, 
                 dz=100e-9, 
                 wavelength=532e-9, 
                 n0=1.33
                 ):
        super().__init__()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.wavelength = wavelength
        self.n0 = n0
        self.k0 = 2 * torch.pi / wavelength
        self.device = torch.device(self.device)

    
        # --- Precompute the frequency-domain propagation kernels ---
        kx, ky = torch.meshgrid(
            torch.fft.fftfreq(nx, d=dx) * 2 * torch.pi,
            torch.fft.fftfreq(ny, d=dy) * 2 * torch.pi,
            indexing='ij'
        )

        se = (self.k0 * n0)**2 - kx**2 - ky**2
        kz = torch.sqrt(se * (se > 0))
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.register_buffer("kz", kz)

        # Propagation kernel for one dz step
        self.register_buffer("K", torch.exp(1j * kz * dz))

        self.microscopy_regime = "scalar_propagation"
        self.forward_case = "loop"

    def forward(self, volume):
        """
        Forward propagation through the refractive index perturbation volume.
        Input: volume (nz, nx, ny) or (B, nz, nx, ny)
        Output: complex field (B, nx, ny)
        """
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)
        B = volume.shape[0]

        E = torch.ones((B, self.nx, self.ny), dtype=torch.complex64, device=self.device)

        # Propagate slice by slice
        for i in range(self.nz):
            E = torch.fft.ifft2(torch.fft.fft2(E) * self.K)
            E = E * torch.exp(1j * volume[:, i] * self.k0 * self.dz)

        # Free-space propagation after the sample (optional)
        E = torch.fft.ifft2(
            torch.fft.fft2(E) * torch.exp(1j * self.kz * self.dz * self.nz),
            dim=(1, 2)
        )

        phase_term = torch.exp(
            (-1j * torch.tensor(self.nz * self.dz * self.k0 * self.n0, dtype=torch.float32, device=self.device))
        )
        return E * phase_term


if __name__ == "__main__":
    import numpy as np


    optics_setup = setup_optics(nsize=64, padding_xy=64, microscopy_regime='fluorescence')
    im_model = imaging_model(optics_setup)

    
    object2 = np.load('../test_data/vol_gauss_mult.npy') - 1.33
    object = np.load('../test_data/vol_gauss_mult.npy') - 1.33   
    object = torch.tensor(object).to('cuda')
    object2 = torch.tensor(object2).to('cuda')

    object = object[8:, 16:, :]
    object2 = object2[8:, 16:, :]

    optics_setup = setup_optics(shape=object.shape, padding_xy=64, microscopy_regime='brightfield')
    im_model = imaging_model(optics_setup)

    # Add random noise
    object_8 = torch.stack([object for _ in range(8)]+[object2 for _ in range(8)])
    
    import time

    #Track gradient
    object_8.requires_grad = True
    
    start = time.time()
    image16 = im_model(object_8)
    print('Time taken:', time.time() - start)

    #Check gradient
    image16.real.sum().backward()
    #print('Gradient:', object_16.grad)

    start = time.time()
    for object in object_8:
        image = im_model(object).detach()
    print('Time taken:', time.time() - start)

    if image.device.type == 'cuda':
        image = image.cpu()

    
    try:
        import matplotlib.pyplot as plt

        im = image.imag
        plt.figure(figsize=(6, 6))
        plt.title('Imaginary part')
        plt.imshow(im)
        plt.colorbar()
        plt.show()
    except:
        pass

    try:
        im = image.real
        plt.figure(figsize=(6, 6))
        plt.title('Real part')
        plt.imshow(im)
        plt.colorbar()
        plt.show()
    except AttributeError:
        pass

