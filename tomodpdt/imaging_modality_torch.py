import deeptrack as dt  # Assuming deeptrack is used for optics

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

def setup_optics(nsize, microscopy_regime='Brightfield', wavelength=532e-9, resolution=100e-9, magnification=10, return_field=True):
    """
    Set up the optical system, prepare simulation parameters, and compute the optical image.

    Args:
        nsize (int): Size of the volume grid.
        microscopy_regime (str): Microscopy regime (default 'Brightfield').
        wavelength (float): Wavelength of light in meters (default 532 nm).
        resolution (float): Optical resolution in meters (default 100 nm).
        magnification (float): Magnification factor (default 1).
        return_field (bool): Whether to return the optical field (default True).

    Returns:
        dict: A dictionary containing optics object, limits, fields, properties, and computed image.
    """

    # To enable case-insensitive comparison
    microscopy_regime = microscopy_regime.lower()

    # Define the optics
    if microscopy_regime == 'brightfield':
        optics = dt.Brightfield(
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, nsize, nsize),
            return_field=return_field
        )
    elif microscopy_regime == 'fluorescence':
        optics = dt.Fluorescence(
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, nsize, nsize),
            # return_field=return_field
        )
    elif microscopy_regime == 'darkfield':
        optics = dt.Darkfield(
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, nsize, nsize),
            #return_field=return_field
        )
    elif microscopy_regime == 'iscat':
        optics = dt.ISCAT(
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, nsize, nsize),
            return_field=return_field
        )

    # Define simulation limits
    limits = torch.tensor([[0, nsize], [0, nsize], [-nsize//2, nsize//2]])

    # Define fields
    padded_nsize = ((nsize + 31) // 32) * 32
    fields = torch.ones((padded_nsize, padded_nsize), dtype=torch.complex64)

    # Extract relevant properties from the optics
    properties = optics.properties()
    filtered_properties = {
        k: v for k, v in properties.items()
        if k in {'padding', 'output_region',
                 'NA', 'wavelength',
                 'refractive_index_medium', 'return_field'}
        }   

    return {
        'microscopy_regime': microscopy_regime,
        "optics": optics,
        "limits": limits,
        "fields": fields,
        "filtered_properties": filtered_properties,
        }


class imaging_model(nn.Module):
    def __init__(self, optics_setup):
        """
        Initialize the imaging model.

        Args:
            optics_setup (dict): A dictionary containing optics
            object, limits, fields, properties, and computed image.
        """

        super().__init__()
        self.microscopy_regime = optics_setup['microscopy_regime'].lower()
        self.optics = optics_setup['optics']
        self.limits = optics_setup['limits']
        self.fields = optics_setup['fields']
        self.filtered_properties = optics_setup['filtered_properties']

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, object):
        """
        Compute the optical image of an object.

        Args:
            object (torch.Tensor): Object to image.

        Returns:
            torch.Tensor: Optical image of the object.
        """

        # Move everything to the same device
        self.limits = self.limits.to(object.device)
        self.fields = self.fields.to(object.device)

        if self.microscopy_regime == 'brightfield' or self.microscopy_regime == 'darkfield' or self.microscopy_regime == 'iscat':
            return self.optics.get(object, self.limits, self.fields, **self.filtered_properties)
        
        elif self.microscopy_regime == 'fluorescence':
            return self.optics.get(object, self.limits, **self.filtered_properties)

        else:
            raise ValueError('Unknown microscopy regime')


class Dummy3d2d(nn.Module):
    def __init__(self, dim=-1):
        self.dim = dim
        super(Dummy3d2d, self).__init__()

    def forward(self, x):
        # Return projection of the 3D volume
        return x.sum(dim=self.dim)

def generate_3d_volume(size, num_layers, layer_densities):
    # Ensure that the number of densities matches the number of layers
    if len(layer_densities) != num_layers:
        raise ValueError("The number of densities must match the number of layers.")
    
    # Create an empty volume
    volume = np.zeros((size, size, size))

    # Define center and radius
    center = size // 2
    radius = size // 3

    # Generate grid
    x, y, z = np.meshgrid(np.arange(size), np.arange(size), np.arange(size))
    distance = np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)

    # Create layers with custom densities
    layer_radius = np.linspace(0, radius, num_layers + 1)

    for i in range(num_layers):
        mask = (distance < layer_radius[i+1]) & (distance >= layer_radius[i])
        volume[mask] = layer_densities[i]

    # Add noise that scales with the density of each layer
    noise = np.zeros_like(volume)
    for i in range(num_layers):
        mask = (distance < layer_radius[i+1]) & (distance >= layer_radius[i])
        layer_noise = np.random.normal(0, 0.2 * layer_densities[i], volume.shape)  # Scale noise by density
        noise[mask] = layer_noise[mask]
    
    volume += noise
    volume = np.clip(volume, 0, np.max(layer_densities))  # Keep values within the range of layer densities

    # Apply slight smoothing to the structure
    volume = scipy.ndimage.gaussian_filter(volume, sigma=1)

    return volume

if __name__ == "__main__":

    nsize = 64
    optics_setup = setup_optics(nsize, microscopy_regime='fluorescence')
    im_model = imaging_model(optics_setup)

    RI_RANGE = (1.333, 1.36)

    # Specify custom densities for each layer
    layer_densities = [3, 2, 1]  # You can modify this list to set custom values for each layer
    num_layers = len(layer_densities)

    # Generate the volume
    volume = generate_3d_volume(nsize, num_layers, layer_densities)

    # Normalize the volume for visualization
    object = (volume - volume.min()) / (volume.max() - volume.min()) * (RI_RANGE[1] - RI_RANGE[0]) + RI_RANGE[0]
    
    # Show 3 projections of the volume
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.title('XY projection')
    plt.imshow(object.sum(axis=0))
    plt.colorbar()
    plt.subplot(132)
    plt.title('XZ projection')
    plt.imshow(object.sum(axis=1))
    plt.colorbar()
    plt.subplot(133)
    plt.title('YZ projection')
    plt.imshow(object.sum(axis=2))
    plt.colorbar()
    plt.show()
    
    object = torch.zeros((nsize, nsize, nsize))

    # Add 5 random dots
    for i in range(10):
        x, y, z = np.random.randint(0, nsize, 3)
        object[x, y, 32] = 1

    object = torch.tensor(object).to('cuda')

    image = im_model(object)

    if image.device.type == 'cuda':
        image = image.cpu()

    
    try:
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
    
