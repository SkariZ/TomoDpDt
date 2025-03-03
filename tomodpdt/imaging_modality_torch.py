import deeptrack as dt  # Assuming deeptrack is used for optics

import torch
import torch.nn as nn
from deeptrack.backend.units import (
    create_context,
    get_active_scale,
    get_active_voxel_size,
)

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

def setup_optics(nsize, padding_xy=64, microscopy_regime='Brightfield', NA=0.7, wavelength=532e-9, resolution=100e-9, magnification=10, return_field=True):
    """
    Set up the optical system, prepare simulation parameters, and compute the optical image.

    Args:
        nsize (int): Size of the volume grid.
        padding_xy (int): Padding in the xy direction (default 64).
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

    # Add padding to the size
    nsize = nsize + padding_xy * 2

    # Define the optics
    if microscopy_regime == 'brightfield':
        optics = dt.Brightfield(
            NA=NA,
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, nsize, nsize),
            return_field=return_field
        )
    elif microscopy_regime == 'fluorescence':
        optics = dt.Fluorescence(
            NA=NA,
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, nsize, nsize),
            # return_field=return_field
        )
    elif microscopy_regime == 'darkfield':
        optics = dt.Darkfield(
            NA=NA,
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, nsize, nsize),
            #return_field=return_field
        )
    elif microscopy_regime == 'iscat':
        optics = dt.ISCAT(
            NA=NA,
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, nsize, nsize),
            return_field=return_field
        )

    # Define simulation limits
    limits = torch.tensor([[0, nsize], [0, nsize], [-(nsize-padding_xy*2)//2, (nsize-padding_xy*2)//2]])

    # Define fields
    padded_nsize = 2*((nsize + 31) // 32) * 32
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
        "padding_xy": padding_xy
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
        self.padding_xy = optics_setup['padding_xy']

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Unpack the properties
        self.NA = self.filtered_properties['NA']
        self.wavelength = self.filtered_properties['wavelength']
        self.refractive_index_medium = self.filtered_properties['refractive_index_medium']
        self.padding = self.filtered_properties['padding']
        self.output_region = self.filtered_properties['output_region']
        self.return_field = self.filtered_properties['return_field'] if 'return_field' in self.filtered_properties else False

        #
        self.padding_value = 1.33 if self.microscopy_regime == 'brightfield' or self.microscopy_regime == 'darkfield' or self.microscopy_regime == 'iscat' else 0

    def forward(self, object, forward_case='loop'):
        self.limits = self.limits.to(object.device)
        self.fields = self.fields.to(object.device)

        if object.dim() == 3:
            return self.imaging_step(object)
        
        if forward_case == 'vmap':
            imaging_vmap = torch.vmap(self.imaging_step, in_dims=0)
            # Do a batch processing with multiple objects
            return imaging_vmap(object)
        
        elif forward_case == 'loop':
            return torch.stack([self.imaging_step(sample) for sample in object])
            
    def imaging_step(self, object):
        """
        Compute the optical image of an object.

        Args:
            object (torch.Tensor): Object to image.

        Returns:
            torch.Tensor: Optical image of the object.
        """
 
        with dt.units.context(
            create_context(
                xpixel=1e-7,
                ypixel=1e-7,
                zpixel=1e-7,
                xscale=1,
                yscale=1,
                zscale=1,
                )
        ):

            if self.padding_xy > 0:
                object = torch.nn.functional.pad(
                    object.permute(2, 1, 0), (self.padding_xy, self.padding_xy, self.padding_xy, self.padding_xy, 0, 0), mode='constant', value=self.padding_value
                    ).permute(2, 1, 0)
            if self.microscopy_regime == 'brightfield' or self.microscopy_regime == 'darkfield' or self.microscopy_regime == 'iscat':
                image = self.optics.get(object, self.limits, self.fields, **self.filtered_properties)

            elif self.microscopy_regime == 'fluorescence':
                image = self.optics.get(object, self.limits, **self.filtered_properties)

            else:
                raise ValueError('Unknown microscopy regime')

        if self.padding_xy > 0:
            image = image[self.padding_xy:-self.padding_xy, self.padding_xy:-self.padding_xy]
            
        return image._value

class Dummy3d2d(nn.Module):
    def __init__(self, dim=-1):
        self.dim = dim
        self.microscopy_regime = 'sum_projection'
        super(Dummy3d2d, self).__init__()

    def forward(self, x):
        # Return projection of the 3D volume
        return x.sum(dim=self.dim, keepdim=True)

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
    optics_setup = setup_optics(nsize, padding_xy=64, microscopy_regime='fluorescence')
    im_model = imaging_model(optics_setup)

    RI_RANGE = (1.33, 1.45)

    # Specify custom densities for each layer
    layer_densities = [1, 2, 3]  # You can modify this list to set custom values for each layer
    num_layers = len(layer_densities)

    # Generate the volume
    volume = generate_3d_volume(nsize, num_layers, layer_densities)
    volume2 = generate_3d_volume(nsize, 1, [1])

    # Normalize the volume for visualization
    object = (volume - volume.min()) / (volume.max() - volume.min()) * (RI_RANGE[1] - RI_RANGE[0]) + RI_RANGE[0]
    object2 = (volume2 - volume2.min()) / (volume2.max() - volume2.min()) * (RI_RANGE[1] - RI_RANGE[0]) + RI_RANGE[0]

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
    
    object = torch.tensor(object).to('cuda')
    object2 = torch.tensor(object2).to('cuda')

    import volumes as V
    object = V.VOL_FLUO
    object2 = V.VOL_FLUO
    object = torch.tensor(object).to('cuda')
    object2 = torch.tensor(object2).to('cuda')

    object_16 = torch.stack([object for _ in range(8)]+[object2 for _ in range(8)])

    import time

    #Track gradient
    object_16.requires_grad = True

    start = time.time()
    image16 = im_model(object_16)
    print('Time taken:', time.time() - start)

    #Check gradient
    image16.real.sum().backward()
    #print('Gradient:', object_16.grad)

    start = time.time()
    for i in range(16):
        image = im_model(object)
    print('Time taken:', time.time() - start)

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
    
