import deeptrack as dt  # Assuming deeptrack is used for optics

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def setup_optics(nsize, microscopy_regime='Brightfield', wavelength=532e-9, resolution=100e-9, magnification=1, return_field=True):
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
        optics = dt.optics_torch.Brightfield(
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, nsize, nsize),
            return_field=return_field
        )
    elif microscopy_regime == 'fluorescence':
        optics = dt.optics_torch.Fluorescence(
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, nsize, nsize),
            # return_field=return_field
        )
    elif microscopy_regime == 'darkfield':
        optics = dt.optics_torch.Darkfield(
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, nsize, nsize),
            #return_field=return_field
        )
    elif microscopy_regime == 'iscat':
        optics = dt.optics_torch.ISCAT(
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

        # Move everything to the same device
        self.limits = self.limits.to(self.device)
        self.fields = self.fields.to(self.device)

    def forward(self, object):
        """
        Compute the optical image of an object.

        Args:
            object (torch.Tensor): Object to image.

        Returns:
            torch.Tensor: Optical image of the object.
        """

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


if __name__ == "__main__":

    nsize = 96
    optics_setup = setup_optics(nsize, microscopy_regime='darkfield')

    im_model = imaging_model(optics_setup)

    # Create a random object, a cube with a smaller cube inside
    object = torch.zeros((96, 96, 96)) + 1.33
    object[16:80, 16:80, 16:80] = 1.4
    object[32:64, 32:64, 32:64] = 1.5
    object[40:56, 40:56, 40:56] = 1.6
    
    object = object.to(torch.device('cuda'))

    image = im_model(object)

    print(image.cpu().shape)
    plt.imshow(image.cpu().imag)
    plt.colorbar()
    plt.show()
    plt.imshow(image.cpu().real)
    plt.colorbar()
    plt.show()
    
    image2 = torch.concatenate((image._value.real, image._value.imag), axis=-1)
    image2 = torch.swapaxes(image2, 0, 2)
