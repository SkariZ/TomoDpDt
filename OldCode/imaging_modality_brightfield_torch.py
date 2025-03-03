import deeptrack as dt  # Assuming deeptrack is used for optics

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def setup_optics(nsize, wavelength=532e-9, resolution=100e-9, magnification=1, return_field=True):
    """
    Set up the optical system, prepare simulation parameters, and compute the optical image.

    Args:
        nsize (int): Size of the volume grid.
        wavelength (float): Wavelength of light in meters (default 532 nm).
        resolution (float): Optical resolution in meters (default 100 nm).
        magnification (float): Magnification factor (default 1).
        return_field (bool): Whether to return the optical field (default True).

    Returns:
        dict: A dictionary containing optics object, limits, fields, properties, and computed image.
    """

    # Define the optics
    optics = dt.optics_torch.Brightfield(
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
        if k in {'padding', 'output_region', 'NA', 'wavelength', 
                 'refractive_index_medium', 'return_field'}
        }   

    return {
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
            optics_setup (dict): A dictionary containing optics object, limits, fields, properties, and computed image.
        """

        super().__init__()
        self.optics = optics_setup['optics']
        self.limits = optics_setup['limits']
        self.fields = optics_setup['fields']
        self.filtered_properties = optics_setup['filtered_properties']

    def forward(self, object):
        """
        Compute the optical image of an object.

        Args:
            object (torch.Tensor): Object to image.

        Returns:
            torch.Tensor: Optical image of the object.
        """
        # Move evertything to the same device
        #self.limits = self.limits.to(object.device)
        #self.fields = self.fields.to(object.device)
        #or key in self.filtered_properties:
        #   self.filtered_properties[key] = self.filtered_properties[key].to(object.device)
        
        # Move everything to the same device
        self.limits = self.limits.to(object.device)
        self.fields = self.fields.to(object.device)

        return self.optics.get(object, self.limits, self.fields, **self.filtered_properties)
    
if __name__ == "__main__":

    nsize = 96
    optics_setup = setup_optics(nsize)

    imaging_model = imaging_model(optics_setup)

    #Create a random object, a cube with a smaller cube inside
    object = torch.zeros((96, 96, 96)) + 1.33
    object[16:80, 16:80, 16:80] = 1.4
    object[32:64, 32:64, 32:64] = 1.5
    object[40:56, 40:56, 40:56] = 1.6
    
    object = object.to(torch.device('cuda'))

    image = imaging_model(object)

    print(image.cpu().shape)
    plt.imshow(image.cpu().imag)
    plt.colorbar()
    plt.show()
    plt.imshow(image.cpu().real)
    plt.colorbar()
    plt.show()
    
    image2 = torch.concatenate((image._value.real, image._value.imag), axis=-1)
    image2 = torch.swapaxes(image2, 0, 2)
