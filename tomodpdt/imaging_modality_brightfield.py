import numpy as np
import deeptrack as dt  # Assuming deeptrack is used for optics

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
    optics = dt.Brightfield(
        wavelength=wavelength,
        resolution=resolution,
        magnification=magnification,
        output_region=(0, 0, nsize, nsize),
        return_field=return_field
    )

    # Define simulation limits
    limits = np.array([[0, nsize], [0, nsize], [-nsize//2, nsize//2]])

    # Define fields
    padded_nsize = ((nsize + 31) // 32) * 32
    fields = np.ones((padded_nsize, padded_nsize), dtype=complex)

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

if __name__ == "__main__":

    nsize = 96
    optics_setup = setup_optics(nsize)

    #Create a random object, a cube with a smaller cube inside
    object = np.zeros((96, 96, 96)) + 1.33
    object[16:80, 16:80, 16:80] = 1.4
    object[32:64, 32:64, 32:64] = 1.5
    object[40:56, 40:56, 40:56] = 1.6

    #Scale object to 

    object = dt.Image(object)
    image = optics_setup['optics'].get(object, optics_setup['limits'], optics_setup['fields'], **optics_setup['filtered_properties'])

    print(image.shape)
    plt.imshow(image.imag)
    plt.colorbar()
    plt.show()
    plt.imshow(image.real)
    plt.colorbar()
    plt.show()
    
