"""
This module contains the implementation of pytorch-based optics which inherit from the original deeptrack optics.
It includes the following classes:

- Optics: Base class for all optical systems.
- Fluorescence: Class for simulating fluorescence microscopy.
- Brightfield: Class for simulating brightfield microscopy.
- ISCAT: Class for simulating Interferometric Scattering (ISCAT) microscopy.
"""

from deeptrack.optics import Optics as OriginalOptics

from typing import Any, Dict, Union, Iterable
from deeptrack.backend.units import (
    ConversionTable,
    get_active_voxel_size,
)

from deeptrack.features import Feature
from deeptrack.image import Image
from deeptrack.types import ArrayLike

from deeptrack import units as u

import numpy as np
import torch


_FASTEST_SIZES = [0]
for n in range(1, 10):
    _FASTEST_SIZES += [2 ** a * 3 ** (n - a - 1) for a in range(n)]
_FASTEST_SIZES = np.sort(_FASTEST_SIZES)


def pad_image_to_fft(
    image: Union[torch.Tensor, np.ndarray],
    axes: Iterable[int] = (0, 1),
) -> Union[torch.Tensor, np.ndarray]:
    """Pads an image to optimize Fast Fourier Transform (FFT) performance.

    This function pads an image by adding zeros to the end of specified axes 
    so that their lengths match the nearest larger size in `_FASTEST_SIZES`. 
    These sizes are selected to optimize FFT computations.

    Parameters
    ----------
    image : torch.Tensor or np.ndarray
        The input image to pad. It should be a PyTorch tensor or a NumPy array.
    axes : Iterable[int], optional
        The axes along which to apply padding. Defaults to `(0, 1)`.

    Returns
    -------
    torch.Tensor or np.ndarray
        The padded image with dimensions optimized for FFT performance.

    Raises
    ------
    ValueError
        If no suitable size is found in `_FASTEST_SIZES` for any axis length.
    """

    def _closest(dim: int) -> int:
        # Returns the smallest value from _FASTEST_SIZES larger than dim.
        for size in _FASTEST_SIZES:
            if size >= dim:
                return size
        raise ValueError(
            f"No suitable size found in _FASTEST_SIZES={_FASTEST_SIZES} "
            f"for dimension {dim}."
        )

    # Compute new shape by finding the closest size for specified axes.
    new_shape = list(image.shape)
    for axis in axes:
        new_shape[axis] = _closest(new_shape[axis])

    # Calculate the padding for each axis.
    pad_width = []
    for i, size in enumerate(new_shape):
        increase = size - image.shape[i]
        pad_width.append((0, increase))
    
    if isinstance(image, np.ndarray):
        return np.pad(image, pad_width, mode="constant")
    
    # Flatten pad_width and apply padding using torch.nn.functional.pad
    pad_width_flattened = [item for sublist in reversed(pad_width) for item in sublist]
    padded_image = torch.nn.functional.pad(image, pad_width_flattened, mode="constant", value=0)
    
    return padded_image


class Optics(OriginalOptics):
    """
    A class that represents the optics of a microscope.
    Inherits from the original Optics class in deeptrack.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "Optics"  # Set the name of the optics to "Optics"
        self._description = "Optics of the microscope"  # Set the description of the optics

    def _pupil_tensor(
            self:  'Optics',
            shape: ArrayLike[int],
            NA: float,
            wavelength: float,
            refractive_index_medium: float,
            include_aberration: bool = True,   
            defocus: Union[float, ArrayLike[float]] = 0,
            **kwargs: Dict[str, Any],
        ):
            """Calculates the pupil function at different focal points.

            Parameters
            ----------
            shape: array_like[int, int]
                The shape of the pupil function.
            NA: float
                The NA of the limiting aperture.
            wavelength: float
                The wavelength of the scattered light in meters.
            refractive_index_medium: float
                The refractive index of the medium.
            voxel_size: array_like[float (, float, float)]
                The distance between pixels in the camera. A third value can be
                included to define the resolution in the z-direction.
            include_aberration: bool
                If True, the aberration is included in the pupil function.
            defocus: float or list[float]
                The defocus of the system. If a list is given, the pupil is
                calculated for each focal point. Defocus is given in meters.

            Returns
            -------
            pupil: array_like[complex]
                The pupil function. Shape is (z, y, x).

            Examples
            --------
            Calculating the pupil function:

            >>> import deeptrack as dt

            >>> optics = dt.Optics()
            >>> pupil = optics._pupil(
            ...     shape=(128, 128),
            ...     NA=0.8,
            ...     wavelength=0.55e-6,
            ...     refractive_index_medium=1.33,
            ... )
            >>> print(pupil.shape)
            (1, 128, 128)
            
            """

            # if device is in kwargs set it
            if 'device' in kwargs:
                device = kwargs['device']
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Calculates the pupil at each z-position in defocus.
            voxel_size = get_active_voxel_size()
            shape = torch.tensor(shape)

            # Pupil radius
            R = NA / wavelength * torch.tensor(voxel_size)[:2]

            x_radius = R[0] * shape[0]
            y_radius = R[1] * shape[1]

            x = (torch.linspace(-(shape[0] / 2), shape[0] / 2 - 1, shape[0])) / x_radius + 1e-8
            y = (torch.linspace(-(shape[1] / 2), shape[1] / 2 - 1, shape[1])) / y_radius + 1e-8

            W, H = torch.meshgrid(x, y, indexing='ij')
            W = W.to(device)
            H = H.to(device)
            RHO = (W ** 2 + H ** 2)
            pupil_function = Image((RHO < 1) + 0.0j, copy=False)

            # Defocus
            z_shift = Image(
                2
                * torch.pi
                * refractive_index_medium
                / wavelength
                * voxel_size[2]
                * torch.sqrt(1 - (NA / refractive_index_medium) ** 2 * RHO)+0j,
                copy=False,
            )

            z_shift._value[z_shift._value.imag != 0] = 0
            try:
                z_shift = torch.nan_to_num(z_shift._value, nan=0.0, posinf=None, neginf=None)
            except TypeError:
                torch.nan_to_num(z_shift, z_shift)

            # Ensure defocus is a list of tensors or numbers
            if isinstance(defocus, list):
                # Convert each element to CPU and NumPy if it's a tensor
                defocus = torch.tensor(defocus)
    
            defocus = torch.reshape(defocus, (-1, 1, 1)).to(device)
            z_shift = defocus * z_shift.unsqueeze(0)
            
            if include_aberration:
                pupil = self.pupil

                if isinstance(pupil, Feature):
                    pupil_function = pupil(pupil_function.cpu().numpy())
                    pupil_function = torch.tensor(pupil_function).to(device)

                elif isinstance(pupil, np.ndarray):
                    pupil_function *= torch.tensor(pupil).to(device)

            pupil_functions = pupil_function * torch.exp(1j * z_shift)

            return pupil_functions 
    
    def _pad_volume_tensor(
        self: 'Optics',
        volume: torch.Tensor,
        limits: torch.Tensor = None,
        padding: torch.Tensor = None,
        output_region: torch.Tensor = None,
        **kwargs: Dict[str, Any],
        ) -> tuple:
        """Pads the volume with zeros to avoid edge effects."""
        
        if limits is None:
            limits = torch.zeros((3, 2), dtype=torch.int32, device=volume.device)

        new_limits = limits.clone()

        # Ensure padding is a tensor
        if not isinstance(padding, torch.Tensor):
            padding = torch.tensor(padding, dtype=torch.int32, device=volume.device)

        # Ensure output_region is properly initialized
        if output_region is None:
            output_region = limits.clone()  # Default to current limits
        elif not isinstance(output_region, torch.Tensor):
            output_region = torch.tensor(output_region, dtype=torch.int32, device=volume.device)

        # Handle None values in output_region (replace with current limits)
        for i in range(4):
            if output_region[i] < 0 or output_region[i] is None:
                output_region[i] = limits[i // 2, i % 2]

        # Update new_limits
        for i in range(2):
            new_limits[i, 0] = torch.min(new_limits[i, 0], output_region[i] - padding[i])
            new_limits[i, 1] = torch.max(new_limits[i, 1], output_region[i + 2] + padding[i + 2])

        # Compute new shape
        new_shape = (new_limits[:, 1] - new_limits[:, 0]).int().tolist()
        new_volume = torch.zeros(new_shape, dtype=volume.dtype, device=volume.device)
        
        # Compute old region
        old_region = (limits - new_limits).int()
        limits = limits.int()

        new_volume = new_volume.clone()  # Ensure new tensor (avoiding in-place ops)
        mask = torch.zeros_like(new_volume, dtype=torch.bool)
        mask[
            old_region[0, 0]: old_region[0, 0] + limits[0, 1] - limits[0, 0],
            old_region[1, 0]: old_region[1, 0] + limits[1, 1] - limits[1, 0],
            old_region[2, 0]: old_region[2, 0] + limits[2, 1] - limits[2, 0]
        ] = True
        
        pad_x = new_volume.shape[0] - volume.shape[0]
        pad_y = new_volume.shape[1] - volume.shape[1]
        pad_z = new_volume.shape[2] - volume.shape[2]
        pad_x1, pad_x2 = pad_x // 2, pad_x - (pad_x // 2)
        pad_y1, pad_y2 = pad_y // 2, pad_y - (pad_y // 2)
        pad_z1, pad_z2 = pad_z // 2, pad_z - (pad_z // 2)

        padded_volume = torch.nn.functional.pad(
            volume, (pad_z1, pad_z2, pad_y1, pad_y2, pad_x1, pad_x2), mode='constant', value=0
            )

        #new_volume = new_volume.masked_scatter(mask, volume)
        new_volume = torch.where(mask, padded_volume, new_volume)

        return new_volume, new_limits
    
class Fluorescence(Optics):
    """Optical device for fluorescent imaging.

    The `Fluorescence` class simulates the imaging process in fluorescence
    microscopy by creating a discretized volume where each pixel represents 
    the intensity of light emitted by fluorophores in the sample. It extends 
    the `Optics` class to include fluorescence-specific functionalities.

    Parameters
    ----------
    NA: float
        Numerical aperture of the optical system.
    wavelength: float
        Emission wavelength of the fluorescent light (in meters).
    magnification: float
        Magnification of the optical system.
    resolution: array_like[float (, float, float)]
        Pixel spacing in the camera. Optionally includes the z-direction.
    refractive_index_medium: float
        Refractive index of the imaging medium.
    padding: array_like[int, int, int, int]
        Padding applied to the sample volume to reduce edge effects.
    output_region: array_like[int, int, int, int], optional
        Region of the output image to extract (x, y, width, height). If None, 
        returns the full image.
    pupil: Feature, optional
        A feature set defining the pupil function at focus. The input is 
        the unaberrated pupil.
    illumination: Feature, optional
        A feature set defining the illumination source.
    upscale: int, optional
        Scaling factor for the resolution of the optical system.
    **kwargs: Dict[str, Any]

    Attributes
    ----------
    __gpu_compatible__: bool
        Indicates whether the class supports GPU acceleration.
    NA: float
        Numerical aperture of the optical system.
    wavelength: float
        Emission wavelength of the fluorescent light (in meters).
    magnification: float
        Magnification of the optical system.
    resolution: array_like[float (, float, float)]
        Pixel spacing in the camera. Optionally includes the z-direction.
    refractive_index_medium: float
        Refractive index of the imaging medium.
    padding: array_like[int, int, int, int]
        Padding applied to the sample volume to reduce edge effects.
    output_region: array_like[int, int, int, int]
        Region of the output image to extract (x, y, width, height).
    voxel_size: function
        Function returning the voxel size of the optical system.
    pixel_size: function
        Function returning the pixel size of the optical system.
    upscale: int
        Scaling factor for the resolution of the optical system.
    limits: array_like[int, int]
        Limits of the volume to be imaged.
    fields: list[Feature]
        List of fields to be imaged

    Methods
    -------
    `get(illuminated_volume: array_like[complex], limits: array_like[int, int], **kwargs: Dict[str, Any]) -> Image`
        Simulates the imaging process using a fluorescence microscope.

    Examples
    --------
    Create a `Fluorescence` instance:

    >>> import deeptrack as dt

    >>> optics = dt.Fluorescence(
    ...     NA=1.4, wavelength=0.52e-6, magnification=60,
    ... )
    >>> print(optics.NA())
    1.4

    """

    __gpu_compatible__ = True

    def get(
        self:  'Fluorescence', 
        illuminated_volume: ArrayLike[torch.complex], 
        limits: ArrayLike[int], 
        **kwargs: Dict[str, Any]
    ) -> Image:
        """Simulates the imaging process using a fluorescence microscope.

        This method convolves the 3D illuminated volume with a pupil function 
        to generate a 2D image projection.

        Parameters
        ----------
        illuminated_volume: array_like[complex]
            The illuminated 3D volume to be imaged.
        limits: array_like[int, int]
            Boundaries of the illuminated volume in each dimension.
        **kwargs: Dict[str, Any]
            Additional properties for the imaging process, such as:
            - 'padding': Padding to apply to the sample.
            - 'output_region': Specific region to extract from the image.

        Returns
        -------
        Image: Image
            A 2D image object representing the fluorescence projection.

        Notes
        -----
        - Empty slices in the volume are skipped for performance optimization.
        - The pupil function incorporates defocus effects based on z-slice.

        Examples
        --------
        Simulate imaging a volume:

        >>> import deeptrack as dt
        >>> import numpy as np

        >>> optics = dt.Fluorescence(
        ...     NA=1.4, wavelength=0.52e-6, magnification=60,
        ... )
        >>> volume = dt.Image(np.ones((128, 128, 10), dtype=complex))
        >>> limits = np.array([[0, 128], [0, 128], [0, 10]])
        >>> properties = optics.properties()
        >>> filtered_properties = {
        ...     k: v for k, v in properties.items() 
        ...     if k in {"padding", "output_region", "NA", 
        ...              "wavelength", "refractive_index_medium"}
        ... }
        >>> image = optics.get(volume, limits, **filtered_properties)
        >>> print(image.shape)
        (128, 128, 1)
        
        """

        # Pad volume
        padded_volume, limits = self._pad_volume_tensor(
            illuminated_volume, limits=limits, **kwargs
        )

        pad = kwargs.get("padding", (0, 0, 0, 0))
        output_region = torch.tensor(
            kwargs.get("output_region", (None, None, None, None)), dtype=torch.int32
        )
        
        output_region = output_region.tolist()  # Convert to list for element-wise modification
        output_region[0] = (
            None if output_region[0] is None else int(output_region[0] - limits[0, 0] - pad[0])
        )
        output_region[1] = (
            None if output_region[1] is None else int(output_region[1] - limits[1, 0] - pad[1])
        )
        output_region[2] = (
            None if output_region[2] is None else int(output_region[2] - limits[0, 0] + pad[2])
        )
        output_region[3] = (
            None if output_region[3] is None else int(output_region[3] - limits[1, 0] + pad[3])
        )
        
        padded_volume = padded_volume[
            output_region[0] : output_region[2],
            output_region[1] : output_region[3],
            :,
        ]
        
        z_limits = limits[2, :]

        output_image = Image(
            torch.zeros((*padded_volume.shape[0:2], 1)).to(padded_volume.device),
            )

        index_iterator = range(padded_volume.shape[2])
        z_iterator = torch.linspace(
            z_limits[0],
            z_limits[1],
            padded_volume.shape[2],
            ).to(padded_volume.device)

        zero_plane = torch.all(padded_volume < 1e-8, axis=(0, 1), keepdims=False)
        z_values = torch.masked_select(z_iterator, ~zero_plane)
        
        volume = pad_image_to_fft(padded_volume, axes=(0, 1))

        #voxel_size = get_active_voxel_size()

        #pupils = self._pupil(
        #    volume.shape[:2], defocus=z_values, include_aberration=False, **kwargs
        #    )

        pupils = self._pupil_tensor(
            volume.shape[:2], defocus=z_values, include_aberration=False, device=volume.device, **kwargs
            )
                
        pupils = [torch.tensor(pupil, dtype=torch.complex64).to(volume.device) for pupil in pupils]
        
        z_index = 0

        # Loop through volume and convolve sample with pupil function
        for i in index_iterator:

            if zero_plane[i]:
                continue

            pupil = pupils[z_index]
            z_index += 1

            psf = torch.square(torch.abs(torch.fft.ifft2(torch.fft.fftshift(pupil))))
            optical_transfer_function = torch.fft.fft2(psf)
            fourier_field = torch.fft.fft2(volume[:, :, i])
            convolved_fourier_field = fourier_field * optical_transfer_function
            field = torch.fft.ifft2(convolved_fourier_field)
            # # Discard remaining imaginary part (should be 0 up to rounding error)
            field = torch.real(field)
            output_image._value[:, :, 0] += field[
                : padded_volume.shape[0], : padded_volume.shape[1]
            ]

        output_image = output_image[pad[0]: -pad[2], pad[1]: -pad[3]]

        # Some better way to do this probably...
        illuminated_volume = Image(illuminated_volume)
        pupils = Image(pupils[0])
        
        output_image.properties = illuminated_volume.properties + pupils.properties

        return output_image

class Brightfield(Optics):
    """Simulates imaging of coherently illuminated samples.

    The `Brightfield` class models a brightfield microscopy setup, imaging 
    samples by iteratively propagating light through a discretized volume.
    Each voxel in the volume represents the effective refractive index 
    of the sample at that point. Light is propagated iteratively through 
    Fourier space and corrected in real space.

    Parameters
    ----------
    illumination: Feature, optional
        Feature-set representing the complex field entering the sample. 
        Default is a uniform field with all values set to 1.
    NA: float
        Numerical aperture of the limiting aperture.
    wavelength: float
        Wavelength of the incident light in meters.
    magnification: float
        Magnification of the optical system.
    resolution: array_like[float (, float, float)]
        Pixel spacing in the camera. A third value can define the 
        resolution in the z-direction.
    refractive_index_medium: float
        Refractive index of the medium.
    padding: array_like[int, int, int, int]
        Padding added to the sample volume to minimize edge effects.
    output_region: array_like[int, int, int, int], optional
        Specifies the region of the image to output (x, y, width, height).
        Default is None, which outputs the entire image.
    pupil: Feature, optional
        Feature-set defining the pupil function. The input is the 
        unaberrated pupil.

    Attributes
    ----------
    __gpu_compatible__: bool
        Indicates whether the class supports GPU acceleration.
    __conversion_table__: ConversionTable
        Table used to convert properties of the feature to desired units.
    NA: float
        Numerical aperture of the optical system.
    wavelength: float
        Wavelength of the scattered light in meters.
    magnification: float
        Magnification of the optical system.
    resolution: array_like[float (, float, float)]
        Pixel spacing in the camera. Optionally includes the z-direction.
    refractive_index_medium: float
        Refractive index of the medium.
    padding: array_like[int, int, int, int]
        Padding applied to the sample volume to reduce edge effects.
    output_region: array_like[int, int, int, int]
        Region of the output image to extract (x, y, width, height).
    voxel_size: function
        Function returning the voxel size of the optical system.
    pixel_size: function
        Function returning the pixel size of the optical system.
    upscale: int
        Scaling factor for the resolution of the optical system.
    limits: array_like[int, int]
        Limits of the volume to be imaged.
    fields: list[Feature]
        List of fields to be imaged.

    Methods
    -------
    `get(illuminated_volume: array_like[complex], 
        limits: array_like[int, int], fields: array_like[complex], 
        **kwargs: Dict[str, Any]) -> Image`
        Simulates imaging with brightfield microscopy.


    Examples
    --------
    Create a `Brightfield` instance:

    >>> import deeptrack as dt

    >>> optics = dt.Brightfield(NA=1.4, wavelength=0.52e-6, magnification=60)
    >>> print(optics.NA())
    1.4
    
    """

    __gpu_compatible__ = True

    __conversion_table__ = ConversionTable(
        working_distance=(u.meter, u.meter),
    )

    def get(
        self:  'Brightfield',
        illuminated_volume: ArrayLike[torch.complex],
        limits: ArrayLike[int],
        fields: ArrayLike[torch.complex],
        **kwargs: Dict[str, Any],
    ) -> Image:
        """Simulates imaging with brightfield microscopy.

        This method propagates light through the given volume, applying 
        pupil functions at various defocus levels and incorporating 
        refraction corrections in real space to produce the final 
        brightfield image.

        Parameters
        ----------
        illuminated_volume: array_like[complex]
            Discretized volume representing the sample to be imaged.
        limits: array_like[int, int]
            Boundaries of the sample volume in each dimension.
        fields: array_like[complex]
            Input fields to be used in the imaging process.
        **kwargs: Dict[str, Any]
            Additional parameters for the imaging process, including:
            - 'padding': Padding to apply to the sample volume.
            - 'output_region': Specific region to extract from the image.
            - 'wavelength': Wavelength of the light.
            - 'refractive_index_medium': Refractive index of the medium.

        Returns
        -------
        Image: Image
            Processed image after simulating the brightfield imaging process.

        Examples
        --------
        Simulate imaging a volume:

        >>> import deeptrack as dt
        >>> import numpy as np

        >>> optics = dt.Brightfield(
        ...     NA=1.4, 
        ...     wavelength=0.52e-6, 
        ...     magnification=60,
        ... )
        >>> volume = dt.Image(np.ones((128, 128, 10), dtype=complex))
        >>> limits = np.array([[0, 128], [0, 128], [0, 10]])
        >>> fields = np.array([np.ones((162, 162), dtype=complex)])
        >>> properties = optics.properties()
        >>> filtered_properties = {
        ...     k: v for k, v in properties.items()
        ...     if k in {'padding', 'output_region', 'NA', 
        ...              'wavelength', 'refractive_index_medium'}
        ... }
        >>> image = optics.get(volume, limits, fields, **filtered_properties)
        >>> print(image.shape)
        (128, 128, 1)
        
        """

        # Pad volume
        padded_volume, limits = self._pad_volume_tensor(
            illuminated_volume, limits=limits, **kwargs
        )

        pad = kwargs.get("padding", (0, 0, 0, 0))
        output_region = torch.tensor(
            kwargs.get("output_region", (None, None, None, None)), dtype=torch.int32
        )
        
        output_region = output_region.tolist()  # Convert to list for element-wise modification
        output_region[0] = (
            None if output_region[0] is None else int(output_region[0] - limits[0, 0] - pad[0])
        )
        output_region[1] = (
            None if output_region[1] is None else int(output_region[1] - limits[1, 0] - pad[1])
        )
        output_region[2] = (
            None if output_region[2] is None else int(output_region[2] - limits[0, 0] + pad[2])
        )
        output_region[3] = (
            None if output_region[3] is None else int(output_region[3] - limits[1, 0] + pad[3])
        )
        
        padded_volume = padded_volume[
            output_region[0] : output_region[2],
            output_region[1] : output_region[3],
            :,
        ]
        
        z_limits = limits[2, :]

        #output_image = Image(
        #    torch.zeros((*padded_volume.shape[0:2], 1))
        #    )

        index_iterator = range(padded_volume.shape[2])
        #z_iterator = torch.linspace(
        #    z_limits[0],
        #    z_limits[1],
        #    padded_volume.shape[2],
        #    ).to(padded_volume.device)

        zero_plane = torch.all(padded_volume == 0, axis=(0, 1), keepdims=False)
        # z_values = z_iterator[~zero_plane]

        volume = pad_image_to_fft(padded_volume, axes=(0, 1))
        
        voxel_size = get_active_voxel_size()

        pupils = [
            self._pupil_tensor(
                volume.shape[:2], defocus=[1], include_aberration=False, device=volume.device, **kwargs
            )[0],
            self._pupil_tensor(
                volume.shape[:2],
                defocus=[-z_limits[1]],
                include_aberration=True,
                device=volume.device,
                **kwargs
            )[0],
            self._pupil_tensor(
                volume.shape[:2],
                defocus=[0],
                include_aberration=True,
                device=volume.device,
                **kwargs
            )[0]
        ]

        # Transform the pupils to torch tensors.
        pupils = [torch.tensor(pupil, dtype=torch.complex64).to(volume.device) for pupil in pupils]

        pupil_step = torch.fft.fftshift(pupils[0])

        light_in = torch.ones((volume.shape[:2]), dtype=torch.complex64).to(volume.device)
        light_in = self.illumination.resolve(light_in)
        light_in = torch.fft.fft2(light_in)

        K = 2 * torch.pi / kwargs["wavelength"]*kwargs["refractive_index_medium"]

        z = z_limits[1]
        for i in index_iterator:
            #light_in = light_in * pupil_step
            light_in = torch.where(zero_plane[i], light_in, light_in * pupil_step)
            #if zero_plane[i]:
            #    continue
            
            ri_slice = volume[:, :, i]
            light = torch.fft.ifft2(light_in)
            light_out = light * torch.exp(1j * ri_slice * voxel_size[-1] * K)
            light_in = torch.fft.fft2(light_out)
  
        shifted_pupil = torch.fft.fftshift(pupils[1])
        light_in_focus = light_in * shifted_pupil

        if len(fields) > 0:
            field = torch.sum(fields, axis=0)
            light_in_focus += field[..., 0]
        shifted_pupil = torch.fft.fftshift(pupils[-1])
        light_in_focus = light_in_focus * shifted_pupil
        # Mask to remove light outside the pupil.
        mask = torch.abs(shifted_pupil) > 0
        light_in_focus = light_in_focus * mask

        output_image = torch.fft.ifft2(light_in_focus)[
            : padded_volume.shape[0], : padded_volume.shape[1]
        ]
        output_image = torch.unsqueeze(output_image, axis=-1)

        # Intensity image if not returning field
        if not kwargs.get("return_field", False):
            output_image = torch.square(torch.abs(output_image))

        output_image = Image(output_image[pad[0] : -pad[2], pad[1] : -pad[3]])

        illuminated_volume = Image(illuminated_volume)
        output_image.properties = illuminated_volume.properties

        return output_image


class ISCAT(Brightfield):
    """Images coherently illuminated samples using Interferometric Scattering 
    (ISCAT) microscopy.

    This class models ISCAT by creating a discretized volume where each pixel
    represents the effective refractive index of the sample. Light is 
    propagated through the sample iteratively, first in the Fourier space 
    and then corrected in the real space for refractive index.

    Parameters
    ----------
    illumination: Feature
        Feature-set defining the complex field entering the sample. Default 
        is a field with all values set to 1.
    NA: float
        Numerical aperture (NA) of the limiting aperture.
    wavelength: float
        Wavelength of the scattered light, in meters.
    magnification: float
        Magnification factor of the optical system.
    resolution: array_like of float
        Pixel spacing in the camera. Optionally includes a third value for 
        z-direction resolution.
    refractive_index_medium: float
        Refractive index of the medium surrounding the sample.
    padding: array_like of int
        Padding for the sample volume to minimize edge effects. Format: 
        (left, right, top, bottom).
    output_region: array_like of int
        Region of the image to output as (x, y, width, height). If None 
        (default), the entire image is returned.
    pupil: Feature
        Feature-set defining the pupil function at focus. The feature-set 
        takes an unaberrated pupil as input.
    illumination_angle: float, optional
        Angle of illumination relative to the optical axis, in radians. 
        Default is π radians.
    amp_factor: float, optional
        Amplitude factor of the illuminating field relative to the reference 
        field. Default is 1.

    Attributes
    ----------
    illumination_angle: float
        The angle of illumination, stored for reference.
    amp_factor: float
        Amplitude factor of the illuminating field.

    Examples
    --------
    Creating an ISCAT instance:
    
    >>> import deeptrack as dt

    >>> iscat = dt.ISCAT(NA=1.4, wavelength=0.532e-6, magnification=60)
    >>> print(iscat.illumination_angle())
    3.141592653589793
    
    """

    def __init__(
        self:  'ISCAT',
        illumination_angle: float = np.pi,
        amp_factor: float = 1, 
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initializes the ISCAT class.

        Parameters
        ----------
        illumination_angle: float
            The angle of illumination, in radians.
        amp_factor: float
            Amplitude factor of the illuminating field relative to the reference 
            field.
        **kwargs: Dict[str, Any]
            Additional parameters for the Brightfield class.

        """

        super().__init__(
            illumination_angle=illumination_angle,
            amp_factor=amp_factor,
            input_polarization="circular",
            output_polarization="circular",
            phase_shift_correction=True,
            **kwargs
            )
        
class Darkfield(Brightfield):
    """Images coherently illuminated samples using Darkfield microscopy.

    This class models Darkfield microscopy by creating a discretized volume 
    where each pixel represents the effective refractive index of the sample. 
    Light is propagated through the sample iteratively, first in the Fourier 
    space and then corrected in the real space for refractive index.

    Parameters
    ----------
    illumination: Feature
        Feature-set defining the complex field entering the sample. Default 
        is a field with all values set to 1.
    NA: float
        Numerical aperture (NA) of the limiting aperture.
    wavelength: float
        Wavelength of the scattered light, in meters.
    magnification: float
        Magnification factor of the optical system.
    resolution: array_like of float
        Pixel spacing in the camera. Optionally includes a third value for 
        z-direction resolution.
    refractive_index_medium: float
        Refractive index of the medium surrounding the sample.
    padding: array_like of int
        Padding for the sample volume to minimize edge effects. Format: 
        (left, right, top, bottom).
    output_region: array_like of int
        Region of the image to output as (x, y, width, height). If None 
        (default), the entire image is returned.
    pupil: Feature
        Feature-set defining the pupil function at focus. The feature-set 
        takes an unaberrated pupil as input.
    illumination_angle: float, optional
        Angle of illumination relative to the optical axis, in radians. 
        Default is π/2 radians.

    Attributes
    ----------
    illumination_angle: float
        The angle of illumination, stored for reference.

    Methods
    -------
    get(illuminated_volume, limits, fields, **kwargs)
        Retrieves the darkfield image of the illuminated volume.

    Examples
    --------
    Creating a Darkfield instance:

    >>> import deeptrack as dt

    >>> darkfield = dt.Darkfield(NA=0.9, wavelength=0.532e-6)
    >>> print(darkfield.illumination_angle())
    1.5707963267948966

    """

    def __init__(
        self: 'Darkfield', 
        illumination_angle: float = np.pi/2, 
        **kwargs: Dict[str, Any]
    ) -> None:
        """Initializes the Darkfield class.

        Parameters
        ----------
        illumination_angle: float
            The angle of illumination, in radians.
        **kwargs: Dict[str, Any]
            Additional parameters for the Brightfield class.

        """

        super().__init__(
            illumination_angle=illumination_angle,
            **kwargs)

    #Retrieve get as super
    def get(
        self: 'Darkfield',
        illuminated_volume: ArrayLike[complex],
        limits: ArrayLike[int],
        fields: ArrayLike[complex],
        **kwargs: Dict[str, Any],
    ) -> Image:
        """Retrieve the darkfield image of the illuminated volume.

        Parameters
        ----------
        illuminated_volume: array_like
            The volume of the sample being illuminated.
        limits: array_like
            The spatial limits of the volume.
        fields: array_like
            The fields interacting with the sample.
        **kwargs: Dict[str, Any]
            Additional parameters passed to the super class's get method.

        Returns
        -------
        numpy.ndarray
            The darkfield image obtained by calculating the squared absolute
            difference from 1.dee
        
        """

        field = super().get(illuminated_volume, limits, fields, return_field=True, **kwargs)
        field._value = torch.square(torch.abs(field._value-torch.mean(field._value)))
        return field
