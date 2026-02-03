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


import functools
import hashlib

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "Optics"
        self._description = "Optics of the microscope"
        # --- caches ---
        self._pupil_cache = {}     # key -> complex tensor (Z,H,W) or (H,W)
        self._fftshift_cache = {}  # key -> fftshifted pupil
        self._otf_cache = {}       # key -> OTF tensor (complex64)

    @staticmethod
    def _tensor_key(x: torch.Tensor, max_len: int = 256):
        """
        Create a small hashable key for a tensor contents (used for z-values).
        We only use CPU float32 and round to reduce key explosion.
        """
        if x is None:
            return None
        x = x.detach().float().cpu()
        if x.numel() > max_len:
            # downsample key if huge (rare)
            idx = torch.linspace(0, x.numel() - 1, max_len).long()
            x = x.flatten()[idx]
        x = torch.round(x * 1e6) / 1e6  # quantize a bit
        return tuple(x.flatten().tolist())

    def _safe_get_pupil(self):
        """Safely retrieve pupil without triggering DeepTrack Feature __getattr__ crash."""
        try:
            return getattr(self, "pupil")
        except Exception:
            # If DeepTrack intercepts attribute access and fails, treat as no pupil/aberration
            return None

    def _cached_pupil_tensor(
        self,
        shape_hw,
        NA,
        wavelength,
        refractive_index_medium,
        include_aberration: bool,
        defocus,
        device,
        **kwargs
    ) -> torch.Tensor:
        """
        Cached wrapper around _pupil_tensor.
        Returns: tensor of shape (Z,H,W) complex64
        """
        H, W = int(shape_hw[0]), int(shape_hw[1])

        # normalize defocus into a tensor on CPU for key
        if isinstance(defocus, (list, tuple)):
            defocus_t = torch.tensor(defocus, dtype=torch.float32)
        elif isinstance(defocus, torch.Tensor):
            defocus_t = defocus.detach().float().cpu()
        else:
            defocus_t = torch.tensor([float(defocus)], dtype=torch.float32)

        try:
            pupil_obj = getattr(self, "pupil")
        except Exception:
            pupil_obj = None

        pupil_id = "none" if pupil_obj is None else f"{type(pupil_obj).__name__}:{id(pupil_obj)}"

        key = (
            "pupil",
            H, W,
            float(NA), float(wavelength), float(refractive_index_medium),
            bool(include_aberration),
            self._tensor_key(defocus_t),
            # include pupil identity if Feature is used
            pupil_id,
        )

        # prevent accidental duplicates if caller passes NA etc inside kwargs
        kwargs = dict(kwargs)
        for k in ["NA", "wavelength", "refractive_index_medium", "defocus", "include_aberration", "device"]:
            kwargs.pop(k, None)

        out = self._pupil_cache.get(key, None)
        if out is None or out.device != device:
            out = self._pupil_tensor(
                shape=(H, W),
                NA=NA,
                wavelength=wavelength,
                refractive_index_medium=refractive_index_medium,
                include_aberration=include_aberration,
                defocus=defocus_t.to(device),
                device=device,
                **kwargs
            ).to(torch.complex64)
            self._pupil_cache[key] = out
        return out
    
    def _pupil_tensor(
        self: "Optics",
        shape,
        NA: float,
        wavelength: float,
        refractive_index_medium: float,
        include_aberration: bool = True,
        defocus=0,
        **kwargs,
    ):
        """
        Torch implementation of pupil function with safe aberration handling.

        - Works even if `self.pupil` is missing (DeepTrack Feature __getattr__ trap).
        - Supports `self.pupil` as deeptrack Feature, numpy array, torch tensor, or None.
        - Returns tensor of shape (Z, H, W) complex64.
        """

        # Device
        device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Active voxel size from deeptrack context
        voxel_size = get_active_voxel_size()  # (dx, dy, dz) in meters

        # Normalize shape to (H, W)
        if isinstance(shape, torch.Tensor):
            H, W = int(shape[0].item()), int(shape[1].item())
        else:
            H, W = int(shape[0]), int(shape[1])

        # Radius scaling
        # R = NA / wavelength * voxel_size[:2]
        # x_radius = R[0] * H, y_radius = R[1] * W
        # Keep consistent with your original code:
        R = (NA / wavelength) * torch.tensor(voxel_size, device=device)[:2]
        x_radius = R[0] * H
        y_radius = R[1] * W

        # Coordinates in pupil plane
        x = (torch.linspace(-(H / 2), H / 2 - 1, H, device=device) / x_radius) + 1e-8
        y = (torch.linspace(-(W / 2), W / 2 - 1, W, device=device) / y_radius) + 1e-8
        Wg, Hg = torch.meshgrid(x, y, indexing="ij")
        RHO = (Wg**2 + Hg**2)

        # Base pupil (unit disk)
        pupil_function = (RHO < 1).to(torch.complex64)

        # z phase shift kernel (complex), shape (H,W)
        z_shift = (
            2
            * torch.pi
            * refractive_index_medium
            / wavelength
            * voxel_size[2]
            * torch.sqrt(1 - (NA / refractive_index_medium) ** 2 * RHO)
        ).to(torch.complex64)

        # Clean invalid values
        # sqrt can yield nan for RHO outside range; also handle imaginary
        z_shift = torch.nan_to_num(z_shift.real, nan=0.0).to(torch.complex64)

        # Normalize defocus to tensor shape (Z,1,1)
        if isinstance(defocus, (list, tuple)):
            defocus_t = torch.tensor(defocus, dtype=torch.float32, device=device)
        elif isinstance(defocus, torch.Tensor):
            defocus_t = defocus.to(device=device, dtype=torch.float32)
        else:
            defocus_t = torch.tensor([float(defocus)], dtype=torch.float32, device=device)

        defocus_t = defocus_t.reshape(-1, 1, 1)  # (Z,1,1)

        # --- Safe aberration handling (NO direct self.pupil access) ---
        if include_aberration:
            try:
                pupil_obj = getattr(self, "pupil")
            except Exception:
                pupil_obj = None

            if pupil_obj is not None:
                # Feature pupil (runs in numpy)
                if isinstance(pupil_obj, Feature):
                    # pupil_function is complex; Feature likely expects numpy array
                    pf_np = pupil_function.detach().cpu().numpy()
                    pupil_np = pupil_obj(pf_np)
                    pupil_t = torch.as_tensor(pupil_np, dtype=torch.complex64, device=device)
                    pupil_function = pupil_function * pupil_t

                # numpy array pupil
                elif isinstance(pupil_obj, np.ndarray):
                    pupil_t = torch.as_tensor(pupil_obj, dtype=torch.complex64, device=device)
                    pupil_function = pupil_function * pupil_t

                # torch tensor pupil
                elif isinstance(pupil_obj, torch.Tensor):
                    pupil_function = pupil_function * pupil_obj.to(device=device, dtype=torch.complex64)

                # unsupported type -> ignore
                else:
                    pass

        # Broadcast: pupil_function (H,W) with defocus phase (Z,H,W)
        # exp(i * defocus * z_shift)
        phase = torch.exp(1j * defocus_t.to(torch.complex64) * z_shift.unsqueeze(0))
        pupil_functions = pupil_function.unsqueeze(0) * phase  # (Z,H,W)

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
            device=padded_volume.device,
        )

        # Planes that are effectively empty (fast skip)
        zero_plane = torch.all(padded_volume < 1e-10, dim=(0, 1))
        z_values = z_iterator[~zero_plane]

        # If everything is empty, return zeros immediately
        if z_values.numel() == 0:
            output_image = Image(torch.zeros((*padded_volume.shape[0:2], 1), device=padded_volume.device))
            output_image = output_image[pad[0]: -pad[2], pad[1]: -pad[3]]
            output_image.properties = Image(illuminated_volume).properties
            return output_image

        # Pad to FFT-friendly size (Hfft, Wfft, Z)
        volume = pad_image_to_fft(padded_volume, axes=(0, 1))
        Hfft, Wfft = int(volume.shape[0]), int(volume.shape[1])

        # Compute pupils for the *active* planes only (Z_active, Hfft, Wfft)
        # IMPORTANT: _pupil_tensor already returns torch tensors -> don't re-wrap with torch.tensor(...)
        pupils = self._pupil_tensor(
            volume.shape[:2],
            defocus=z_values,
            include_aberration=False,
            device=volume.device,
            **kwargs,
        ).to(torch.complex64)

        # Make sure caches exist
        if not hasattr(self, "_otf_cache"):
            self._otf_cache = {}

        # Cache key base includes optics settings and FFT shape
        NA = float(kwargs["NA"])
        wavelength = float(kwargs["wavelength"])
        n_medium = float(kwargs["refractive_index_medium"])

        z_index = 0

        # Loop through volume and convolve sample with OTF
        for i in index_iterator:

            if zero_plane[i]:
                continue

            pupil = pupils[z_index]
            zv = float(z_values[z_index].detach().cpu())
            z_index += 1

            # --- Cached OTF for this defocus plane ---
            otf_key = ("fluor_otf", Hfft, Wfft, NA, wavelength, n_medium, zv)

            otf = self._otf_cache.get(otf_key, None)
            if otf is None or otf.device != volume.device:
                psf = torch.abs(torch.fft.ifft2(torch.fft.fftshift(pupil))) ** 2
                otf = torch.fft.fft2(psf).to(torch.complex64)
                self._otf_cache[otf_key] = otf

            # Convolution in Fourier domain
            fourier_field = torch.fft.fft2(volume[:, :, i])
            field = torch.fft.ifft2(fourier_field * otf).real

            output_image._value[:, :, 0] += field[: padded_volume.shape[0], : padded_volume.shape[1]]

        # Crop final output
        output_image = output_image[pad[0]: -pad[2], pad[1]: -pad[3]]

        illuminated_volume = Image(illuminated_volume)

        # If no active planes -> pupils can be empty, so don’t index pupils[0]
        if isinstance(pupils, torch.Tensor) and pupils.numel() > 0 and pupils.shape[0] > 0:
            pupils_img = Image(pupils[0])
            output_image.properties = illuminated_volume.properties + pupils_img.properties
        else:
            # Just keep the illuminated volume properties
            output_image.properties = illuminated_volume.properties

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

        # --- Cached pupils (robust to changing optical parameters) ---
        Hfft, Wfft = int(volume.shape[0]), int(volume.shape[1])

        # Pull required optical params explicitly
        NA = float(kwargs["NA"])
        wavelength = float(kwargs["wavelength"])
        n_medium = float(kwargs["refractive_index_medium"])

        # Any extra kwargs that _pupil_tensor may accept (avoid double-passing)
        pupil_kwargs = dict(kwargs)
        for k in [
            "NA", "wavelength", "refractive_index_medium",
            "defocus", "include_aberration", "device",
            # not part of pupil; safe to remove
            "padding", "output_region", "return_field"
        ]:
            pupil_kwargs.pop(k, None)

        # Defocus values used by your model
        defocus_step = 1.0
        defocus_focus = float(-z_limits[1])
        defocus_final = 0.0

        # Get pupils (cached) -> each returns (Z,H,W); index [0] for single defocus
        p0 = self._cached_pupil_tensor(
            (Hfft, Wfft),
            NA=NA,
            wavelength=wavelength,
            refractive_index_medium=n_medium,
            include_aberration=False,
            defocus=[defocus_step],
            device=volume.device,
            **pupil_kwargs
        )[0]

        p1 = self._cached_pupil_tensor(
            (Hfft, Wfft),
            NA=NA,
            wavelength=wavelength,
            refractive_index_medium=n_medium,
            include_aberration=True,
            defocus=[defocus_focus],
            device=volume.device,
            **pupil_kwargs
        )[0]

        p2 = self._cached_pupil_tensor(
            (Hfft, Wfft),
            NA=NA,
            wavelength=wavelength,
            refractive_index_medium=n_medium,
            include_aberration=True,
            defocus=[defocus_final],
            device=volume.device,
            **pupil_kwargs
        )[0]

        # Cache fftshifted pupils too (keys include all important optics settings)
        key0 = ("fftshift", Hfft, Wfft, NA, wavelength, n_medium, False, defocus_step)
        key1 = ("fftshift", Hfft, Wfft, NA, wavelength, n_medium, True,  defocus_focus)
        key2 = ("fftshift", Hfft, Wfft, NA, wavelength, n_medium, True,  defocus_final)

        cached = self._fftshift_cache.get(key0, None)
        if cached is None or cached.device != volume.device:
            self._fftshift_cache[key0] = torch.fft.fftshift(p0)
        cached = self._fftshift_cache.get(key1, None)
        if cached is None or cached.device != volume.device:
            self._fftshift_cache[key1] = torch.fft.fftshift(p1)
        cached = self._fftshift_cache.get(key2, None)
        if cached is None or cached.device != volume.device:
            self._fftshift_cache[key2] = torch.fft.fftshift(p2)

        pupil_step = self._fftshift_cache[key0]
        shifted_pupil_focus = self._fftshift_cache[key1]
        shifted_pupil_final = self._fftshift_cache[key2]

        # Initial light field
        light_in = torch.ones(volume.shape[:2], dtype=torch.complex64, device=volume.device)
        light_in = self.illumination.resolve(light_in)
        light_in = torch.fft.fft2(light_in)

        K = 2 * torch.pi / kwargs["wavelength"] * kwargs["refractive_index_medium"]

        for i in index_iterator:
            light_in = light_in * pupil_step

            ri_slice = volume[:, :, i]
            light = torch.fft.ifft2(light_in)
            light_out = light * torch.exp(1j * ri_slice * voxel_size[-1] * K)
            light_in = torch.fft.fft2(light_out)

        # pupil at focus (already fftshifted)
        light_in_focus = light_in * shifted_pupil_focus

        if len(fields) > 0:
            field = torch.sum(fields, axis=0)
            light_in_focus = light_in_focus + field[..., 0]

        # final pupil (already fftshifted)
        light_in_focus = light_in_focus * shifted_pupil_final

        mask = torch.abs(shifted_pupil_final) > 0
        light_in_focus = light_in_focus * mask

        output_image = torch.fft.ifft2(light_in_focus)[:padded_volume.shape[0], :padded_volume.shape[1]]
        output_image = torch.unsqueeze(output_image, dim=-1)

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
