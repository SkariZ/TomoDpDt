import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

try:
    import tomodpdt.image_modalities_dt as dt
except Exception:
    import image_modalities_dt as dt

import deeptrack
from deeptrack.backend.units import create_context
from deeptrack.aberrations import SphericalAberration


def setup_optics(
    shape=None,
    nsize=None,
    padding_xy=64,
    microscopy_regime="brightfield",
    NA=0.7,
    wavelength=532e-9,
    resolution=100e-9,
    magnification=1,
    return_field=True,
    **optics_kwargs,
):
    """
    Build a DeepTrack optics object + metadata.

    Parameters
    ----------
    shape : tuple[int,int,int] or None
        (nz, ny, nx) OR (nx, ny, nz)?  -> we standardize internally to (nx, ny, nz) metadata.
        In this file we keep the *metadata* as (nx, ny, nz) for consistency with DeepTrack limits.
    nsize : int or None
        Cubic volume size if shape is None.
    padding_xy : int
        Pad x/y before forward model evaluation.
    microscopy_regime : str
        "brightfield", "fluorescence", "darkfield", "iscat"
    NA, wavelength, resolution, magnification, return_field : floats/bool
        Standard optics parameters.
    optics_kwargs : dict
        Extra keyword args passed into the dt.<Modality>(...) constructor, e.g.
        refractive_index_medium=..., pupil=..., illumination=..., etc.

    Returns
    -------
    dict with keys: microscopy_regime, optics, limits, fields, filtered_properties, padding_xy, resolution, shape
    """
    microscopy_regime = microscopy_regime.lower()

    # Determine shape (metadata as nx, ny, nz)
    if shape is not None:
        if len(shape) != 3:
            raise ValueError("`shape` must be length-3.")

        # Your volumes are usually (nz, ny, nx) tensors.
        # Here we accept either convention and try to infer it.
        # Heuristic: if you pass a torch tensor shape from obj (nz,ny,nx), treat it as such.
        # If you pass a tuple intended as (nx,ny,nz), you can override by passing shape_order="nxnyz".
        shape_order = optics_kwargs.pop("shape_order", "nznyx")  # default to your internal tensor layout
        if shape_order == "nznyx":
            nz, ny, nx = map(int, shape)
        elif shape_order == "nxnyz":
            nx, ny, nz = map(int, shape)
        else:
            raise ValueError("shape_order must be 'nznyx' or 'nxnyz'.")
    elif nsize is not None:
        nx = ny = nz = int(nsize)
    else:
        raise ValueError("Provide either `shape` or `nsize`.")

    # Padded sizes (pad x/y only)
    padded_nx = nx + 2 * int(padding_xy)
    padded_ny = ny + 2 * int(padding_xy)

    # Default pupil for fluorescence if none provided
    if microscopy_regime == "fluorescence":
        optics_kwargs.setdefault("pupil", SphericalAberration())
        return_field = False  # fluorescence returns intensity-like output in your dt wrapper

    # Build optics (pass through additional kwargs)
    if microscopy_regime == "brightfield":
        optics = dt.Brightfield(
            NA=NA,
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, padded_nx, padded_ny),
            return_field=return_field,
            **optics_kwargs,
        )
    elif microscopy_regime == "fluorescence":
        optics = dt.Fluorescence(
            NA=NA,
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, padded_nx, padded_ny),
            **optics_kwargs,
        )
    elif microscopy_regime == "darkfield":
        optics = dt.Darkfield(
            NA=NA,
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, padded_nx, padded_ny),
            **optics_kwargs,
        )
        return_field = False
    elif microscopy_regime == "iscat":
        optics = dt.ISCAT(
            NA=NA,
            wavelength=wavelength,
            resolution=resolution,
            magnification=magnification,
            output_region=(0, 0, padded_nx, padded_ny),
            return_field=return_field,
            **optics_kwargs,
        )
    else:
        raise ValueError(f"Unknown microscopy_regime: {microscopy_regime}")

    # Limits in (x, y, z) order expected by your dt optics
    limits = torch.tensor(
        [[0, padded_nx], [0, padded_ny], [-nz / 2, nz / 2]],
        dtype=torch.float32,
    )

    # Precompute fields (for coherent modalities)
    padded_xy_for_fft = 2 * ((max(padded_nx, padded_ny) + 31) // 32) * 32
    fields = torch.ones((padded_xy_for_fft, padded_xy_for_fft), dtype=torch.complex64)

    # Filtered properties passed into optics.get(...)
    # Keep it permissive: only include keys that actually exist.
    properties = optics.properties()
    allow = {"padding", "output_region", "NA", "wavelength", "refractive_index_medium", "return_field"}
    filtered_properties = {k: v for k, v in properties.items() if k in allow}

    # Ensure return_field is aligned with what we requested (some optics may expose it)
    filtered_properties["return_field"] = return_field if "return_field" in allow else return_field

    return {
        "microscopy_regime": microscopy_regime,
        "optics": optics,
        "limits": limits,
        "fields": fields,
        "filtered_properties": filtered_properties,
        "padding_xy": int(padding_xy),
        "resolution": float(resolution),
        # store metadata as (nx, ny, nz)
        "shape": (nx, ny, nz),
    }


class imaging_model(nn.Module):
    """
    Torch wrapper around DeepTrack optics.

    Input volumes are expected as torch tensors in (nz, ny, nx) layout (your internal convention).
    """

    def __init__(
        self,
        optics_setup: dict,
        device: torch.device | None = None,
        forward_case: str | None = None,
        padding_value: float = 0.0,
        fluorescence_eps: float = 1e-12,
        lazy_background: bool = True,
    ):
        super().__init__()

        self.microscopy_regime = optics_setup["microscopy_regime"].lower()
        self.optics = optics_setup["optics"]
        self.limits = optics_setup["limits"]
        self.fields = optics_setup["fields"]
        self.filtered_properties = optics_setup["filtered_properties"]
        self.padding_xy = int(optics_setup["padding_xy"])
        self.resolution = float(optics_setup["resolution"])
        self.nx, self.ny, self.nz = optics_setup["shape"]  # metadata (nx,ny,nz)

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.padding_value = float(padding_value)

        # Forward strategy
        if forward_case is None:
            # fluorescence: loop is most reliable (and you saw it faster)
            self.forward_case = "loop" if self.microscopy_regime == "fluorescence" else "vmap"
        else:
            self.forward_case = forward_case

        # Brightfield background correction computed lazily
        self._lazy_background = bool(lazy_background)
        self.V0 = None
        self.V0_phase = None

        # Fluorescence baseline subtraction to avoid "all-zero" pathology
        self._fluor_eps = float(fluorescence_eps)
        self._fluor_baseline = None

    def _ensure_device_buffers(self, device):
        self.limits = self.limits.to(device)
        self.fields = self.fields.to(device)

    def _compute_brightfield_background(self, device):
        # Compute V0 and phase correction once (no grad)
        with torch.no_grad():
            empty = torch.zeros((self.nz, self.ny, self.nx), dtype=torch.float32, device=device)
            img = self._imaging_step_core(empty)  # already cropped, raw field
            self.V0 = img
            self.V0_phase = torch.median(torch.angle(self.V0))
            self.V0 = self.V0 * torch.exp(-1j * self.V0_phase)

    def forward(self, obj, vmap: bool = True):
        self._ensure_device_buffers(obj.device)

        # single volume
        if obj.dim() == 3:
            return self.imaging_step(obj)

        # batch dim 1
        if obj.dim() == 4 and obj.size(0) == 1:
            return self.imaging_step(obj.squeeze(0)).unsqueeze(0)

        # batch
        if obj.dim() == 4:
            if self.microscopy_regime == "fluorescence":
                # Explicit loop is fastest/most reliable (your benchmark confirmed)
                return torch.stack([self.imaging_step(obj[i]) for i in range(obj.shape[0])], dim=0)

            if self.forward_case == "vmap" and vmap:
                imaging_vmap = torch.vmap(self.imaging_step, in_dims=0)
                return imaging_vmap(obj)

            return torch.stack([self.imaging_step(sample) for sample in obj], dim=0)

        raise ValueError(f"Expected obj dim 3 or 4, got {obj.dim()}")

    def imaging_step(self, obj: torch.Tensor):
        """
        Full imaging step including background/baseline corrections.
        Input: (nz, ny, nx)
        Output: typically (ny, nx, 1) or complex field thereof
        """
        obj = obj.to(device=self.device)

        # lazily compute brightfield background
        if self.microscopy_regime == "brightfield" and self._lazy_background and self.V0 is None:
            self._compute_brightfield_background(device=obj.device)

        out = self._imaging_step_core(obj)

        # brightfield background correction
        if self.microscopy_regime == "brightfield" and self.V0 is not None:
            out = out * torch.exp(-1j * self.V0_phase)
            out = out - self.V0 + 1

        # fluorescence baseline subtraction (avoids all-zero hack)
        if self.microscopy_regime == "fluorescence":
            if self._fluor_baseline is None or self._fluor_baseline.device != out.device:
                with torch.no_grad():
                    empty = torch.zeros_like(obj)
                    base = self._imaging_step_core(empty + self._fluor_eps)
                    self._fluor_baseline = base.detach()
            out = out - self._fluor_baseline

        return out

    def _imaging_step_core(self, obj: torch.Tensor):
        """
        Core DeepTrack call + padding/cropping.
        No background subtraction here; returns raw optics output as torch tensor.
        """
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
            # pad x/y if needed (obj is nz,ny,nx; convert to nx,ny,nz for padding)
            if self.padding_xy > 0:
                obj = F.pad(
                    obj.permute(2, 1, 0),  # (nx, ny, nz)
                    (0, 0, self.padding_xy, self.padding_xy, self.padding_xy, self.padding_xy),
                    mode="constant",
                    value=self.padding_value,
                ).permute(2, 1, 0)  # back to (nz, ny, nx)

            # optics call
            if self.microscopy_regime in {"brightfield", "darkfield", "iscat"}:
                image = self.optics.get(obj, self.limits, self.fields, **self.filtered_properties)
            elif self.microscopy_regime == "fluorescence":
                # add tiny epsilon everywhere, then subtract baseline in imaging_step()
                image = self.optics.get(obj + self._fluor_eps, self.limits, **self.filtered_properties)
            else:
                raise ValueError(f"Unknown microscopy regime: {self.microscopy_regime}")

        # crop out padding in xy
        if self.padding_xy > 0:
            image = image[self.padding_xy : -self.padding_xy, self.padding_xy : -self.padding_xy]

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


class LearnableProjection(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.microscopy_regime = 'learnable_projection'
        self.conv = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

    def forward(self, x):
        x = self.conv(x)
        return x.sum(dim=-1, keepdim=True)


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

def make_batch(vol_a: torch.Tensor, vol_b: torch.Tensor | None = None, n_a=8, n_b=8):
    vols = [vol_a for _ in range(n_a)]
    if vol_b is not None:
        vols += [vol_b for _ in range(n_b)]
    batch = torch.stack(vols, dim=0)  # (B, Z, Y, X)
    return batch

@torch.no_grad()
def time_loop(model, batch):
    # loop over batch with detach (forward only)
    start = time.time()
    for v in batch:
        _ = model(v)
    return time.time() - start

def test_model(microscopy_regime: str,
               vol_a: torch.Tensor,
               vol_b: torch.Tensor | None = None,
               padding_xy=64,
               return_field=True,
               device=None):

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vol_a = vol_a.to(device=device, dtype=torch.float32)
    vol_b = vol_b.to(device=device, dtype=torch.float32) if vol_b is not None else None

    # Setup optics to match actual volume shape (Z,Y,X)
    optics_setup = setup_optics(shape=tuple(vol_a.shape), padding_xy=padding_xy,
                                microscopy_regime=microscopy_regime, return_field=return_field)
    model = imaging_model(optics_setup).to(device).eval()

    # Build batch as a LEAF tensor so gradients accumulate here
    batch = make_batch(vol_a, vol_b, n_a=8, n_b=(8 if vol_b is not None else 0))
    batch = batch.detach().clone().requires_grad_(True)   # ensure leaf

    # --- Batched forward timing ---
    start = time.time()
    out = model(batch)   # expect (B,H,W) or (B,H,W,1) depending on your optics wrapper
    t_batch = time.time() - start

    # --- Scalar loss + backward ---
    # Use a loss that works for both real and complex outputs
    if torch.is_complex(out):
        loss = (out.real**2 + out.imag**2).mean()
    else:
        loss = (out**2).mean()

    loss.backward()

    # --- Grad checks (CHECK INPUT grads, not output grads) ---
    grad = batch.grad
    grad_ok = (grad is not None) and torch.isfinite(grad).all().item() and (grad.abs().max().item() > 0)

    print(f"\n=== {microscopy_regime.upper()} ===")
    print("device:", device)
    print("batch.shape:", tuple(batch.shape))
    print("out.shape:", tuple(out.shape))
    print("out.dtype:", out.dtype, "| complex?", torch.is_complex(out))
    print("batch forward time:", t_batch)
    print("loss:", float(loss.detach().cpu()))
    print("batch.grad is None?", grad is None)
    if grad is not None:
        print("grad mean abs:", grad.abs().mean().item())
        print("grad max abs:", grad.abs().max().item())
        print("grad finite:", torch.isfinite(grad).all().item())
    print("✅ gradients OK" if grad_ok else "❌ gradients NOT OK")

    # --- Optional: timing loop mode ---
    t_loop = time_loop(model, batch.detach())
    print("loop forward time:", t_loop)

    return model, out.detach(), batch.grad.detach() if batch.grad is not None else None


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your test volume(s)
    vol_a = np.load("../test_data/vol_gauss_mult.npy") - 1.33
    vol_b = np.load("../test_data/vol_gauss_mult.npy") - 1.33

    vol_a = torch.tensor(vol_a, device=device)
    vol_b = torch.tensor(vol_b, device=device)

    # same cropping you did
    vol_a = vol_a[8:, 16:, :]
    vol_b = vol_b[8:, 16:, :]

    # Brightfield test
    test_model("brightfield", vol_a, vol_b, padding_xy=64, return_field=True, device=device)

    # Fluorescence test (if you have a fluorescence volume, pass it here)
    # Example: fluorescence volume should be nonnegative, sparse-ish
    fluor = torch.tensor(np.load("../test_data/vol_fluo.npy"), device=device)
    fluor = fluor[8:, 16:, :]
    test_model("fluorescence", fluor, None, padding_xy=64, return_field=False, device=device)
