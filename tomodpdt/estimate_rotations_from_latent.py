import torch
import numpy as np
from scipy.signal import savgol_filter
import scipy.signal as signal

def process_latent_space(
    z,
    frames,
    initial_axes=None,
    peaks_period_range=[20, 100],
    window_length=11,
    polyorder=2,
    max_peaks=7,
    min_peaks=2,
    basis_functions=10,
):
    """
    Process the latent space to compute quaternions and peaks.

    Parameters:
    - z (torch.Tensor): Latent space representation (N, 2).
    - frames (torch.Tensor): Input frames.
    - max_n_projections (int): Maximum number of frames to process.
    - step_size (int): Step size for frame selection.
    - normalize (bool): Whether to normalize frames.
    - initial_axes (str): Axis for rotations ('x', 'y', 'z').
    - peaks_period_range (list): Range for peak detection.
    - window_length (int): Window length for smoothing.
    - polyorder (int): Polynomial order for smoothing.
    - max_peaks (int): Maximum allowed number of peaks.
    - min_peaks (int): Minimum required number of peaks.

    Returns:
    - dict: Processed data containing quaternions, peaks, and smoothed distances etc.
    """

    # Ensure frames and latent space are tensors
    if not isinstance(frames, torch.Tensor):
        frames = torch.tensor(frames)

    if initial_axes is None:
        # Estimate the initial axis by looking at the std in the latent space
        std_x = torch.std(frames, dim=(0, 1, 2)).sum()
        std_y = torch.std(frames, dim=(0, 1, 3)).sum()

        if std_x > std_y:
            initial_axes = 'x'
        else:
            initial_axes = 'y'

    # Return device
    device = z.device

    # Compute distances and smooth them
    res = np.array(1 - compute_normalized_distances(z).cpu().numpy())
    res = res * np.linspace(0.95, 1, len(res))
    res = savgol_filter(res, window_length=window_length, polyorder=polyorder)
    res /= max(res)  # Normalize

    # Detect peaks
    peaks = find_peaks(res, peaks_period_range=peaks_period_range, max_peaks=max_peaks, min_peaks=min_peaks)

    # Interpolate angles based on peaks
    total_timesteps = max(peaks) + 1
    angles = torch.zeros(total_timesteps)

    for i, t in enumerate(peaks):
        angles[t] = i * 2 * torch.pi

    for i in range(1, len(peaks)):
        start, end = peaks[i - 1], peaks[i]
        if end > start:
            angles[start:end] = torch.linspace(angles[start], angles[end], end - start)

    # Initialize angle rotations
    angle_rotations = torch.zeros(max(peaks), 4)
    angle_rotations[:, 0] = angles[:-1]


    if initial_axes == 'x':
        angle_rotations[:, 1] = 1  # [angle, 1, 0, 0]
    elif initial_axes == 'y':
        angle_rotations[:, 2] = 1  # [angle, 0, 1, 0]
    elif initial_axes == 'z':
        angle_rotations[:, 3] = 1  # [angle, 0, 0, 1]

    # Convert axis-angle to quaternions
    axes = angle_rotations[:, 1:]
    norms = torch.norm(axes, dim=1, keepdim=True)
    axes = axes / norms

    half_angles = angle_rotations[:, 0] / 2
    qw = torch.cos(half_angles)
    sin_half_angles = torch.sin(half_angles)
    q_xyz = axes * sin_half_angles.unsqueeze(1)
    quaternions = torch.cat((qw.unsqueeze(1), q_xyz), dim=1)

    # Generate basis functions and initialize coefficients
    basis = generate_basis_functions(quaternions.shape[0], basis_functions)
    coeffs = initialize_basis_functions(basis, quaternions)

    # Return processed data as dictionary and torch tensors on the same device

    return {
        "quaternions": quaternions.to,
        "coeffs": coeffs.to(device),
        "basis": basis.to(device),
        "peaks": torch.tensor(peaks).to(device),
        "smoothed_distances": torch.tensor(res).to(device)
        }


def generate_basis_functions(N_points, num_basis):
    """Generate basis functions for time points."""

    t = torch.linspace(0.2, 1, N_points).unsqueeze(1)  # Time points (num_points, 1)

    basis = torch.cat(
        [torch.ones_like(t)] +  # Constant term
        [torch.cos(2 * torch.pi * (k + 1) * t) for k in range(num_basis // 2)] +
        [torch.sin(2 * torch.pi * (k + 1) * t) for k in range(num_basis // 2)],
        dim=1
    )
    return basis

def initialize_basis_functions(basis, quaternions):
    """
    Initialize the basis functions using the initial quaternion values
    """

    # Generate the basis functions. Solve the least squares problem to find the coefficients
    coeffs = torch.linalg.lstsq(basis, quaternions).solution

    return coeffs

def compute_normalized_distances(z):
    """Compute normalized distances from the first point in latent space."""
    d0 = z[0]  # Reference point (first row)
    dists = torch.sqrt(((z - d0) ** 2).sum(dim=1))
    return dists / dists.max()

def find_peaks(res, peaks_period_range=[20, 100], max_peaks=7, min_peaks=2):
        """Find peaks in smoothed distance data."""
        height = 0.8 * np.max(res)
        distance_range = (peaks_period_range[0], peaks_period_range[1], 10)

        for dist in range(*distance_range):
            peaks = signal.find_peaks(res, distance=dist, height=height, prominence=0.7)[0]
            if min_peaks < len(peaks) < max_peaks:
                break

        peaks = np.append(0, peaks)  # Ensure first peak at index 0

        if len(peaks) < min_peaks:
            peaks = np.append(peaks, len(res) - 1)  # Add last peak if necessary

        return peaks

# Example usage
if __name__ == "__main__":

    # Generate some random data
    z = torch.randn(100, 2)
    frames = torch.randn(100, 2, 32, 32)

    # Process latent space
    processed_data = process_latent_space(z, frames)

    # Print processed data
    print(processed_data)
    print(processed_data["quaternions"].shape)
    print(processed_data["coeffs"].shape)
    print(processed_data["basis"].shape)
    print(processed_data["peaks"])
    print(processed_data["smoothed_distances"].shape)