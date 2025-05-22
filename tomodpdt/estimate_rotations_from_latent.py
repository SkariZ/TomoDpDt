import torch
import numpy as np
from scipy.signal import savgol_filter
import scipy.signal as signal
from scipy.linalg import svd
import cv2


def process_latent_space(
    z,
    frames,
    initial_axes=None,
    quaternions=None,
    initial_frames_per_rotation=None,
    peaks_period_range=None,
    window_length=11,
    polyorder=2,
    max_peaks=7,
    min_peaks=2,
    prominence=0.5,
    height_factor=0.775,
    basis_functions=12,
    intial_axes_case='cv2_flow',
    **kwargs
):
    """
    Process the latent space to compute quaternions and peaks.

    Parameters:
    - z (torch.Tensor): Latent space representation.
    - frames (torch.Tensor): Frames to compute optical flow.
    - initial_axes (str): Initial rotation axis ('x', 'y', 'z').
    - quaternions (torch.Tensor): Precomputed quaternions.
    - initial_frames_per_rotation (int): Initial estimate of frames per rotation.
    - peaks_period_range (list): Range for peak detection.
    - window_length (int): Window length for Savitzky-Golay filter.
    - polyorder (int): Polynomial order for Savitzky-Golay filter.
    - max_peaks (int): Maximum number of peaks to detect.
    - min_peaks (int): Minimum number of peaks to detect.
    - prominence (float): Prominence for peak detection.
    - height_factor (float): Height factor for peak detection.
    - basis_functions (int): Number of basis functions.
    - intial_axes_case (str): Method to determine initial axes ('cv2_flow' or 'std').

    Returns:
    - dict: Processed data with quaternions, coefficients, basis functions, peaks, and smoothed distances.
    """

    device = z.device

    if not isinstance(frames, torch.Tensor):
        frames = torch.tensor(frames)

    # Auto-determine initial axis if not given
    if initial_axes is None:
        if intial_axes_case == 'cv2_flow':
            flow_vectors = compute_optical_flow(frames[:, 0].cpu().numpy())
            initial_axes = classify_rotation_axis(flow_vectors)
        else:
            std_x = torch.std(frames[1:] - frames[-1:], dim=(0, 1, 2)).sum()
            std_y = torch.std(frames[1:] - frames[-1:], dim=(0, 1, 3)).sum()
            initial_axes = 'x' if std_x > std_y else 'y'

    # Compute distances and smooth them
    res = np.array(1 - compute_normalized_distances(z).cpu().numpy())
    res = res * np.linspace(0.95, 1, len(res))
    res = savgol_filter(res, window_length=window_length, polyorder=polyorder)
    res /= max(res)

    # Adjust peak detection parameters if an initial period estimate is given
    if initial_frames_per_rotation is not None:
        # Set peak search range around the expected period ±30%
        low = int(initial_frames_per_rotation * 0.7)
        high = int(initial_frames_per_rotation * 1.3)
        peaks_period_range = [low, high]

        # Ensure the range is within reasonable limits
        expected_peaks = len(z) / initial_frames_per_rotation
        min_peaks = max(2, int(expected_peaks * 0.6))
        max_peaks = max(min_peaks + 1, int(expected_peaks * 1.4))
    else:
        # Default peak search range
        if peaks_period_range is None:
            peaks_period_range = [20, 100]

    # Detect peaks
    peaks = find_peaks(
        res, 
        peaks_period_range=peaks_period_range, 
        max_peaks=max_peaks, 
        min_peaks=min_peaks, 
        prominence=prominence,
        height_factor=height_factor
        )

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
    
    if quaternions is None:
        quaternions = torch.cat((qw.unsqueeze(1), q_xyz), dim=1)
    else:
        peaks = torch.tensor([0, len(quaternions) - 1])

    # Generate basis functions and initialize coefficients
    basis = generate_basis_functions(quaternions.shape[0], basis_functions)
    coeffs = initialize_basis_functions(basis, quaternions)

    # Return processed data as dictionary and torch tensors on the same device
    return {
        "quaternions": quaternions.to(device),
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


def find_peaks(res, peaks_period_range=[20, 100], max_peaks=7,
               min_peaks=2, prominence=0.5, height_factor=0.775):
    """Find peaks in smoothed distance data."""
    height = height_factor * np.max(res)
    distance_range = (peaks_period_range[0], peaks_period_range[1], 10)

    for dist in range(*distance_range):
        try:
            peaks = signal.find_peaks(res, distance=dist, height=height,
                                      prominence=prominence)[0]
        except:
            peaks = []
            
        if min_peaks < len(peaks) < max_peaks:
            break

    peaks = np.append(0, peaks)  # Ensure first peak at index 0

    if len(peaks) < min_peaks:
        peaks = np.append(peaks, len(res)-1)  # Add last peak if necessary

    # If there is a higher height after the last peak, add it
    if len(peaks) > 1 and res[-1] > res[peaks[-1]]:
        peaks = np.append(peaks, len(res)-1)

    return peaks


def compute_optical_flow(frames):
    """
    Computes dense optical flow between consecutive frames.
    :param frames: NumPy array of shape (T, 64, 64) with values in range [0,1].
    :return: Motion vectors (dx, dy) for sampled points.
    """
    T, H, W = frames.shape
    flow_vectors = []

    for t in range(T - 1):
        prev_gray = (frames[t] * 255).astype(np.uint8)  # Convert to 0-255
        next_gray = (frames[t + 1] * 255).astype(np.uint8)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

        # Sample motion vectors at every 4 pixels
        y, x = np.mgrid[0:H:4, 0:W:4].reshape(2, -1).astype(int)
        dx, dy = flow[y, x].T
        points = np.column_stack([x, y, dx, dy])
        flow_vectors.append(points)

    return np.vstack(flow_vectors)


def classify_rotation_axis(flow_vectors):
    """
    Uses PCA to classify whether the main axis of rotation is along X or Y.
    Returns 'X' if horizontal, 'Y' if vertical.
    """
    displacements = flow_vectors[:, 2:]  # (dx, dy)

    # PCA using Singular Value Decomposition (SVD)
    _, _, Vt = svd(displacements, full_matrices=False)
    principal_axis = Vt[0]  # First principal component

    # Classify based on the dominant motion direction
    if abs(principal_axis[0]) > abs(principal_axis[1]):
        return "x"  # Horizontal rotation
    else:
        return "y"  # Vertical rotation


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