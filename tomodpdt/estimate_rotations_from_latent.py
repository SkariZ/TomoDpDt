import torch
import numpy as np
from scipy.signal import savgol_filter
import scipy.signal as signal
from scipy.linalg import svd
import cv2
from skimage.restoration import unwrap_phase


def process_latent_space(
    z,
    frames,
    initial_axes=None,
    quaternions=None,
    initial_frames_per_rotation=None,
    peaks_period_range=None,  # [min_period, max_period] in frames
    window_length=11,
    polyorder=2,
    max_peaks=7,
    min_peaks=2,
    prominence=0.1,
    width=10,
    basis_functions=15,
    intial_axes_case='cv2_flow',
    rotation_period='2pi',
    **kwargs
):
    """
    Process the latent space to compute quaternions and peaks.

    Parameters:
    ----------
    z : torch.Tensor
        Latent space representation.
    frames : torch.Tensor
        Input image sequence to compute optical flow.
    initial_axes : str
        Initial rotation axis ('x', 'y', 'z'). Auto-estimated if None.
    quaternions : torch.Tensor
        Optional precomputed quaternions.
    initial_frames_per_rotation : int
        Initial guess of frames per rotation.
    peaks_period_range : list[int]
        [min_period, max_period] in frames for filtering detected peaks.
    window_length : int
        Window length for Savitzky-Golay smoothing.
    polyorder : int
        Polynomial order for smoothing.
    max_peaks : int
        Maximum number of peaks to detect.
    min_peaks : int
        Minimum number of peaks to detect.
    prominence : float
        Minimum prominence for peak detection.
    width : int
        Minimum width for peaks.
    basis_functions : int
        Number of basis functions for quaternion fitting.
    intial_axes_case : str
        Strategy to determine initial rotation axis ('cv2_flow' or 'std').

    Returns:
    -------
    dict
        Dictionary with:
        - 'quaternions'
        - 'coeffs'
        - 'basis'
        - 'peaks'
        - 'smoothed_distances'
    """

    if rotation_period == '2pi':
        rotation_period = 2 * np.pi
    elif rotation_period == 'pi':
        rotation_period = np.pi
    else:
        raise ValueError("rotation_period must be either 'pi' or '2pi'.")

    device = z.device
    if not isinstance(frames, torch.Tensor):
        frames = torch.tensor(frames)

    # Auto-determine initial axis if not given
    if initial_axes is None:
        if intial_axes_case == 'cv2_flow':
            flow_vectors = compute_optical_flow(frames[:, 0].cpu().numpy())
            initial_axes = classify_rotation_axis(flow_vectors)
        elif intial_axes_case == 'std':
            std_x = torch.std(frames[1:] - frames[-1:], dim=(0, 1, 2)).sum()
            std_y = torch.std(frames[1:] - frames[-1:], dim=(0, 1, 3)).sum()
            initial_axes = 'x' if std_x > std_y else 'y'

    # Compute smoothed distance signal
    res = np.array(1 - compute_normalized_distances(z).cpu().numpy())
    res = res * np.linspace(0.95, 1, len(res))
    try:
        res = savgol_filter(res, window_length=window_length, polyorder=polyorder)
    except Exception:
        pass
    res /= max(res)

    # If an initial period estimate is given, define an expected range
    if initial_frames_per_rotation is not None:
        low = int(initial_frames_per_rotation * 0.7)
        high = int(initial_frames_per_rotation * 1.3)
        peaks_period_range = [low, high]

        expected_peaks = len(z) / initial_frames_per_rotation
        min_peaks = max(2, int(expected_peaks * 0.6))
        max_peaks = max(min_peaks + 1, int(expected_peaks * 1.4))
    elif peaks_period_range is None:
        # Default broad range if nothing provided
        peaks_period_range = [10, len(z) // 2]

    # Detect peaks
    raw_peaks, = find_peaks(
        res, prominence=prominence, width=width
    )

    # Filter peaks based on period range (distance between consecutive peaks)
    if len(raw_peaks) > 1:
        diffs = np.diff(raw_peaks)
        valid_mask = (diffs >= peaks_period_range[0]) & (diffs <= peaks_period_range[1])
        # Always keep the first peak, then add valid ones
        peaks = [raw_peaks[0]]
        for i, keep in enumerate(valid_mask):
            if keep:
                peaks.append(raw_peaks[i + 1])
        peaks = np.array(peaks)
    else:
        peaks = raw_peaks

    # Clip to min/max allowed peaks
    if len(peaks) > max_peaks:
        peaks = peaks[:max_peaks]
    if len(peaks) < min_peaks:
        raise ValueError(f"Not enough valid peaks found ({len(peaks)})")

    # Interpolate angles between peaks
    total_timesteps = max(peaks) + 1
    angles = torch.zeros(total_timesteps)
    for i, t in enumerate(peaks):
        angles[t] = i * rotation_period
    for i in range(1, len(peaks)):
        start, end = peaks[i - 1], peaks[i]
        if end > start:
            angles[start:end] = torch.linspace(angles[start], angles[end], end - start)

    # Initialize rotation axis
    angle_rotations = torch.zeros(max(peaks), 4)
    angle_rotations[:, 0] = angles[:-1]

    if initial_axes == 'x':
        angle_rotations[:, 1] = 1
    elif initial_axes == 'y':
        angle_rotations[:, 2] = 1
    elif initial_axes == 'z':
        angle_rotations[:, 3] = 1

    # Convert to quaternions
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

    # Ensure quaternion continuity
    quaternions = ensure_quaternion_continuity(quaternions)

    # Enforce initial axis direction
    quaternions = enforce_initial_axis_direction(quaternions, axis=initial_axes)

    # Generate basis functions and coefficients
    basis = generate_basis_functions(quaternions.shape[0], basis_functions)
    coeffs = initialize_basis_functions(basis, quaternions)

    return {
        "quaternions": quaternions.to(device),
        "coeffs": coeffs.to(device),
        "basis": basis.to(device),
        "peaks": torch.tensor(peaks).to(device),
        "smoothed_distances": torch.tensor(res).to(device),
    }


def process_cross_correlation(
    frames, 
    normalize=True, 
    width=10,
    prominence=0.1,
    rotation_period="2pi",
    initial_axes=None,
    initial_axes_case='cv2_flow',
    basis_functions=15,
    **kwargs
    ):

    if not isinstance(frames, torch.Tensor):
        frames = torch.tensor(frames)

    # Keep track of device    
    device = frames.device

    # Unwrap phases
    frames = unwrap_phase_batch(frames)
    frames = torch.tensor(frames, dtype=torch.float32)

    # Compute cross-correlation series
    cc = compute_cc_series(frames, normalize=normalize)

    # Smooth cross-correlation series
    try:
        print("Smoothing cross-correlation series...")
        cc = savgol_filter(cc.cpu().numpy(), window_length=11, polyorder=2)
    except:
        print("Smoothing failed, using raw cross-correlation series...")
        cc = cc.cpu().numpy()
    cc = torch.tensor(cc, dtype=torch.float32)
    cc = cc / max(cc)


    # Compute angles from peaks
    angles, peaks = compute_angles_from_peaks(cc, n_frames=frames.shape[0], width=width, prominence=prominence, rotation_period=rotation_period)

    # Auto-determine initial axis if not given
    if initial_axes is None:
        if initial_axes_case == 'cv2_flow':
            flow_vectors = compute_optical_flow(frames[:, 0].cpu().numpy())
            initial_axes = classify_rotation_axis(flow_vectors)
        elif initial_axes_case == 'std':
            std_x = torch.std(frames[1:] - frames[-1:], dim=(0, 1, 2)).sum()
            std_y = torch.std(frames[1:] - frames[-1:], dim=(0, 1, 3)).sum()
            initial_axes = 'x' if std_x > std_y else 'y'

    # Convert angles to quaternions
    quaternions = quaternions_from_angles(angles, n_quaternions=len(angles), axis=initial_axes)
    quaternions = torch.tensor(quaternions, dtype=torch.float32)

    # Ensure quaternion continuity
    quaternions = ensure_quaternion_continuity(quaternions)

    # Enforce initial axis direction
    quaternions = enforce_initial_axis_direction(quaternions, axis=initial_axes)

    basis = generate_basis_functions(quaternions.shape[0], basis_functions)
    coeffs = initialize_basis_functions(basis, quaternions)

    # Return processed data as dictionary and torch tensors on the same device
    return {
        "quaternions": quaternions.to(device),
        "coeffs": coeffs.to(device),
        "basis": basis.to(device),
        "peaks": torch.tensor(peaks).to(device),
        "smoothed_distances": torch.tensor(cc).to(device)
        }

def ensure_quaternion_continuity(quaternions):
    """Flip quaternion signs to enforce temporal continuity."""
    quats = quaternions.clone()
    for i in range(1, quats.shape[0]):
        if torch.dot(quats[i - 1], quats[i]) < 0:
            quats[i] = -quats[i]
    return quats

def enforce_initial_axis_direction(quaternions, axis='x'):
    """
    Ensures the first quaternion's rotation axis starts in the positive direction.

    Parameters
    ----------
    quaternions : torch.Tensor
        Quaternion sequence (N, 4)
    axis : str
        Axis to enforce positivity ('x', 'y', or 'z')

    Returns
    -------
    torch.Tensor
        Possibly flipped quaternion sequence.
    """
    axis_idx = {'x': 1, 'y': 2, 'z': 3}[axis]
    if quaternions[0, axis_idx] < 0:
        quaternions = -quaternions
    return quaternions


def generate_basis_functions(N_points, num_basis, t_start=0.1, t_end=0.9):
    """
    Generate smooth, fixed-frequency sine/cosine basis functions with a constant term.

    Args:
        N_points (int): Number of time points.
        num_basis (int): Total number of basis functions (including constant term).
        t_start (float): Start of time interval.
        t_end (float): End of time interval.

    Returns:
        torch.Tensor: (N_points, num_basis) basis function matrix.
    """
    t = torch.linspace(t_start, t_end, N_points).unsqueeze(1)  # (N_points, 1)

    n_harmonics = (num_basis - 1) // 2

    cos_terms = [torch.cos(2 * torch.pi * (k + 1) * t) for k in range(n_harmonics)]
    sin_terms = [torch.sin(2 * torch.pi * (k + 1) * t) for k in range(n_harmonics)]
    const_term = [torch.ones_like(t)]

    basis = torch.cat(cos_terms + sin_terms + const_term, dim=1)

    # Truncate if num_basis is not odd (e.g., to remove extra sine/cosine)
    return basis[:, :num_basis]


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
               min_peaks=2, width=10, prominence=0.25):
    """Find peaks in smoothed distance data."""

    distance_range = (peaks_period_range[0], peaks_period_range[1], 10)

    for dist in range(*distance_range):
        try:
            peaks = signal.find_peaks(res, distance=dist, prominence=prominence, width=width)[0]
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
    try:
        _, _, Vt = svd(displacements, full_matrices=False)
    except np.linalg.LinAlgError:
        # If SVD fails, fallback to simple classification
        return "x" if np.mean(displacements[:, 0]) > np.mean(displacements[:, 1]) else "y"
    
    principal_axis = Vt[0]  # First principal component

    # Classify based on the dominant motion direction
    if abs(principal_axis[0]) > abs(principal_axis[1]):
        return "x"  # Horizontal rotation
    else:
        return "y"  # Vertical rotation


def unwrap_phase_batch(E_batch):
    """
    Unwrap the phase for a batch of complex images.
    
    Parameters:
    - E_batch: list or array of complex 2D arrays
    
    Returns:
    - list of unwrapped phases
    """
    return [unwrap_phase(torch.angle(E).cpu().numpy()) for E in E_batch]


def cross_correlation_2d(a, b):
    """
    Compute normalized cross-correlation between two 2D arrays.
    
    Parameters:
    - a, b: 2D arrays
    
    Returns:
    - float: maximum of cross-correlation
    """
    fft_a = torch.fft.fft2(a)
    fft_b = torch.fft.fft2(b)
    cc = torch.fft.ifft2(fft_a * fft_b)
    return torch.abs(cc).max()


def compute_cc_series(PU, normalize=False):
    """
    Compute cross-correlation series relative to the first frame.
    
    Parameters:
    - PU: list of 2D unwrapped phases
    
    Returns:
    - np.ndarray: cross-correlation values
    """
    n = len(PU)
    
    # Normalize phases if required
    if normalize:
        PU = [(p - torch.mean(p)) / torch.std(p) for p in PU]

    # Compute cross-correlation with respect to the first frame
    cc = [cross_correlation_2d(PU[0], PU[i]) for i in range(n)]

    return torch.tensor(cc)


def compute_angles_from_peaks(
    cc,
    n_frames,
    width=10,
    prominence=0.5,
    rotation_period="2pi",
    ):
    """
    Compute rotation angles from cross-correlation peaks.

    Parameters
    ----------
    cc : array-like
        Cross-correlation signal over time.
    n_frames : int
        Total number of frames.
    width : int, optional
        Minimum width of peaks for scipy.signal.find_peaks.
    prominence : float, optional
        Peak prominence threshold for robust detection.
    rotation_period : str, optional
        Defines how much rotation each detected cycle represents:
        - 'pi'  : Each peak → π radians (180°)
        - '2pi' : Each peak → 2π radians (360°) [default]

    Returns
    -------
    angles : torch.Tensor
        Interpolated rotation angles over time.
    peaks : np.ndarray
        Detected peak indices.
    """

    cc = np.asarray(cc)
    peaks, props = signal.find_peaks(cc, width=width, prominence=prominence)

    if len(peaks) < 2:
        raise ValueError("Not enough peaks detected for angle estimation.")

    # Choose angular increment based on model
    if rotation_period.lower() == "pi":
        delta_angle = np.pi
    elif rotation_period.lower() == "2pi":
        delta_angle = 2 * np.pi
    else:
        raise ValueError("rotation_period must be either 'pi' or '2pi'.")

    angles = torch.zeros(n_frames)

    # Assign continuous angles between peaks
    for i, pk in enumerate(peaks):
        if i == 0:
            start, end = 0, pk
            angles[start:end] = torch.linspace(0, delta_angle, end - start)
        else:
            start, end = peaks[i - 1], pk
            angles[start:end] = torch.linspace(delta_angle * (i - 1), delta_angle * i, end - start)

    # Extend beyond last peak if needed
    last_pk = peaks[-1]
    remaining = n_frames - last_pk
    if remaining > 0:
        angles[last_pk:] = torch.linspace(delta_angle * (len(peaks) - 1), delta_angle * len(peaks), remaining)

    return angles, peaks



def quaternions_from_angles(th, n_quaternions, axis='y'):
    """
    Initialize quaternions along a single rotation axis (y-axis in this example).
    
    Parameters:
    - th: array of angles
    - n_quaternions: number of quaternions to generate
    
    Returns:
    - np.ndarray: (N, 4) array of quaternions
    """
    Q_start = torch.zeros((n_quaternions, 4))
    Q_start[:, 0] = torch.cos(-th / 2)  # w

    sin_th_2 = torch.sin(-th / 2)
    if axis == 'x':
        Q_start[:, 1] = sin_th_2  # x
    elif axis == 'y':
        Q_start[:, 2] = sin_th_2  # y
    elif axis == 'z':
        Q_start[:, 3] = sin_th_2  # z

    return Q_start


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