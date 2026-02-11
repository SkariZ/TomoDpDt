import torch
import numpy as np
from scipy.signal import savgol_filter
import scipy.signal as signal
from scipy.linalg import svd
import cv2
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt

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
    prominence=0.2,
    width=15,
    basis_functions=15,
    initial_axes_case='cv2_flow',
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
        if initial_axes_case == 'cv2_flow':
            flow_vectors = compute_optical_flow(frames[:, 0].cpu().numpy())
            initial_axes = classify_rotation_axis(flow_vectors)
        elif initial_axes_case == 'std':
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
        peaks_period_range = [len(z) // 8, len(z) // 2]

    # Detect peaks
    raw_peaks = find_peaks(
        res,
        peaks_period_range=peaks_period_range,
        max_peaks=max_peaks,
        min_peaks=min_peaks, 
        prominence=prominence, 
        width=width
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
        raise ValueError("Not enough peaks detected for angle estimation.")

    # Interpolate angles between peaks
    total_timesteps = max(peaks) + 1
    angles = torch.zeros(total_timesteps)
    for i, t in enumerate(peaks):
        angles[t] = i * rotation_period
    for i in range(1, len(peaks)):
        start, end = peaks[i - 1], peaks[i]
        if end > start:
            angles[start:end] = torch.linspace(angles[start], angles[end], end - start)

    # --- Extend beyond last peak using the same angular speed as the last segment ---
    last_pk = peaks[-1]
    n_frames = len(z)
    remaining = n_frames - last_pk

    if remaining > 0:
        if len(peaks) > 1:
            # Estimate angular speed (radians per frame) from last segment
            prev_pk = peaks[-2]
            frames_per_rotation = last_pk - prev_pk
            angular_speed = rotation_period / frames_per_rotation
        else:
            # Fallback if only one peak detected
            angular_speed = rotation_period / n_frames

        # Continue rotation smoothly with the same angular speed
        last_angle = angles[last_pk]
        angles_remain = last_angle + torch.arange(1, remaining + 1) * angular_speed

        # Concatenate to form full angle series
        angles = torch.cat((angles, angles_remain), dim=0)

    # Trim angles to match latent space length. For safety.
    angles = angles[:len(z)]

    # Initialize rotation axis
    angle_rotations = torch.zeros(len(angles)-1, 4)
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


def process_latent_phase(
    z,
    frames=None,
    initial_axes="y",
    rotation_period="2pi",   # "pi", "2pi", or "auto"
    basis_functions=15,
    smooth_window=11,
    smooth_poly=2,
    min_peak_distance=15,    # frames, to suppress spurious close peaks
    **kwargs
):
    """
    Phase-based rotation initialization from 2D latent space.
    Uses incremental signed angle (atan2 of cross/dot) -> cumulative angle.
    Returns SAME dict schema as your other methods.
    """

    device = z.device
    z = z.detach()

    # --- 0) center to reduce drift (critical) ---
    zc = z - z.mean(dim=0, keepdim=True)  # (N,2)

    # Optional: make it more circular (helps a lot when ellipse-like)
    # (simple PCA whitening)
    Z = zc
    cov = (Z.T @ Z) / (Z.shape[0] - 1 + 1e-8)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    W = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals + 1e-6)) @ eigvecs.T
    zw = (Z @ W.T)

    # --- 1) incremental signed angle between consecutive vectors ---
    v0 = zw[:-1]                       # (N-1,2)
    v1 = zw[1:]                        # (N-1,2)
    dot = (v0 * v1).sum(dim=1)         # (N-1,)
    cross = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]  # (N-1,)
    dtheta = torch.atan2(cross, dot + 1e-8)            # in (-pi, pi)

    # cumulative angle (starts at 0)
    theta = torch.cat([torch.zeros(1, device=device), torch.cumsum(dtheta, dim=0)], dim=0)  # (N,)

    # --- 2) smooth cumulative angle (optional but helps) ---
    theta_np = theta.detach().cpu().numpy()
    if smooth_window is not None and smooth_window > 3 and smooth_window < len(theta_np):
        if smooth_window % 2 == 0:
            smooth_window += 1
        try:
            theta_np = savgol_filter(theta_np, window_length=smooth_window, polyorder=smooth_poly)
        except Exception:
            pass
    theta_s = torch.tensor(theta_np, device=device, dtype=torch.float32)

    # --- 3) pick period (pi vs 2pi) ---
    if rotation_period == "pi":
        period = np.pi
    elif rotation_period == "2pi":
        period = 2 * np.pi
    elif rotation_period == "auto":
        # choose pi or 2pi based on "return-to-start" quality:
        # compute candidate peaks for both, then keep the one whose peaks land closer to z0.
        def get_peaks_for(period_val):
            k = torch.floor(theta_s / period_val)
            crossings = torch.where(k[1:] > k[:-1])[0] + 1
            peaks = torch.cat([torch.tensor([0], device=device), crossings]).unique()
            # enforce min distance
            if len(peaks) > 1 and min_peak_distance is not None:
                kept = [int(peaks[0])]
                for t in peaks[1:].tolist():
                    if t - kept[-1] >= min_peak_distance:
                        kept.append(int(t))
                peaks = torch.tensor(kept, device=device)
            return peaks

        peaks_pi = get_peaks_for(np.pi)
        peaks_2pi = get_peaks_for(2 * np.pi)

        z0 = zw[0]
        def score(peaks):
            if len(peaks) < 2:
                return 1e9
            d = torch.norm(zw[peaks] - z0, dim=1)
            # robust: median distance at peaks (smaller = better)
            return float(torch.median(d).item())

        period = np.pi if score(peaks_pi) < score(peaks_2pi) else 2 * np.pi
    else:
        raise ValueError("rotation_period must be 'pi', '2pi', or 'auto'")

    # --- 4) peaks by thresholding cumulative angle crossings ---
    k = torch.floor(theta_s / period)
    crossings = torch.where(k[1:] > k[:-1])[0] + 1
    peaks = torch.cat([torch.tensor([0], device=device), crossings]).unique()

    # suppress peaks too close together (kills the intermittent false ones)
    if len(peaks) > 1 and min_peak_distance is not None:
        kept = [int(peaks[0])]
        for t in peaks[1:].tolist():
            if t - kept[-1] >= min_peak_distance:
                kept.append(int(t))
        peaks = torch.tensor(kept, device=device)

    if len(peaks) < 2:
        raise ValueError("Not enough rotations detected from latent phase.")

    # --- 5) angles timeline (piecewise linear between peaks) ---
    n = z.shape[0]
    angles = torch.zeros(n, device=device, dtype=torch.float32)

    for i, t in enumerate(peaks.tolist()):
        angles[t] = i * float(period)

    for i in range(1, len(peaks)):
        s = int(peaks[i - 1].item())
        e = int(peaks[i].item())
        if e > s:
            angles[s:e] = torch.linspace(angles[s].item(), angles[e].item(), e - s, device=device)

    # extend tail
    last_pk = int(peaks[-1].item())
    if last_pk < n - 1:
        if len(peaks) > 1:
            prev_pk = int(peaks[-2].item())
            frames_per = max(1, last_pk - prev_pk)
            w = float(period) / frames_per
        else:
            w = float(period) / max(1, n)
        tail = torch.arange(0, n - last_pk, device=device) * w
        angles[last_pk:] = angles[last_pk] + tail

    # --- 6) angles -> quaternions ---
    axes = torch.zeros((n, 3), device=device)
    if initial_axes == "x":
        axes[:, 0] = 1.0
    elif initial_axes == "y":
        axes[:, 1] = 1.0
    else:
        axes[:, 2] = 1.0

    half = angles / 2.0
    qw = torch.cos(half)
    qxyz = axes * torch.sin(half).unsqueeze(1)
    quaternions = torch.cat([qw.unsqueeze(1), qxyz], dim=1)

    quaternions = ensure_quaternion_continuity(quaternions)
    quaternions = enforce_initial_axis_direction(quaternions, axis=initial_axes)

    basis = generate_basis_functions(quaternions.shape[0], basis_functions).to(device)
    coeffs = initialize_basis_functions(basis, quaternions).to(device)

    # for plotting: map cumulative angle to [0,1] saw-ish signal whose maxima correspond to peaks
    cc = (torch.cos((theta_s % float(period)) / float(period) * 2 * np.pi) + 1.0) * 0.5
    cc = cc / (cc.max() + 1e-8)

    return {
        "quaternions": quaternions.to(device),
        "coeffs": coeffs.to(device),
        "basis": basis.to(device),
        "peaks": peaks.to(device),
        "smoothed_distances": cc.to(device),
    }

def process_cross_correlation(
    frames, 
    normalize=False, 
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

    frames_original = frames.clone()

    # Keep track of device    
    device = frames.device

    if frames.shape[1] == 1:
        # use simple normalized correlation to first frame
        x0 = frames[0, 0]
        cc = []
        for i in range(frames.shape[0]):
            xi = frames[i, 0]
            cc.append(torch.mean((x0 - x0.mean()) * (xi - xi.mean())) / (x0.std()*xi.std() + 1e-6))
        cc = torch.stack(cc)
    else:
        # Unwrap phases
        frames = unwrap_phase_batch(frames)
        frames = torch.tensor(frames, dtype=torch.float32)

        # Compute cross-correlation series
        cc = compute_cc_series(frames, normalize=normalize)

    # Smooth cross-correlation series
    try:
        cc = savgol_filter(cc.cpu().numpy(), window_length=11, polyorder=2)
    except:
        cc = cc.cpu().numpy()

    cc = torch.tensor(cc, dtype=torch.float32)
    cc = cc / max(cc)

    # Compute angles from peaks
    angles, peaks = compute_angles_from_peaks(cc, n_frames=frames.shape[0], width=width, prominence=prominence, rotation_period=rotation_period)

    # Auto-determine initial axis if not given
    if initial_axes is None:
        if initial_axes_case == 'cv2_flow':
            flow_vectors = compute_optical_flow(frames_original[:, 0].cpu().numpy())
            initial_axes = classify_rotation_axis(flow_vectors)
        elif initial_axes_case == 'std':
            std_x = torch.std(frames_original[1:] - frames_original[-1:], dim=(0, 1, 2)).sum()
            std_y = torch.std(frames_original[1:] - frames_original[-1:], dim=(0, 1, 3)).sum()
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

    # If there is a high height after the last peak, add it
    if len(peaks) > 1 and res[-1] >= res[peaks[-1]]*0.9 and (len(res)-1 - peaks[-1]) >= int(peaks_period_range[0]*0.5):
        peaks = np.append(peaks, len(res)-1)

    # Check if there are any outlier peaks that have a very low value compared to the others
    peak_values = res[peaks]
    med_peak_value = np.median(peak_values)
    filtered_peaks = [peaks[0]]  # Always keep the first peak
    for pk in peaks[1:]:
        if res[pk] >= 0.66 * med_peak_value:
            filtered_peaks.append(pk)

    return filtered_peaks


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


def unwrap_phase_batch_old(E_batch):
    """
    Unwrap the phase for a batch of complex images.
    
    Parameters:
    - E_batch: list or array of complex 2D arrays
    
    Returns:
    - list of unwrapped phases
    """

    # If input is a torch tensor with real values, and 2 channels (real, imag), convert to complex
    if isinstance(E_batch, torch.Tensor) and E_batch.dim() == 4 and E_batch.size(1) == 2:
        E_batch = E_batch[:, 0] + 1j * E_batch[:, 1]

    # Make sure E_batch is complex
    E_batch = E_batch.type(torch.complex64)

    return [unwrap_phase(torch.angle(E).cpu().numpy()) for E in E_batch]


def unwrap_phase_batch(E_batch):
    """
    E_batch: torch.Tensor of shape (T,2,H,W) representing complex (real, imag)
             OR complex torch tensor (T,H,W)
    returns: torch.Tensor (T,H,W) float32
    """
    device = E_batch.device if isinstance(E_batch, torch.Tensor) else "cpu"

    # convert 2-channel real/imag -> complex numpy
    if isinstance(E_batch, torch.Tensor) and E_batch.dim() == 4 and E_batch.size(1) == 2:
        E = (E_batch[:, 0] + 1j * E_batch[:, 1]).detach().cpu().numpy()
    elif isinstance(E_batch, torch.Tensor) and torch.is_complex(E_batch):
        E = E_batch.detach().cpu().numpy()
    else:
        # already numpy or something
        E = np.asarray(E_batch)

    # unwrap each frame’s phase
    out = []
    for k in range(E.shape[0]):
        ph = np.angle(E[k])
        out.append(unwrap_phase(ph))
    out = np.stack(out, axis=0)

    return torch.tensor(out, dtype=torch.float32, device=device)


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
    prominence=0.25,
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
    peaks, props = adaptive_find_peaks(
        cc,
        width=width,
        prominence=prominence,
        min_peaks=2,
        max_peaks=10,
        max_tries=50,
    )

    if len(peaks) < 2:
        raise ValueError("Not enough peaks detected for angle estimation.")

    peak_values = cc[peaks]
    med_peak_value = np.median(peak_values)
    filtered_peaks = [] 
    for i, pk in enumerate(peak_values):
        if pk >= 0.66 * med_peak_value:
            filtered_peaks.append(peaks[i])
    peaks = np.array(filtered_peaks)

    if len(peaks) < 2:
        raise ValueError("Not enough valid peaks detected after filtering.")

    # Choose angular increment based on model
    if rotation_period.lower() == "pi":
        delta_angle = torch.pi
    elif rotation_period.lower() == "2pi":
        delta_angle = 2 * torch.pi
    else:
        raise ValueError("rotation_period must be either 'pi' or '2pi'.")

    angles = torch.zeros(n_frames, dtype=torch.float32)

    # If the first peak is not at index 0, we can optionally set it to 0 or just start from the first detected peak
    if peaks[0] != 0:
        peaks = np.insert(peaks, 0, 0)  # Ensure first peak at index 0

    # Assign continuous angles between peaks
    for i, pk in enumerate(peaks):
        angles[pk] = i * delta_angle
        if i > 0:
            start, end = peaks[i - 1], pk
            if end > start:
                angles[start:end] = torch.linspace(angles[start], angles[end], end - start)


    # Extend beyond last peak if needed
    last_pk = peaks[-1]
    remaining = n_frames - last_pk
    if remaining > 0:
        angles[last_pk:] = torch.linspace(delta_angle * (len(peaks) - 1), delta_angle * len(peaks), remaining)

    return angles, peaks


def adaptive_find_peaks(
    cc,
    width=10,
    prominence=0.1,
    min_peaks=2,
    max_peaks=10,
    max_tries=10,
    prominence_decay=0.5,
    width_decay=0.8,
):
    """
    Robustly find peaks by relaxing parameters until peaks are found.

    Parameters
    ----------
    cc : np.ndarray
        1D cross-correlation signal.
    width : float
        Initial width for scipy.signal.find_peaks.
    prominence : float
        Initial prominence for peak detection.
    min_peaks, max_peaks : int
        Desired range of valid peak counts.
    max_tries : int
        How many relaxation steps to attempt.
    prominence_decay : float
        Multiplicative factor to reduce prominence each iteration.
    width_decay : float
        Multiplicative factor to reduce width each iteration.

    Returns
    -------
    peaks : np.ndarray
        Indices of detected peaks.
    props : dict
        Peak properties from scipy.signal.find_peaks.
    """

    peaks, props = np.array([]), {}
    p, w = prominence, width

    for i in range(max_tries):
        peaks, props = signal.find_peaks(cc, prominence=p, width=w)
        if min_peaks <= len(peaks) <= max_peaks:
            print(f"✅ Found {len(peaks)} peaks after {i+1} tries (prom={p:.4f}, width={w:.2f})")
            break
        # Gradually relax criteria
        p *= prominence_decay
        w *= width_decay

    if len(peaks) < min_peaks:
        print(f" Could not find enough peaks (found {len(peaks)} after {max_tries} tries).")

    return peaks, props


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


def swap_axis_case(quats: torch.Tensor, to_axis: str):
    """
    quats: (T,4) [w,x,y,z], where ONLY ONE of x/y/z is nonzero per row.
    to_axis: "x" | "y" | "z"
    Returns new quats with same w and same sin(theta/2) magnitude, but placed on chosen axis.
    """
    assert quats.ndim == 2 and quats.shape[1] == 4, "Expected (T,4) [w,x,y,z]"
    assert to_axis in ("x", "y", "z")

    q = quats.clone()

    # find the current active axis per row (which of x,y,z is nonzero)
    v = q[:, 1:]                     # (T,3)
    amp = v.abs().amax(dim=1)        # (T,) magnitude of the active component (abs)
    sign = v.sign()
    # pick sign from whichever component is active (max abs)
    idx = v.abs().argmax(dim=1)      # (T,)
    s = sign[torch.arange(q.shape[0], device=q.device), idx]  # (T,)
    s = torch.where(s == 0, torch.ones_like(s), s)            # avoid 0 sign

    # make new xyz
    v_new = torch.zeros_like(v)
    j = {"x": 0, "y": 1, "z": 2}[to_axis]
    v_new[:, j] = s * amp

    q[:, 1:] = v_new
    return q

def make_3_axis_candidates(quats: torch.Tensor):
    return {
        "x": swap_axis_case(quats, "x"),
        "y": swap_axis_case(quats, "y"),
        "z": swap_axis_case(quats, "z"),
    }


def plot_signal_with_peaks(signal, peaks, title="", ylabel="signal"):
    """
    signal: (T,) torch or numpy
    peaks: 1D iterable of indices
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()
    if isinstance(peaks, torch.Tensor):
        peaks = peaks.detach().cpu().numpy()

    plt.figure(figsize=(10, 3))
    plt.plot(signal, lw=2)
    plt.scatter(peaks, signal[peaks], color="red", zorder=3, label="peaks")
    plt.title(title)
    plt.xlabel("frame")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import torch

    # -----------------------------
    # Create synthetic latent space
    # -----------------------------
    T = 200
    t = torch.linspace(0, 6 * torch.pi, T)

    z = torch.stack([
        torch.cos(t),
        torch.sin(t)
    ], dim=1)

    # add drift + noise (realistic)
    z = z + 0.05 * torch.randn_like(z)
    z = z + 0.002 * torch.arange(T).unsqueeze(1)

    z = torch.tensor(np.load('latent_space2.npy')).to('cuda')

    # -----------------------------
    # Phase-based method
    # -----------------------------
    out_phase = process_latent_phase(
        z,
        frames=None,
        initial_axes="y",
        rotation_period="2pi",
        basis_functions=15,
        smooth_window=11,
        smooth_poly=2,
    )

    plot_signal_with_peaks(
        out_phase["smoothed_distances"],
        out_phase["peaks"],
        title="Phase-based latent signal + peaks",
        ylabel="phase (unwrapped)",
    )

    print("Phase peaks:", out_phase["peaks"].cpu().numpy())
    print("Phase peak diffs:", torch.diff(out_phase["peaks"]).cpu().numpy())

    # -----------------------------
    # Distance-based (your original)
    # -----------------------------
    out_dist = process_latent_space(
        z,
        frames=torch.zeros(T, 2, 32, 32),  # dummy
        initial_axes="y",
        rotation_period="2pi",
    )

    plot_signal_with_peaks(
        out_dist["smoothed_distances"],
        out_dist["peaks"],
        title="Distance-based latent signal + peaks",
        ylabel="1 - normalized distance",
    )

    print("Distance peaks:", out_dist["peaks"].cpu().numpy())
    print("Distance peak diffs:", torch.diff(out_dist["peaks"]).cpu().numpy())