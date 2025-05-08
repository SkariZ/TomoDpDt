import numpy as np
from scipy.linalg import svd
import cv2

# Generate a sinusoidal quaternion
def generate_sinusoidal_quaternion(omega=2 * np.pi, phi=np.pi / 8,
                                   psi=np.pi / 6, duration=2,
                                   samples=200):
    """
    Generate a sinusoidal quaternion over time.
    
    Parameters:
        omega (float): Angular frequency (e.g., 2π for 1 Hz).
        phi (float): Phase offset for q1, q2, q3.
        psi (float): Additional phase relationship for q2, q3.
        duration (float): Total simulation duration in seconds.
        samples (int): Number of time samples.
    
    Returns:
        np.ndarray: Array of shape (samples, 4), where each row is [q0, q1, q2, q3].
    """
    # Time array
    t = np.linspace(0, duration, samples)
    
    # Components of the quaternion
    q0 = np.cos(omega * t)
    q1 = np.sin(omega * t) * np.cos(phi)
    q2 = np.sin(omega * t) * np.sin(phi) * np.cos(psi)
    q3 = np.sin(omega * t) * np.sin(phi) * np.sin(psi)
    
    # Normalize quaternion to ensure it remains valid
    norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    q0, q1, q2, q3 = q0 / norm, q1 / norm, q2 / norm, q3 / norm
    
    # Stack components to create a quaternion array
    Q_accum = np.array([q0, q1, q2, q3]).T
    
    return Q_accum


def generate_random_sinusoidal_quaternion(omega=2 * np.pi, phi=np.pi / 8,
                                          psi=np.pi / 6, duration=2,
                                          samples=200):
    """
    Generate a sinusoidal quaternion over time.
    
    Parameters:
        omega (float): Angular frequency (e.g., 2π for 1 Hz).
        phi (float): Phase offset for q1, q2, q3.
        psi (float): Additional phase relationship for q2, q3.
        duration (float): Total simulation duration in seconds.
        samples (int): Number of time samples.
    
    Returns:
        np.ndarray: Array of shape (samples, 4), where each row is [q0, q1, q2, q3].
    """
    # Time array
    t = np.linspace(0, duration, samples)
    
    # Components of the quaternion
    q0 = np.cos(omega * t)
    
    q1 = np.sin(omega * t) * np.cos(phi)
    q2 = np.sin(omega * t) * np.sin(phi) * np.cos(psi)
    q3 = np.sin(omega * t) * np.sin(phi) * np.sin(psi)

    # Shuffle the name of the components so x, y, z are not always in the same order
    components = [q1, q2]
    np.random.shuffle(components)
    q1, q2 = components
    
    # Normalize quaternion to ensure it remains valid
    norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    q0, q1, q2, q3 = q0 / norm, q1 / norm, q2 / norm, q3 / norm
    
    # Stack components to create a quaternion array
    Q_accum = np.array([q0, q1, q2, q3]).T
    
    return Q_accum


def generate_noisy_sinusoidal_quaternion(omega=2 * np.pi, phi=np.pi / 8,
                                         psi=np.pi / 6, duration=2,
                                         samples=200, noise=0.025):
    """
    Generate a noisy sinusoidal quaternion over time.
    
    Parameters:
        omega (float): Angular frequency (e.g., 2π for 1 Hz).
        phi (float): Phase offset for q1, q2, q3.
        psi (float): Additional phase relationship for q2, q3.
        duration (float): Total simulation duration in seconds.
        samples (int): Number of time samples.
        noise (float): Standard deviation of the noise.
    
    Returns:
        np.ndarray: Array of shape (samples, 4), where each row is [q0, q1, q2, q3].
    """
    # Generate a clean quaternion
    Q_accum = generate_sinusoidal_quaternion(omega, phi, psi, duration, samples)
    
    # Add noise to the quaternion
    noise = np.random.normal(0, noise, size=(samples, 4))
    Q_noisy = Q_accum + noise
    
    # Normalize quaternion to ensure it remains valid
    norm = np.sqrt(np.sum(Q_noisy**2, axis=1))
    Q_noisy = Q_noisy / norm[:, None]

    return Q_noisy


def generate_smooth_varying_quaternion(omega1=2 * np.pi, omega2=np.pi / 3, 
                                       phi_base=np.pi / 8, psi_base=np.pi / 6, 
                                       duration=2, samples=200):
    """
    Generate a smoothly varying quaternion with multiple frequency components.
    
    Parameters:
        omega1 (float): Primary angular frequency.
        omega2 (float): Secondary angular frequency for smooth variation.
        phi_base (float): Base phase shift.
        psi_base (float): Base phase shift for q2, q3.
        duration (float): Duration in seconds.
        samples (int): Number of samples.
    
    Returns:
        np.ndarray: (samples, 4) array of smoothly varying quaternions.
    """
    t = np.linspace(0, duration, samples)

    # Slowly modulate the rotation axis over time
    phi = phi_base + np.random.uniform(0.1, 0.3) * np.sin(omega2 * t)
    psi = psi_base + np.random.uniform(0.05, 0.2) * np.cos(omega2 * t)

    # Primary rotation component
    q0 = np.cos(omega1 * t) * np.cos(0.5 * omega2 * t)  
    q1 = np.sin(omega1 * t) * np.cos(phi)  
    q2 = np.sin(omega1 * t) * np.sin(phi) * np.cos(psi)  
    q3 = np.sin(omega1 * t) * np.sin(phi) * np.sin(psi)  

    # Normalize quaternion smoothly
    norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2 + 1e-8)  # Avoid division issues
    q0, q1, q2, q3 = q0 / norm, q1 / norm, q2 / norm, q3 / norm

    return np.array([q0, q1, q2, q3]).T


def smooth_random_walk(n, scale=0.025):
    """Generate a smooth random walk with small changes over time."""
    steps = np.random.normal(loc=0.0, scale=scale, size=n)
    return np.cumsum(steps)  # Cumulative sum to make it smooth


def generate_random_varying_quaternion(omega1=2 * np.pi, omega2=np.pi / 3, 
                                       phi_base=np.pi / 8, psi_base=np.pi / 6, 
                                       duration=2, samples=200, noise_scale=0.02):
    """
    Generate a quaternion with smooth random variations over time.
    
    Parameters:
        omega1 (float): Base angular frequency.
        omega2 (float): Secondary frequency for modulation.
        phi_base (float): Base phase shift.
        psi_base (float): Base phase shift for q2, q3.
        duration (float): Duration in seconds.
        samples (int): Number of samples.
        noise_scale (float): Strength of random variations.
    
    Returns:
        np.ndarray: (samples, 4) array of smoothly varying quaternions.
    """
    t = np.linspace(0, duration, samples)

    # Generate smooth randomness for phase shifts
    phi_variation = smooth_random_walk(samples, scale=noise_scale)
    psi_variation = smooth_random_walk(samples, scale=noise_scale)

    phi = phi_base + np.random.uniform(0.1, 0.3) * np.sin(omega2 * t) + phi_variation  
    psi = psi_base + np.random.uniform(0.05, 0.2) * np.cos(omega2 * t) + psi_variation

    # Generate quaternion components
    q0 = np.cos(omega1 * t) * np.cos(0.5 * omega2 * t)  
    q1 = np.sin(omega1 * t) * np.cos(phi)  
    q2 = np.sin(omega1 * t) * np.sin(phi) * np.cos(psi)  
    q3 = np.sin(omega1 * t) * np.sin(phi) * np.sin(psi)  

    # Shuffle the name of the components so x, y, z are not always in the same order
    components = [q1, q2]
    np.random.shuffle(components)
    q1, q2 = components

    # Normalize quaternion smoothly
    norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2 + 1e-8)  
    q0, q1, q2, q3 = q0 / norm, q1 / norm, q2 / norm, q3 / norm

    return np.array([q0, q1, q2, q3]).T

def compute_optical_flow(frames):
    """
    Computes dense optical flow between consecutive frames.
    :param frames: NumPy array of shape (T, 64, 64) with values in range [0,1].
    :return: Motion vectors (dx, dy) for sampled points.
    """

    import cv2

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

def align_frames_cv2(frames):
    # Ensure frames are in the correct format
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)  # Add batch dimension

    frames = frames[:, 0].cpu().numpy()

    # Compute dense optical flow between consecutive frames
    flow_vectors = compute_optical_flow(frames)

    discplacement = flow_vectors[:, 2:4]  # Extract dx, dy components
    _, _, Vt = svd(discplacement, full_matrices=False)
    principal_axis = Vt[0]  # First principal component
    principal_axis = principal_axis / np.linalg.norm(principal_axis)  # Normalize

    # Compute rotation angle and align so that the first component is along the x-axis
    if principal_axis[0] > 0:
        rotation_angle = np.arctan2(principal_axis[1], principal_axis[0])
    else:
        rotation_angle = np.arctan2(-principal_axis[1], -principal_axis[0])

    # Convert rotation angle from radians to degrees
    rotation_angle = np.degrees(rotation_angle)

    # Align the images to have no tilt
    tilt_frames = np.zeros_like(frames)
    for i in range(len(frames)):
        tilt_frames[i] = cv2.warpAffine(
            frames[i], 
            cv2.getRotationMatrix2D((frames[i].shape[1] // 2, frames[i].shape[0] // 2), rotation_angle, 1),
            (frames[i].shape[1], frames[i].shape[0])
            )
    
    return tilt_frames

if __name__ == "__main__":
    import torch
    import torch.nn as nn

    projections = np.load('C:/Users/Fredrik/Desktop/RotationsData/Movies_cropped_np/movie11.npy')
    projections = projections[::8]
    projections = torch.tensor(projections, dtype=torch.float32).unsqueeze(1)

    #Downsample 2x
    projections = nn.functional.avg_pool2d(projections, 2)

    projs = align_frames_cv2(projections)

    samples = 400
    duration = 2

    import matplotlib.pyplot as plt
    Q = generate_random_sinusoidal_quaternion(duration=duration, samples=samples)
    plt.plot(Q)
    plt.title("Random Sinusoidal Quaternion")
    plt.legend(["q0", "q1", "q2", "q3"])
    plt.show()
    
    Q = generate_noisy_sinusoidal_quaternion(duration=duration, samples=samples)
    plt.plot(Q)
    plt.title("Noisy Sinusoidal Quaternion")
    plt.legend(["q0", "q1", "q2", "q3"])
    plt.show()
    
    Q = generate_smooth_varying_quaternion(duration=duration, samples=samples)
    plt.plot(Q)
    plt.title("Smooth Varying Quaternion")
    plt.legend(["q0", "q1", "q2", "q3"])
    plt.show()

    Q = generate_random_varying_quaternion(duration=duration, samples=samples)
    plt.plot(Q)
    plt.title("Random Varying Quaternion")
    plt.legend(["q0", "q1", "q2", "q3"])
    plt.show()