import numpy as np
from scipy.linalg import svd
import cv2
import matplotlib.pyplot as plt


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
                                       duration=2, samples=200, noise_scale=0.025):
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

    phi = phi_base + np.random.uniform(0.025, 0.1) * np.sin(omega2 * t) + phi_variation  
    psi = psi_base + np.random.uniform(0.025, 0.1) * np.cos(omega2 * t) + psi_variation

    # Generate quaternion components
    q0 = np.cos(omega1 * t) * np.cos(0.5 * omega2 * t)  
    q1 = np.sin(omega1 * t) * np.cos(phi)  
    q2 = np.sin(omega1 * t) * np.sin(phi) * np.cos(psi)  + smooth_random_walk(samples, scale=noise_scale/8)
    q3 = np.sin(omega1 * t) * np.sin(phi) * np.sin(psi)  + smooth_random_walk(samples, scale=noise_scale/8)

    # Shuffle the name of the components so x, y, z are not always in the same order
    components = [q1, q2]
    np.random.shuffle(components)
    q1, q2 = components

    # Normalize quaternion smoothly
    norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2 + 1e-8)  
    q0, q1, q2, q3 = q0 / norm, q1 / norm, q2 / norm, q3 / norm

    return np.array([q0, q1, q2, q3]).T

def generate_integrated_angular_velocity_quaternion(
        duration=2.0, samples=200, noise_scale=0.35, base_speed=1.5):
    """
    Generate a realistic random smooth rotation by integrating
    a time-varying angular velocity vector.
    
    Returns:
        (samples, 4) array of unit quaternions.
    """
    dt = duration / samples
    t = np.linspace(0, duration, samples)

    # Smooth angular velocity components
    wx = base_speed * np.sin(0.5 * t) + smooth_random_walk(samples, noise_scale)
    wy = base_speed * np.cos(0.7 * t) + smooth_random_walk(samples, noise_scale)
    wz = base_speed * np.sin(1.3 * t + 0.4) + smooth_random_walk(samples, noise_scale)

    # Allocate quaternion array
    Q = np.zeros((samples, 4))
    q = np.array([1., 0., 0., 0.])  # Start at identity rotation

    for i in range(samples):
        w = np.array([0, wx[i], wy[i], wz[i]])

        # Quaternion derivative:  dq/dt = 0.5 * q ⊗ w
        dq = 0.5 * quaternion_multiply(q, w) * dt
        q = q + dq

        # Normalize to stay on SO(3)
        q = q / np.linalg.norm(q)

        Q[i] = q

    return Q


def quaternion_multiply(q1, q2):
    """Quaternion product q1 ⊗ q2 (w,x,y,z format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def generate_axis_switching_quaternion(
        duration=2.0, samples=200, switch_points=3, jitter=0.15):
    """
    Generate a quaternion where rotation happens around multiple
    main axes with smooth transitions between them.
    
    Returns:
        (samples, 4) array of quaternions.
    """

    t = np.linspace(0, duration, samples)

    # Predefined axes to morph between
    axes = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0]
    ])

    # Normalize
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)

    # Indices where we switch axes
    switch_idx = np.linspace(0, samples - 1, switch_points + 2).astype(int)

    q_list = []
    q = np.array([1., 0., 0., 0.])

    for k in range(len(switch_idx) - 1):
        a0 = axes[np.random.randint(len(axes))]
        a1 = axes[np.random.randint(len(axes))]

        # Smooth blend parameter
        blend = np.linspace(0, 1, switch_idx[k+1] - switch_idx[k])

        # Axis evolves from a0 → a1
        axis = (1-blend)[:, None] * a0 + blend[:, None] * a1
        axis = axis / np.linalg.norm(axis, axis=1, keepdims=True)

        # Angular velocity magnitude
        speed = 2*np.pi*(1 + jitter*np.random.randn())

        for i in range(len(blend)):
            angle = speed * (duration / samples)

            # Quaternion for rotation around axis
            sin_half = np.sin(angle/2)
            dq = np.array([
                np.cos(angle/2),
                axis[i,0] * sin_half,
                axis[i,1] * sin_half,
                axis[i,2] * sin_half
            ])

            q = quaternion_multiply(q, dq)
            q = q / np.linalg.norm(q)
            q_list.append(q.copy())

    return np.array(q_list)

def generate_ou_quaternion(
        duration=2.0,
        samples=200,
        tau=0.2,                # correlation time (s)
        sigma=1.1,              # noise strength
        base_speed=1.75,         # drift rotation speed
        seed=None):
    """
    Generate a quaternion trajectory using an Ornstein–Uhlenbeck (OU)
    process in angular velocity space:
        dω = -(1/τ) * ω dt + σ dW
    The quaternion is updated via exponential map:
        q <- q ⊗ exp(0.5 * ω dt)
    
    Returns:
        Q: (samples, 4) array of unit quaternions.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = duration / samples

    # Angular velocity vector ω = (ωx, ωy, ωz)
    omega = np.zeros(3)

    Q = np.zeros((samples, 4))
    q = np.array([1., 0., 0., 0.])  # identity quaternion

    for i in range(samples):
        # OU update: dω = -(ω/τ)*dt + σ*sqrt(dt)*N(0,1)
        dW = np.random.normal(0, np.sqrt(dt), size=3)
        domega = -(omega / tau) * dt + sigma * dW
        omega = omega + domega

        # Add a gentle mean rotation speed
        omega = omega + base_speed * 0.05

        # Convert angular velocity vector → quaternion increment
        angle = np.linalg.norm(omega) * dt
        axis = omega / (np.linalg.norm(omega) + 1e-8)

        dq = np.array([
            np.cos(angle/2),
            axis[0] * np.sin(angle/2),
            axis[1] * np.sin(angle/2),
            axis[2] * np.sin(angle/2)
        ])

        # Update rotation
        q = quaternion_multiply(q, dq)
        q = q / np.linalg.norm(q)

        Q[i] = q

    return Q

def normalize_quaternions_to_identity(quats):
    """
    Adjusts a sequence of quaternions so that the first quaternion becomes [1, 0, 0, 0]
    by left-multiplying all quaternions by the inverse of the first quaternion.
    
    Parameters
    ----------
    quats : np.ndarray
        Array of shape (N, 4) containing quaternions [w, x, y, z].
    
    Returns
    -------
    np.ndarray
        Corrected quaternions of shape (N, 4).
    """
    q0 = quats[0]
    q0_inv = np.array([q0[0], -q0[1], -q0[2], -q0[3]])

    quats_corrected = []
    for q in quats:
        w = q0_inv[0]*q[0] - q0_inv[1]*q[1] - q0_inv[2]*q[2] - q0_inv[3]*q[3]
        x = q0_inv[0]*q[1] + q0_inv[1]*q[0] + q0_inv[2]*q[3] - q0_inv[3]*q[2]
        y = q0_inv[0]*q[2] - q0_inv[1]*q[3] + q0_inv[2]*q[0] + q0_inv[3]*q[1]
        z = q0_inv[0]*q[3] + q0_inv[1]*q[2] - q0_inv[2]*q[1] + q0_inv[3]*q[0]
        quats_corrected.append([w, x, y, z])

    return np.array(quats_corrected)

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

def compute_angular_velocity(Q, dt=1.0):
    """Compute quaternion angular velocity magnitude."""
    dQ = np.diff(Q, axis=0) / dt
    return np.linalg.norm(dQ, axis=1)

def quaternion_axis(Q):
    """Return normalized imaginary parts projected to sphere."""
    axis = Q[:,1:4]
    norm = np.linalg.norm(axis, axis=1, keepdims=True) + 1e-8
    return axis / norm

def quaternion_norm_error(Q):
    return np.abs(np.linalg.norm(Q, axis=1) - 1)

def plot_quaternion(ax, Q, title, show_ylabel=False, show_xlabel=False, grid=False):
    frames = np.arange(len(Q))

    ax.plot(frames, Q[:,0], color="black", label="q0", linewidth=2)
    ax.plot(frames, Q[:,1], color="royalblue", label="q1", linewidth=2)
    ax.plot(frames, Q[:,2], color="crimson", label="q2", linewidth=2)
    ax.plot(frames, Q[:,3], color="darkorange", label="q3", linewidth=2)

    ax.set_ylim([-1.05, 1.05])
    ax.set_title(title, fontsize=12)

    if show_ylabel:
        ax.set_ylabel("Quaternion components", fontsize=10)
    if show_xlabel:
        ax.set_xlabel("Frame", fontsize=10)

    # Set yticks to -1, -0.5, 0, 0.5, 1
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    if grid:
        ax.grid(True, linestyle=":", color="0.85")

    ax.legend(frameon=False, fontsize=10)

if __name__ == "__main__":

    samples = 200
    duration = 2

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

    # Physically realistic random rotation
    Q = generate_integrated_angular_velocity_quaternion(duration=duration, samples=samples)
    plt.plot(Q) 
    plt.title("Integrated Angular Velocity Quaternion")
    plt.legend(["q0","q1","q2","q3"]) 
    plt.show()

    # Axis switching rotation
    Q = generate_axis_switching_quaternion(duration=duration, samples=samples)
    plt.plot(Q)
    plt.title("Axis Switching Quaternion")
    plt.legend(["q0","q1","q2","q3"])
    plt.show()

    Q = generate_ou_quaternion(duration=duration, samples=samples, tau=0.4, sigma=1.5)
    plt.plot(Q)
    plt.title("Quaternion Ornstein–Uhlenbeck Process")
    plt.legend(["q0", "q1", "q2", "q3"])
    plt.show()



    # -------------------------------------------------------
    # Generate all six quaternion trajectories
    # -------------------------------------------------------
    samples = 200
    duration = 2

    Q1 = generate_random_sinusoidal_quaternion(duration=duration, samples=samples)
    Q2 = generate_smooth_varying_quaternion(duration=duration, samples=samples)
    Q3 = generate_random_varying_quaternion(duration=duration, samples=samples)
    Q4 = generate_integrated_angular_velocity_quaternion(duration=duration, samples=samples)
    Q5 = generate_axis_switching_quaternion(duration=duration, samples=samples)
    Q6 = generate_ou_quaternion(duration=duration, samples=samples, tau=0.4, sigma=1.5)

    # -------------------------------------------------------
    # Build a clean 2×3 figure
    # -------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    panel_labels = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"]

    for ax, label in zip(axes.flatten(), panel_labels):
        ax.text(
            -0.1, 1.075, label,
            transform=ax.transAxes,
            fontsize=13, fontweight="bold",
            va="top", ha="left"
        )


    plot_quaternion(axes[0,0], Q1, "Random Sinusoidal Quaternion", show_ylabel=True)
    plot_quaternion(axes[0,1], Q2, "Smooth Varying Quaternion")
    plot_quaternion(axes[0,2], Q3, "Random Varying Quaternion")

    plot_quaternion(axes[1,0], Q4, "Integrated Angular Velocity Quaternion", show_ylabel=True, show_xlabel=True)
    plot_quaternion(axes[1,1], Q5, "Axis Switching Quaternion", show_xlabel=True)
    plot_quaternion(axes[1,2], Q6, "Quaternion Ornstein–Uhlenbeck Process", show_xlabel=True)

    plt.tight_layout()

    # Show figure
    plt.show()

    # -------------------------------------------------------
    # Save as high-quality PDF and SVG
    # -------------------------------------------------------
    #fig.savefig("quaternion_rotation_models.pdf", dpi=300, bbox_inches="tight")
    #fig.savefig("quaternion_rotation_models.svg", dpi=300, bbox_inches="tight")


    # Do one nice plot of generate_random_sinusoidal_quaternion
    Q = generate_random_sinusoidal_quaternion(duration=duration, samples=samples)
    fig, ax = plt.subplots(figsize=(6, 3))
    plot_quaternion(ax, Q, "Random Sinusoidal Quaternion", show_ylabel=True, show_xlabel=True)
    plt.tight_layout()
    plt.savefig("random_sinusoidal_quaternion.png", dpi=300, bbox_inches="tight")
    plt.show()