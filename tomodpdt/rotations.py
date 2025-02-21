import numpy as np

# Generate a sinusoidal quaternion
def generate_sinusoidal_quaternion(omega=2 * np.pi, phi=np.pi / 8, psi=np.pi / 6, duration=2, samples=200, *args, **kwargs):
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


def generate_random_sinusoidal_quaternion(omega=2 * np.pi, phi=np.pi / 8, psi=np.pi / 6, duration=2, samples=200, *args, **kwargs):
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

    # Shuffle the name of the components
    components = [q1, q2, q3]
    np.random.shuffle(components)
    q1, q2, q3 = components
    
    # Normalize quaternion to ensure it remains valid
    norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    q0, q1, q2, q3 = q0 / norm, q1 / norm, q2 / norm, q3 / norm
    
    # Stack components to create a quaternion array
    Q_accum = np.array([q0, q1, q2, q3]).T
    
    return Q_accum


def generate_noisy_sinusoidal_quaternion(omega=2 * np.pi, phi=np.pi / 8, psi=np.pi / 6, duration=2, samples=200, noise=0.025, *args, **kwargs):
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


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    Q = generate_random_sinusoidal_quaternion()
    plt.plot(Q)
    plt.legend(["q0", "q1", "q2", "q3"])
    plt.show()
    print(Q.shape)