import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

import deeptrack as dt

import torch
import torch.nn as nn

# Import modules from the tomodpdt package
import rotations as R
import forward_module as FM
import imaging_modality_torch as IMT

# Set the random seed for reproducibility
np.random.seed(123)
torch.manual_seed(123)


def generate_3d_volume(size, num_layers, layer_densities):
    # Ensure that the number of densities matches the number of layers
    if len(layer_densities) != num_layers:
        raise ValueError("The number of densities must match the number of layers.")
    
    # Create an empty volume
    volume = np.zeros((size, size, size))

    # Define center and radius
    center = size // 2
    radius = size // 3

    # Generate grid
    x, y, z = np.meshgrid(np.arange(size), np.arange(size), np.arange(size))
    distance = np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)

    # Create layers with custom densities
    layer_radius = np.linspace(0, radius, num_layers + 1)

    for i in range(num_layers):
        mask = (distance < layer_radius[i+1]) & (distance >= layer_radius[i])
        volume[mask] = layer_densities[i]

    # Add noise that scales with the density of each layer
    noise = np.zeros_like(volume)
    for i in range(num_layers):
        mask = (distance < layer_radius[i+1]) & (distance >= layer_radius[i])
        layer_noise = np.random.normal(0, 0.2 * layer_densities[i], volume.shape)  # Scale noise by density
        noise[mask] = layer_noise[mask]
    
    volume += noise
    volume = np.clip(volume, 0, np.max(layer_densities))  # Keep values within the range of layer densities

    # Apply slight smoothing to the structure
    volume = scipy.ndimage.gaussian_filter(volume, sigma=1)

    return volume

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set the device
SIZE = 64  # Size of the 3D object
RI_RANGE = (1.33, 1.42)
VOLUME_CASE = 1


if VOLUME_CASE == 1:
    grid = np.zeros((SIZE, SIZE, SIZE))  # 3D grid

    # Gaussian blob parameters
    centers = [(16, 32, 32), (48, 32, 32), (32, 16, 32), 
            (32, 48, 32), (32, 32, 16), (32, 32, 48),
            ]  # Center positions

    # Generate blobs
    sigma_random = np.random.uniform(2, 6, len(centers))
    for i, center in enumerate(centers):
        x, y, z = np.indices((SIZE, SIZE, SIZE))  # 3D coordinates
        blob = np.exp(-((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) / (2 * sigma_random[i]**2))
        grid += blob  # Add each blob to the grid

    # Normalize the grid for visualization
    grid /= grid.max()

    # Scale to RI range
    VOL = grid * (RI_RANGE[1] - RI_RANGE[0]) + RI_RANGE[0]

elif VOLUME_CASE == 2:
    # Parameters
    size = 64  # Volume size (64x64x64)
    num_layers = 5  # Number of layers inside the sphere

    # Specify custom densities for each layer
    layer_densities = [1, 2, 3, 4, 5]  # You can modify this list to set custom values for each layer

    # Generate the volume
    volume = generate_3d_volume(size, num_layers, layer_densities)

    # Normalize the volume for visualization
    VOL = (volume - volume.min()) / (volume.max() - volume.min()) * (RI_RANGE[1] - RI_RANGE[0]) + RI_RANGE[0]


def create_data(volume=VOL, image_modality='sum_projection', samples=400, rotation_case='sinusoidal'):

    # Create a 3D object
    object = torch.tensor(volume, dtype=torch.float32, device=DEV)

    # Create a quaternion
    if rotation_case == 'noisy_sinusoidal':
        quaternions = R.generate_noisy_sinusoidal_quaternion(duration=2, samples=samples, noise=0.001)
    elif rotation_case == 'sinusoidal':
        quaternions = R.generate_sinusoidal_quaternion(duration=2, samples=samples)
    elif rotation_case == 'random_sinusoidal':
        quaternions = R.generate_random_sinusoidal_quaternion(duration=2, samples=samples, noise=0.001)
    elif rotation_case == '1ax':
        quaternions = R.generate_random_sinusoidal_quaternion(duration=2, samples=samples, phi=0, psi=0, noise=0.001)
        
    quaternions = torch.tensor(quaternions, dtype=torch.float32, device=DEV)

    ch = 1
    # Create an imaging modality
    if image_modality == 'sum_projection':
        imaging_model = IMT.Dummy3d2d(dim=-1)

    elif image_modality.lower() == 'brightfield':
        optics = IMT.setup_optics(SIZE, microscopy_regime='Brightfield')
        imaging_model = IMT.imaging_model(optics)
        ch = 2

    elif image_modality.lower() == 'darkfield':
        optics = IMT.setup_optics(SIZE, microscopy_regime='Darkfield')
        imaging_model = IMT.imaging_model(optics)

    elif image_modality.lower() == 'iscat':
        optics = IMT.setup_optics(SIZE, microscopy_regime='Iscat')
        imaging_model = IMT.imaging_model(optics)

    elif image_modality.lower() == 'fluorescence':
        optics = IMT.setup_optics(SIZE, microscopy_regime='Fluorescence')
        imaging_model = IMT.imaging_model(optics)
    else:
        raise ValueError('Unknown imaging modality')
    
    # Create a rotation model
    rotmod = FM.ForwardModelSimple(N=SIZE)

    # Dataset
    projections = torch.zeros((samples, ch, volume.shape[1], volume.shape[2]))

    # Generate the dataset
    for i in range(samples):
        # Progress in percentage
        if i % 100 == 0: 
            print(f'{i/samples * 100:.1f}%')

        volume_new = rotmod.apply_rotation(
            volume=torch.tensor(volume, dtype=torch.float32).to('cuda'), 
            q=torch.tensor(quaternions[i], dtype=torch.float32).to('cuda')
            )
        
        # Compute the image
        image = imaging_model(volume_new)

        if image_modality == 'sum_projection':
            projections[i, 0] = image.cpu().squeeze()

        if image_modality.lower() in ['brightfield']:
            projections[i, 0] = image.real.cpu().squeeze()
            projections[i, 1] = image.imag.cpu().squeeze()

        if image_modality.lower() in ['darkfield', 'iscat', 'fluorescence']:
            projections[i, 0] = image.cpu().squeeze().real
    
    return object, quaternions, projections, imaging_model


if __name__=='__main__':

    object, quaternions, projections, imaging_model = create_data(image_modality='brightfield', rotation_case='random_sinusoidal', samples=25)

    # Plot the object
    plt.imshow(object.cpu().squeeze().numpy().sum(2))
    plt.colorbar()
    plt.title('Object')
    plt.show()

    # Plot the projections
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        for j in range(3):
            ax[i, j].imshow(projections[i * 3 + j, 0].cpu().numpy())
            ax[i, j].set_title(f'Projection {i * 3 + j}')
    plt.show()
    
    print('Object shape:', object.shape)
    print('Quaternions shape:', quaternions.shape)
    print('Projections shape:', projections.shape)
    
    plt.plot(quaternions.cpu().numpy())
    plt.legend(['q0', 'q1', 'q2', 'q3'])
    plt.title('Quaternions')
    plt.show()