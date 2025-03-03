import torch
import numpy as np
import scipy.ndimage

import matplotlib.pyplot as plt

### Setttings ###
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set the device
SIZE = 64  # Size of the 3D object
RI_RANGE = (1.33, 1.42)

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


### VOLUME 1 ###
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
VOL_GAUSS = grid * (RI_RANGE[1] - RI_RANGE[0]) + RI_RANGE[0]
#################

### VOLUME 2 ###
# Parameters
num_layers = 5  # Number of layers inside the sphere

# Specify custom densities for each layer
layer_densities = [1, 2, 3, 4, 5]  # You can modify this list to set custom values for each layer

# Generate the volume
volume = generate_3d_volume(SIZE, num_layers, layer_densities)

# Normalize the volume for visualization
VOL_SHELL = (volume - volume.min()) / (volume.max() - volume.min()) * (RI_RANGE[1] - RI_RANGE[0]) + RI_RANGE[0]
##################

### VOLUME 3 ###
grid = np.zeros((SIZE, SIZE, SIZE))

# Random points
num_points = 10
thickness = 1
points = np.random.randint(SIZE//8, SIZE-SIZE//8, (num_points, 3))

# Set the random points to 1
for point in points:
    grid[
        point[0]-thickness:point[0]+thickness, 
        point[1]-thickness:point[1]+thickness, 
        point[2]-thickness:point[2]+thickness
        ] = 1

VOL_FLUO = grid
##################

### VOLUME 4 ###
grid = np.zeros((SIZE, SIZE, SIZE))

# Random points
num_points = 12

# add small gaussian blobs at random positions
for i in range(10):
    x, y, z = np.random.randint(SIZE//8, SIZE-SIZE//8, 3)
    blob = np.exp(-((x - np.arange(SIZE))**2 + (y - np.arange(SIZE)[:, None])**2 + (z - np.arange(SIZE)[:, None, None])**2) / (2 * 2**2))
    grid += blob

grid /= grid.max()

VOL_GAUSS_MULT = grid * (RI_RANGE[1] - RI_RANGE[0]) + RI_RANGE[0]

##################


if __name__== "__main__":

    ### VOLUME 1 ###
    fig, ax = plt.subplots(1, 3, figsize=(8, 3))
    fig.suptitle('3D Gaussian Blob')
    for j in range(3):
        ax[j].imshow(VOL_GAUSS.sum(axis=j))
        ax[j].set_title(f'Axis {j}')
    plt.show()

    ### VOLUME 2 ###
    fig, ax = plt.subplots(1, 3, figsize=(8, 3))
    fig.suptitle('3D SPHERE SHELL')
    for j in range(3):
        ax[j].imshow(VOL_SHELL.sum(axis=j))
        ax[j].set_title(f'Axis {j}')
    plt.show()

    ### VOLUME 3 ###
    fig, ax = plt.subplots(1, 3, figsize=(8, 3))
    fig.suptitle('3D FLUORESCENT POINTS')
    for j in range(3):
        ax[j].imshow(VOL_FLUO.sum(axis=j))
        ax[j].set_title(f'Axis {j}')
    plt.show()

    ### VOLUME 4 ###
    fig, ax = plt.subplots(1, 3, figsize=(8, 3))
    fig.suptitle('3D Gaussian Blob')
    for j in range(3):
        ax[j].imshow(VOL_GAUSS_MULT.sum(axis=j))
        ax[j].set_title(f'Axis {j}')
    plt.show()
