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

def sample_positions_3D(num_points, area_size, min_distance, edge_margin=8):
    """
    Generate random 3D positions while ensuring a minimum distance between them 
    and keeping them away from edges.

    Parameters:
    - num_points (int): Number of points to generate.
    - area_size (tuple): (width, height, depth) of the 3D space.
    - min_distance (float): Minimum Euclidean distance between points.
    - edge_margin (float): Minimum distance from edges.

    Returns:
    - np.array: Array of shape (num_points, 3) with sampled (x, y, z) positions.
    """
    positions = []

    # Define valid sampling range (avoiding edges)
    min_bounds = np.array([edge_margin, edge_margin, edge_margin])
    max_bounds = np.array(area_size) - edge_margin

    while len(positions) < num_points:
        # Generate a random (x, y, z) point within the valid range
        candidate = np.random.uniform(min_bounds, max_bounds)

        # Check if it's at least min_distance away from all existing points
        if all(np.linalg.norm(candidate - np.array(p)) >= min_distance for p in positions):
            positions.append(candidate)

    return np.array(positions).astype(int)


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
points = sample_positions_3D(num_points, (SIZE, SIZE, SIZE), min_distance=SIZE//8)

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

# Generate random positions with a minimum distance between them
positions = sample_positions_3D(num_points, (SIZE, SIZE, SIZE), min_distance=SIZE//8)

# add small gaussian blobs at random positions
for i in range(10):
    x, y, z = positions[i]
    blob = np.exp(-((x - np.arange(SIZE))**2 + (y - np.arange(SIZE)[:, None])**2 + (z - np.arange(SIZE)[:, None, None])**2) / (2 * 2**2))
    grid += blob

grid /= grid.max()

VOL_GAUSS_MULT = grid * (RI_RANGE[1] - RI_RANGE[0]) + RI_RANGE[0]

##################

### VOLUME 5 ###

grid = np.zeros((SIZE, SIZE, SIZE))

# Random shapes at random positions

# 6 random positions that are not too close to the edges and not too close to each other
positions = sample_positions_3D(6, (SIZE, SIZE, SIZE), min_distance=SIZE//4)

count = 0
# 3 random squares of random sizes
for i in range(3):
    x, y, z = positions[count]
    size = np.random.randint(SIZE//16, SIZE//6)
    grid[x:x+size, y:y+size, z:z+size] = 0.5
    count += 1

# 3 random circles of random sizes
for i in range(3):
    x, y, z = positions[count]
    radius = np.random.randint(SIZE//16, SIZE//8)
    blob = np.exp(-((x - np.arange(SIZE))**2 + (y - np.arange(SIZE)[:, None])**2 + (z - np.arange(SIZE)[:, None, None])**2) / (2 * radius**2))
    grid += blob
    count += 1

grid /= grid.max()

VOL_RANDOM = grid * (RI_RANGE[1] - RI_RANGE[0]) + RI_RANGE[0]



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
    fig.suptitle('3D Gaussian Blobs Small')
    for j in range(3):
        ax[j].imshow(VOL_GAUSS_MULT.sum(axis=j))
        ax[j].set_title(f'Axis {j}')
    plt.show()

    ### VOLUME 5 ###
    fig, ax = plt.subplots(1, 3, figsize=(8, 3))
    fig.suptitle('3D Random Shapes')
    for j in range(3):
        ax[j].imshow(VOL_RANDOM.sum(axis=j))
        ax[j].set_title(f'Axis {j}')
    plt.show()

    # Save the volumes as numpy files in ../test_data/
    np.save('../test_data/vol_gauss.npy', VOL_GAUSS)
    np.save('../test_data/vol_shell.npy', VOL_SHELL)
    np.save('../test_data/vol_fluo.npy', VOL_FLUO)
    np.save('../test_data/vol_gauss_mult.npy', VOL_GAUSS_MULT)
    np.save('../test_data/vol_random.npy', VOL_RANDOM)