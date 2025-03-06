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

import volumes as V

VOL_GAUSS = V.VOL_GAUSS
VOL_FLUO = V.VOL_FLUO
VOL_GAUSS_MULT = V.VOL_GAUSS_MULT
VOL_SHELL = V.VOL_SHELL
VOL_RANDOM = V.VOL_RANDOM

# Settings
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set the device
SIZE = 64  # Size of the 3D object


def create_data(volume_case='gaussian_multiple', image_modality='sum_projection', samples=400, duration=2, rotation_case='sinusoidal'):

    image_modality = image_modality.lower()

    if image_modality == 'fluorescence':
        volume_case = 'fluorescence'

    # Create a 3D object
    if volume_case == 'gaussian':
        object = torch.tensor(VOL_GAUSS, dtype=torch.float32, device=DEV)
    elif volume_case == 'fluorescence':
        object = torch.tensor(VOL_FLUO, dtype=torch.float32, device=DEV)
    elif volume_case == 'gaussian_multiple':
        object = torch.tensor(VOL_GAUSS_MULT, dtype=torch.float32, device=DEV)
    elif volume_case == 'shell':
        object = torch.tensor(VOL_SHELL, dtype=torch.float32, device=DEV)
    elif volume_case == 'random':
        object = torch.tensor(VOL_RANDOM, dtype=torch.float32, device=DEV)
    else:
        raise ValueError('Unknown volume case')

    # Create quaternions
    if rotation_case == 'noisy_sinusoidal':
        quaternions = R.generate_noisy_sinusoidal_quaternion(duration=duration, samples=samples, noise=0.001)
    elif rotation_case == 'sinusoidal':
        quaternions = R.generate_sinusoidal_quaternion(duration=duration, samples=samples)
    elif rotation_case == 'random_sinusoidal':
        quaternions = R.generate_random_sinusoidal_quaternion(duration=duration, samples=samples)
    elif rotation_case == '1ax':
        quaternions = R.generate_random_sinusoidal_quaternion(duration=duration, samples=samples, phi=0, psi=0, noise=0.001)
    elif rotation_case == 'smooth_varying':
        quaternions = R.generate_smooth_varying_quaternion(duration=duration, samples=samples)
    elif rotation_case == 'smooth_varying_random':
        quaternions = R.generate_smooth_varying_quaternion(duration=duration, samples=samples)
    else:
        raise ValueError('Unknown rotation case')

    quaternions = torch.tensor(quaternions, dtype=torch.float32, device=DEV)

    ch = 1
    # Create an imaging modality
    if image_modality == 'sum_projection':
        imaging_model = IMT.Sum3d2d(dim=-1)
    
    elif image_modality.lower() == 'sum_projection_avg_weighted':
        imaging_model = IMT.SumAvgWeighted3d2d(dim=-1)

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
    projections = torch.zeros((samples, ch, object.shape[1], object.shape[2]))
    quaternions = torch.tensor(quaternions, dtype=torch.float32).to('cuda')

    # Generate the dataset
    for i in range(samples):
        # Progress in percentage
        if i % 100 == 0: 
            print(f'{i/samples * 100:.1f}%')

        volume_new = rotmod.apply_rotation(
            volume=object, 
            q=quaternions[i]
            )       
        
        # Compute the image
        image = imaging_model(volume_new)

        if image_modality == 'sum_projection' or image_modality == 'sum_projection_avg_weighted':
            projections[i, 0] = image.cpu().squeeze()

        if image_modality.lower() in ['brightfield']:
            projections[i, 0] = image.real.cpu().squeeze()
            projections[i, 1] = image.imag.cpu().squeeze()

        if image_modality.lower() in ['darkfield', 'iscat', 'fluorescence']:
            projections[i, 0] = image.cpu().squeeze().real
    
    return object, quaternions, projections, imaging_model


if __name__=='__main__':

    object, quaternions, projections, imaging_model = create_data(
        image_modality='darkfield', 
        rotation_case='random_sinusoidal', 
        samples=10
        )

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