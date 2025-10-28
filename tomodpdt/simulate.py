import numpy as np
import matplotlib.pyplot as plt
import torch

# Import modules from the tomodpdt package
try:
    import tomodpdt.rotations as R
    import tomodpdt.forward_module as FM
    import tomodpdt.imaging_modality_torch as IMT
    import tomodpdt.volumes as V
    import tomodpdt.application as A
except:
    import rotations as R
    import forward_module as FM
    import imaging_modality_torch as IMT
    import volumes as V
    import application as A

# Set the random seed for reproducibility
# np.random.seed(123)
# torch.manual_seed(123)

VOL_GAUSS = V.VOL_GAUSS
VOL_FLUO = V.VOL_FLUO
VOL_GAUSS_MULT = V.VOL_GAUSS_MULT
VOL_SHELL = V.VOL_SHELL
VOL_RANDOM = V.VOL_RANDOM

# Settings
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set the device


def create_data(
        volume=None, 
        volume_case='gaussian_multiple', 
        image_modality='sum_projection', 
        samples=400, 
        duration=2, 
        rotation_case='sinusoidal',
        translations=None,
        ):
    """
    Generate a dataset of 3D objects and their 2D projections.
    Handles non-cubic volumes with shape (nx, ny, nz).
    """

    image_modality = image_modality.lower() if isinstance(image_modality, str) else image_modality

    if image_modality == 'fluorescence':
        volume_case = 'fluorescence'

    # Load 3D object
    if volume is not None:
        object = torch.tensor(volume, dtype=torch.float32, device=DEV)
    else:
        volume_dict = {
            'gaussian': VOL_GAUSS,
            'fluorescence': VOL_FLUO,
            'gaussian_multiple': VOL_GAUSS_MULT,
            'shell': VOL_SHELL,
            'random': VOL_RANDOM
        }
        if volume_case not in volume_dict:
            raise ValueError(f'Unknown volume case: {volume_case}')
        object = torch.tensor(volume_dict[volume_case], dtype=torch.float32, device=DEV)

    nx, ny, nz = object.shape  # Use actual dimensions
    
    # Create quaternions
    if isinstance(rotation_case, (np.ndarray, torch.Tensor)):
        quaternions = rotation_case
    else:
        rotation_fn_dict = {
            'noisy_sinusoidal': R.generate_noisy_sinusoidal_quaternion,
            'sinusoidal': R.generate_sinusoidal_quaternion,
            'random_sinusoidal': R.generate_random_sinusoidal_quaternion,
            '1ax': R.generate_random_sinusoidal_quaternion,
            'smooth_varying': R.generate_smooth_varying_quaternion,
            'smooth_varying_random': R.generate_smooth_varying_quaternion
        }
        if rotation_case not in rotation_fn_dict:
            raise ValueError(f'Unknown rotation case: {rotation_case}')
        quaternions = rotation_fn_dict[rotation_case](duration=duration, samples=samples) if rotation_case != '1ax' else R.generate_random_sinusoidal_quaternion(duration=duration, samples=samples, phi=0, psi=0)

    quaternions = torch.tensor(quaternions, dtype=torch.float32, device=DEV)

    # Handle translations
    if translations is not None:
        translations = torch.tensor(translations, dtype=torch.float32, device=DEV)
    else:
        translations = None

    # Number of samples
    samples = quaternions.shape[0]

    # Imaging model
    if isinstance(image_modality, torch.nn.Module):
        imaging_model = image_modality
    else:
        if image_modality == 'sum_projection':
            imaging_model = IMT.Sum3d2d(dim=-1)
        elif image_modality == 'sum_projection_avg_weighted':
            imaging_model = IMT.SumAvgWeighted3d2d(dim=-1)
        elif image_modality in ['brightfield', 'darkfield', 'iscat', 'fluorescence']:
            optics = IMT.setup_optics(shape=(nx, ny, nz), microscopy_regime=image_modality.capitalize())
            imaging_model = IMT.imaging_model(optics)
        elif image_modality == 'scalar_propagation':
            imaging_model = IMT.PropagationImagingModel(nx=nx, ny=ny, nz=nz, wavelength=0.532)
        else:
            raise ValueError(f'Unknown imaging modality: {image_modality}')

    # Number of channels
    ch = 1
    if getattr(imaging_model, 'microscopy_regime', '') == 'brightfield' and getattr(imaging_model, 'filtered_properties', {}).get('return_field', False):
        ch = 2
        V0 = imaging_model(object*0).to(DEV)
        V0_phase = torch.median(torch.angle(V0)).to(DEV)
        V0 = V0 * torch.exp(-1j * V0_phase)

    # Rotation model
    rotmod = A.Tomography(
        volume_size=object.shape, # The size of the volume
        )

    # Dataset
    projections = torch.zeros((samples, ch, nx, ny), device=DEV)
    
    # Generate dataset
    for i in range(samples):
        if i % 100 == 0 and i > 0:
            print(f'Simulating... {i/samples * 100:.1f}%')

        # Rotate volume
        volume_rot = rotmod.apply_rotation_batch(volume=object, quaternions=quaternions[i:i+1], translations=translations[i:i+1] if translations is not None else None)

        # Forward model
        image = imaging_model(volume_rot)

        if imaging_model.microscopy_regime in ['sum_projection', 'sum_projection_avg_weighted']:
            projections[i, 0] = image.cpu().squeeze()
            
        elif imaging_model.microscopy_regime == 'brightfield' and ch == 2:
            image = image * torch.exp(-1j * V0_phase)
            image = image - V0 + 1
            projections[i, 0] = image.real.cpu().squeeze()
            projections[i, 1] = image.imag.cpu().squeeze()
        else:
            projections[i, 0] = image.cpu().squeeze().real

    return object, quaternions, projections, imaging_model


if __name__ == '__main__':

    object, quaternions, projections, imaging_model = create_data(
        image_modality='sum_projection', 
        rotation_case='random_sinusoidal', 
        samples=10,
        duration=0.1,
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