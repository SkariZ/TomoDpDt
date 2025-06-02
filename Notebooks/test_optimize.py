import sys
sys.path.append('..')

import tomodpdt

import numpy as np
import time
import matplotlib.pyplot as plt

import deeplay as dl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
from skimage.restoration import unwrap_phase
from skimage.transform import resize


DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', DEV)

SAVE_PATH = 'results/optimize_test/'
os.makedirs(SAVE_PATH, exist_ok=True)


# Make volume to a cube
def make_cube(volume):
    # Get the shape of the volume
    z, y, x = volume.shape

    # Find the minimum dimension
    min_dim = min(z, y, x)

    # Calculate the center of the volume
    center_z = z // 2
    center_y = y // 2
    center_x = x // 2

    # Calculate the half size of the cube
    half_size = min_dim // 2

    # Calculate the start and end indices for each dimension
    start_z = center_z - half_size
    end_z = center_z + half_size
    start_y = center_y - half_size
    end_y = center_y + half_size
    start_x = center_x - half_size
    end_x = center_x + half_size

    # Extract the cube from the volume
    cube = volume[start_z:end_z, start_y:end_y, start_x:end_x]

    return cube

# Make volume to a cube
def make_square(image):

    # Get the shape of the image
    x, y = image.shape

    # Find the minimum dimension
    min_dim = min(x, y)
    # Calculate the center of the image
    center_x = x // 2
    center_y = y // 2
    # Calculate the half size of the square
    half_size = min_dim // 2
    # Calculate the start and end indices for each dimension
    start_x = center_x - half_size
    end_x = center_x + half_size
    start_y = center_y - half_size
    end_y = center_y + half_size
    # Extract the square from the image
    square = image[start_x:end_x, start_y:end_y]
    # Create a new square image with the desired shape
    new_square = np.zeros((min_dim, min_dim), dtype=image.dtype)
    # Assign the values from the original square to the new square
    new_square[:square.shape[0], :square.shape[1]] = square
    return new_square

def correctfield(field, n_iter=5):
    """
    Correct field
    """

    if field.dtype == torch.float32:
        field = field.to(torch.complex64)

    f_new = field.clone()

    # Normalize with mean of absolute value.
    f_new = f_new / torch.mean(torch.abs(f_new))

    for _ in range(n_iter):
        f_new = f_new * torch.exp(-1j * torch.median(torch.angle(f_new)))

    return f_new

if __name__ == "__main__":

    data = np.load('D:/NewDataMonica/data_full_09_10_2024_HEK_03_04_us4.npz', allow_pickle=True)

    E = data['E']
    E = E[..., :, 13:-13]
    E = np.stack([make_square(e) for e in E])
    E = E[:, 12:-12, 12:-12]

    E_upd = E.copy()
    E = correctfield(torch.tensor(E_upd, device=DEV, dtype=torch.complex64), n_iter=5).cpu().numpy()

    phases = np.stack([unwrap_phase(np.angle(E[i])) for i in range(E.shape[0])])

    for i in range(E_upd.shape[0]):
        proj_upd = phases[i]
        mask = proj_upd < 0.58
        E_upd[i][mask] = 1 

    E = E_upd
    del E_upd

    N = 144
    E_r = np.stack([resize(e.real, (N, N)) for e in E])
    E_i = np.stack([resize(e.imag, (N, N)) for e in E])

    E = E_r + 1j * E_i
    E = np.stack([np.real(E), np.imag(E)], axis=1)

    Q = data['q']

    q_new = Q.copy()

    q_new[..., 1] = Q[..., 2]
    q_new[..., 2] = Q[..., 3]
    q_new[..., 3] = Q[..., 1]
    Q = q_new
    del q_new


    from tomodpdt.imaging_modality_torch import setup_optics, imaging_model

    # Setup the optics
    optics_setttings = setup_optics(
            nsize=N, 
            padding_xy=N//2, 
            microscopy_regime='Brightfield', 
            NA=1.15, 
            wavelength=640e-9, 
            resolution=200e-9, 
            magnification=1, 
            return_field=True)

    # Generate the imaging model
    brightfield_model = imaging_model(optics_setup=optics_setttings)


    #Take 250 consecutive frames each time
    c_frames = 250
    iterations = 75
    batch_size_object_only = 16 # Batch size for the object only optimization
    counter = 0

    for i in range(0, E.shape[0], c_frames):
        counter += 1
        # Create folder for each batch
        batch_folder = os.path.join(SAVE_PATH, f'batch_{counter}')
        os.makedirs(batch_folder, exist_ok=True)

        print(f'Processing frames {i} to {i + c_frames}')
        E_batch = E[i:i+250]
        Q_batch = Q[i:i+250]

        # Convert to torch tensors
        E_torch = torch.tensor(E_batch, device=DEV, dtype=torch.float32)
        Q_torch = torch.tensor(Q_batch, device=DEV, dtype=torch.float32)

        # Create the tomographic_model
        tomographic_model = tomodpdt.Tomography(
            volume_size=(N, N, N), # The size of the volume
            initial_volume='zeros', # 'refraction' since we are optimizing the refractive index
            rotation_optim_case='quaternion', # 'basis' or 'quaternion', 'basis' is smoother
            imaging_model=brightfield_model, # The imaging model,
            translation_maxmin=8, # The maximum and minimum translation
         )
        
        # Initialize the parameters
        tomographic_model.initialize_parameters(E_torch, normalize=False, initial_frames_per_rotation=125, max_epochs=200)

        tomographic_model.rotation_params = torch.nn.Parameter(
            torch.tensor(Q_torch, device=DEV, dtype=torch.float32)
            )
        
        n_idx = len(tomographic_model.frames) # Number of frames
        idx = torch.arange(n_idx) # Index of frames

        # Toggle the gradients of the quaternion parameters to False
        tomographic_model.toggle_gradients_quaternion(False)

        # Move the model to device
        tomographic_model.move_all_to_device(DEV)

        # Train the model
        start_time = time.time()
        trainer = dl.Trainer(max_epochs=iterations, accelerator="auto", log_every_n_steps=10)
        trainer.fit(tomographic_model, DataLoader(idx, batch_size=batch_size_object_only, shuffle=True))
        print("Training time: ", (time.time() - start_time) / 60, " minutes")

        vol = tomographic_model.volume.cpu().detach().numpy()

        # Save the volume
        np.save(os.path.join(batch_folder, 'volume.npy'), vol)

        # Plot the volume
        tomodpdt.plotting.plot_sum_object(vol + 1.33, save_folder=batch_folder+'/', save_name='sum_object.png')

        # Save the tomographic model
        torch.save(tomographic_model.state_dict(), os.path.join(batch_folder, 'tomographic_model.pth'))

        del tomographic_model, E_torch, Q_torch, vol
        torch.cuda.empty_cache()



