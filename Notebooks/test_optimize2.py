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

import cv2


DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', DEV)

SAVE_PATH = 'results/optimize_test3/'
os.makedirs(SAVE_PATH, exist_ok=True)


def centralize_frame(frame, x, y):
    h, w = frame.shape[:2]

    # Calculate shift required to move (x, y) to (w/2, h/2)
    shift_x = (w / 2) - x
    shift_y = (h / 2) - y

    # Apply shift to the entire frame with a constant border (black)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    centralized_frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return centralized_frame


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

    E = data['E'][:1995]
    E_upd = E.copy()
    E = correctfield(torch.tensor(E_upd, device=DEV, dtype=torch.complex64), n_iter=5).cpu().numpy()

    phases = np.stack([unwrap_phase(np.angle(E[i])) for i in range(E.shape[0])])

    # Pad  with zeros to make it square
    max_size = max(phases.shape[1], phases.shape[2])
    diff_size = np.abs(phases.shape[1] - phases.shape[2])
    if phases.shape[2] < phases.shape[1]:
        phases = np.pad(phases, ((0, 0), (0, 0), (diff_size // 2, diff_size // 2)), mode='constant')
        E = np.pad(E, ((0, 0), (0, 0), (diff_size // 2, diff_size // 2)), mode='constant')
    else:
        phases = np.pad(phases, ((0, 0), (diff_size // 2, diff_size // 2), (0, 0)), mode='constant')
        E = np.pad(E, ((0, 0), (diff_size // 2, diff_size // 2), (0, 0)), mode='constant')


    Q = data['q'][:1995]
    q_new = Q.copy()

    q_new[..., 1] = Q[..., 2]
    q_new[..., 2] = Q[..., 3]
    q_new[..., 3] = Q[..., 1]
    Q = q_new
    del q_new


    from tomodpdt.imaging_modality_torch import setup_optics, imaging_model
    N = 128

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

    from tomodpdt import helpers
    handler = helpers.MaskRCNNHandler()


    #Take 250 consecutive frames each time
    c_frames = 200
    iterations = 200
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
        phases_batch = phases[i:i+250]

        projs = []
        for i, proj in enumerate(phases_batch):
            proj_upd = proj.copy()
            proj_upd[proj_upd < 0.57] = 0
            projs.append(proj_upd)
        projs = np.stack(projs)
        projs = torch.tensor(projs).unsqueeze(1).to(DEV).float()
        _, est_xy = helpers.track_and_centralize(frames=projs, maskrcnn=handler)

        median_xy = np.median(est_xy, axis=0)

        aligned_phases = []
        median_align = True
        for i in range(projs.shape[0]):
            if median_align:
                # Use median coordinates for alignment
                x, y = median_xy
            else:
                # Use estimated coordinates for alignment
                x, y = est_xy[i]
            c_frame = centralize_frame(projs[i, 0].cpu().numpy(), x, y)

            aligned_phases.append(c_frame)
        aligned_phases = np.stack(aligned_phases)

        E_upd = E_batch.copy()
        for i in range(E_upd.shape[0]):
            tmp = aligned_phases[i]
            
            # Set values below threshold to 1
            mask = tmp < 0.57

            real_part = np.real(E_upd[i])
            imag_part = np.imag(E_upd[i])

            # Align them using estimated center coordinates
            if median_align:
                # Use median coordinates for alignment
                x, y = median_xy
            else:
                # Use estimated coordinates for alignment
                x, y = est_xy[i]
                
            real_part = centralize_frame(real_part, x, y)
            imag_part = centralize_frame(imag_part, x, y)

            # Combine real and imaginary parts back into a complex field
            E_upd[i] = real_part + 1j * imag_part

            # Set values below threshold to 1
            E_upd[i][mask] = 1 
        E_batch = E_upd
        del E_upd, aligned_phases, projs

        pad_x, pad_y = (E_batch.shape[1]- N) // 2, (E_batch.shape[2] - N) // 2
        E_batch = E_batch[:, pad_x:pad_x + N, pad_y:pad_y + N]

        # Make 2 channels for imag and real part
        E_batch = np.stack([np.real(E_batch), np.imag(E_batch)], axis=1)

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
        tomodpdt.plotting.plot_sum_object(vol + 1.33, save_folder=batch_folder + '/', save_name='sum_object.png')

        # Save the tomographic model
        torch.save(tomographic_model.state_dict(), os.path.join(batch_folder, 'tomographic_model.pth'))

        del tomographic_model, E_torch, Q_torch, vol
        torch.cuda.empty_cache()




# Read  all the volumes and plot them
volumes = []
for folder in os.listdir(SAVE_PATH):
    if folder.startswith('batch_'):
        vol = np.load(os.path.join(SAVE_PATH, folder, 'volume.npy'))
        volumes.append(vol)
volumes = np.stack(volumes)

# Combine the volumes by averaging
combined_volume = np.median(volumes, axis=0)

tomodpdt.plotting.plot_sum_object(combined_volume + 1.33, 
    save_folder=SAVE_PATH, save_name='sum_object_combined_median.png')

# Plot the combined volume
tomodpdt.plotting.visualize_3d_volume(
    combined_volume + 1.33)