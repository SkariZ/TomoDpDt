import sys
sys.path.append('..')

import tomodpdt
from tomodpdt.imaging_modality_torch import SumAvgWeighted3d2d, Sum3d2d

import numpy as np
import os
import time
import matplotlib.pyplot as plt

import deeplay as dl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', DEV)

plotly_3d = False # Do interactive 3D plots with plotly
SAVE_FOLDER = 'C:/Users/Fredrik/Desktop/tomo_sims_c/' # Save the results to a folder
os.makedirs(SAVE_FOLDER, exist_ok=True)

from medmnist import OrganMNIST3D
from torchvision import transforms
import torch

# Define a transform that properly converts the 4D numpy array to a tensor
transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32))  # Convert to Tensor
])

# Load dataset with the new transform
train_dataset = OrganMNIST3D(root='../test_data', split='train', download=True, transform=transform)

# Define the model and which axis to sum over
SUM_MODEL = Sum3d2d(dim=-1)

ROT_CASES = ['noisy_sinusoidal', 'sinusoidal', 
            'random_sinusoidal', '1ax', 
            'smooth_varying','smooth_varying_random'
            ]
   
N_VOLUMES = 5
N_SAMPLES = 400 # Number of projections
N_DURATION = 2 # Number of full revolutions

SEED = 133792
np.random.seed(SEED)
VOL_IDX = np.random.randint(0, len(train_dataset), N_VOLUMES)

from tomodpdt.imaging_modality_torch import setup_optics, imaging_model

# Setup the optics
optics_setttings = setup_optics(
        nsize=48, 
        padding_xy=64, 
        microscopy_regime='Brightfield', 
        NA=0.7, 
        wavelength=532e-9, 
        resolution=100e-9, 
        magnification=1, 
        return_field=True)

# Generate the imaging model
brightfield_model = imaging_model(optics_setup=optics_setttings)
SUM_MODEL = brightfield_model

if __name__ == "__main__":

    
    for i in range(N_VOLUMES):
        volume, class_label = train_dataset[VOL_IDX[i]]
        volume = volume[0].numpy()

        # ´Normalize volume to 1.33-1.38
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
        volume = volume * (1.38 - 1.33) + 1.33
        volume = np.clip(volume, 1.33, 1.38)

        # Upsample volume to 48x48x48 by nearest neighbor interpolation in torch
        volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Add batch and channel dimensions
        volume = torch.nn.functional.interpolate(volume, size=(48, 48, 48), mode='nearest')
        volume = volume.squeeze(0).squeeze(0) # Remove batch and channel dimensions
        volume = volume.numpy() # Convert to numpy array
    
        Nc = volume.shape[0]

        for rotation_case in ROT_CASES:

            save_folder = os.path.join(SAVE_FOLDER, f'Volume_{VOL_IDX[i]}_' + rotation_case+'/')
            os.makedirs(save_folder, exist_ok=True)
            print('Saving to:', save_folder)

            test_object, q_gt, projections, imaging_model = tomodpdt.simulate.create_data(
                volume=volume, # The volume we want to reconstruct
                image_modality=SUM_MODEL, # We use the sum model
                rotation_case=rotation_case, # We rotate the object around 1 main axis, but the other 2 axes are also non-zero an
                samples=N_SAMPLES, # Number of projections
                duration=N_DURATION # Duration is the number of full revolutions
                )
            
            # Create the tomographic_model
            tomographic_model = tomodpdt.Tomography(
                volume_size=(Nc, Nc, Nc), # The size of the volume
                initial_volume='refraction', # 'refraction' or 'zeros', 'refraction' is the initial volume
                rotation_optim_case='basis', # 'basis' or 'quaternion', 'basis' is smoother
                imaging_model=SUM_MODEL, # The imaging model
                )
            
            # Initialize the parameters
            tomographic_model.initialize_parameters(projections, normalize=True)

            tomodpdt.plotting.plots_initial(tomographic_model, gt=q_gt.to('cpu'), save_folder=save_folder)
            plt.close('all')

            N = len(tomographic_model.frames) # Number of frames
            idx = torch.arange(N) # Index of frames

            epochs_object_only = 100 # Number of epochs for the object only optimization
            batch_size_object_only = 64 # Batch size for the object only optimization

            # Toggle the gradients of the quaternion parameters to False
            tomographic_model.toggle_gradients_quaternion(False)

            # Move the model to device
            tomographic_model.move_all_to_device(DEV)

            # Train the model
            start_time = time.time()
            trainer = dl.Trainer(max_epochs=epochs_object_only, accelerator="auto", log_every_n_steps=10)
            trainer.fit(tomographic_model, DataLoader(idx, batch_size=batch_size_object_only , shuffle=True))
            print("Training time: ", (time.time() - start_time) / 60, " minutes")

            try:
                plt.figure(figsize=(8, 8))
                trainer.history.plot()
                plt.title('Training history')
                plt.savefig(os.path.join(save_folder, 'training_history_object_only.png'))
                plt.close()
            except:
                print("No training history available.")
            
            ### 3.4 - Optimize the 3D volume and the rotation parameters
            epochs_object_rot = 1250
            batch_size_object_rot = 128

            # Toggle the gradients of the quaternion parameters
            tomographic_model.toggle_gradients_quaternion(True)

            # Move the model to device
            tomographic_model.move_all_to_device(DEV)

            # Train the model
            start_time = time.time()
            trainer = dl.Trainer(max_epochs=epochs_object_rot, accelerator="auto", log_every_n_steps=10)
            trainer.fit(tomographic_model, DataLoader(idx, batch_size=batch_size_object_rot, shuffle=False))
            print("Training time: ", (time.time() - start_time) / 60, " minutes")

            try:
                plt.figure(figsize=(8, 8))
                trainer.history.plot()
                plt.title('Training history')
                plt.savefig(os.path.join(save_folder, 'training_history_full_training.png'))
                plt.close()
            except:
                print("No training history available.")
            

            # Move it to the GPU if possible for faster plotting
            tomographic_model.move_all_to_device(DEV)

            # Visualize the final volume and rotations.
            tomodpdt.plotting.plots_optim(tomographic_model, gt_q=q_gt.to('cpu'), gt_v=test_object.to('cpu'), plot_3d=plotly_3d, save_folder=save_folder)
            plt.close('all')

            ## 5 - Save the reconstructed volume and the parameters
            if save_folder is not None:
                
                # save the volume as a numpy array
                np.save(f'{save_folder}/volume.npy', tomographic_model.volume.cpu().detach().numpy())

                # save the quaternions as a numpy array
                quaternions_pred = tomographic_model.get_quaternions_final().detach().cpu().numpy()
                np.save(f'{save_folder}/quaternions.npy', quaternions_pred)
