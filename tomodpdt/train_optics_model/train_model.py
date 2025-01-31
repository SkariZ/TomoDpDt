import torch
import torch.nn as nn
import torch.nn.functional as F

import deeplay as dl
import deeptrack as dt
from torch.utils.data import DataLoader, TensorDataset

import model
import object_getter
import imaging_modality_brightfield as imb

import matplotlib.pyplot as plt

NORMALIZATION_FACTORS = {
    'magnification': (1, 5),
    'wavelength': (400e-9, 700e-9),
    'resolution': (800e-10, 300e-9),
    'NA': (0.7, 1.3)
}

def normalize_params(params, normalization_factors):
    
    magnification_min, magnification_max = normalization_factors['magnification']
    wavelength_min, wavelength_max = normalization_factors['wavelength']
    resolution_min, resolution_max = normalization_factors['resolution']
    NA_min, NA_max = normalization_factors['NA']

    magnification = (params[:, 0] - magnification_min) / (magnification_max - magnification_min)
    wavelength = (params[:, 1] - wavelength_min) / (wavelength_max - wavelength_min)
    resolution = (params[:, 2] - resolution_min) / (resolution_max - resolution_min)
    NA = (params[:, 3] - NA_min) / (NA_max - NA_min)

    return torch.stack([magnification, wavelength, resolution, NA], dim=1)

def simulation(optics, object):
    """
    Simulate the imaging process of the optical system.

    Args:
        optics (dict): A dictionary containing the optical system.
        object (np.ndarray): The object to image.

    Returns:
        np.ndarray: The simulated image.
    """

    object = dt.Image(object)

    image = optics['optics'].get(object, optics['limits'], optics['fields'], **optics['filtered_properties'])

    return image

def get_optics(NA, wavelength, resolution, magnification):

    optics = imb.setup_optics(
        nsize=64,
        NA=NA, 
        wavelength=wavelength,
        resolution=resolution, 
        magnification=magnification, 
        return_field=True,
        )
    
    return optics

def simulate_training_samples(magnification_range, wavelength_range, resolution_range, NA_range, volume_size, n_samples=32):

    x_3d = torch.zeros(n_samples, 1, volume_size, volume_size, volume_size)
    y_2d = torch.zeros(n_samples, 2, volume_size, volume_size)
    params = torch.zeros(n_samples, 4)

    for i in range(n_samples):
        
        magnification = torch.rand(1) * (magnification_range[1] - magnification_range[0]) + magnification_range[0]
        wavelength = torch.rand(1) * (wavelength_range[1] - wavelength_range[0]) + wavelength_range[0]
        resolution = torch.rand(1) * (resolution_range[1] - resolution_range[0]) + resolution_range[0]
        NA = torch.rand(1) * (NA_range[1] - NA_range[0]) + NA_range[0]

        # Normalize the parameters
        magnification_norm = (magnification - magnification_range[0]) / (magnification_range[1] - magnification_range[0])
        wavelength_norm = (wavelength - wavelength_range[0]) / (wavelength_range[1] - wavelength_range[0])
        resolution_norm = (resolution - resolution_range[0]) / (resolution_range[1] - resolution_range[0])
        NA_norm = (NA - NA_range[0]) / (NA_range[1] - NA_range[0])

        params[i] = torch.tensor([magnification_norm, wavelength_norm, resolution_norm, NA_norm])

        # Get the optics
        optics = get_optics(NA.numpy(), wavelength.numpy(), resolution.numpy(), magnification.numpy())

        # Get the object
        object = object_getter.get_random_objects(
            shape=(volume_size, volume_size, volume_size), n_objects=1, inside_ri_range=(1.34, 1.60)
            )

        # Simulate the imaging process
        image = simulation(optics, object)

        # Set the object to the volume
        # Normalize it so 1.33 is 0 and 1.6 is 1
        object = object.squeeze()
        object = (object - 1.33) / (1.6 - 1.33)
        x_3d[i] = object.unsqueeze(0)

        # Set the image to the 2D tensor
        y_2d[i] = torch.stack([torch.tensor(image.real.squeeze()), torch.tensor(image.imag.squeeze())], dim=0)

    return x_3d, y_2d, params

if __name__ == "__main__":

    num_params = 4  # Magnification, wavelength, resolution, NA

    magnification_range = (1, 5)
    wavelength_range = (400e-9, 700e-9)
    resolution_range = (800e-10, 300e-9)
    NA_range = (0.7, 1.3)

    volume_size = 64

    # Initialize model
    model = model.NeuralMicroscope(num_params=num_params)

    #number of parameters in the model
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Generate training sample
    x_3d, y_2d, params = simulate_training_samples(
        magnification_range, 
        wavelength_range, 
        resolution_range, 
        NA_range, 
        volume_size, 
        n_samples=64
        )

    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    #Costum loss
    def custom_loss(output, target, crop=8):
        #Crop the output to avoid big errors at the edges
        output = output[:, :, crop:-crop, crop:-crop]
        target = target[:, :, crop:-crop, crop:-crop]
        return torch.mean(torch.abs(output - target))

    #Generate new samples every 5 epochs
    total_losses = []
    for epoch in range(1000):

        if epoch % 5 == 0 and epoch > 0:
            x_3d, y_2d, params = simulate_training_samples(
                magnification_range, 
                wavelength_range, 
                resolution_range, 
                NA_range, 
                volume_size, 
                n_samples=64
                )

        x_3d = x_3d.to(device)
        params = params.to(device)
        y_2d = y_2d.to(device)

        data_loader = DataLoader(
            TensorDataset(x_3d, params, y_2d), 
            batch_size=32, shuffle=True
            )

        total_loss = 0
        for x, p, y in data_loader:
            optimizer.zero_grad()

            output = model(x, p)

            loss = custom_loss(output, y)

            loss.backward()
            optimizer.step()

            # Clip gradients by their norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            total_loss += loss.item()

        total_losses.append(total_loss)  
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


    # Test the model
    for _ in range(5):
        x_3d, y_2d, params = simulate_training_samples(
            magnification_range, 
            wavelength_range, 
            resolution_range, 
            NA_range, 
            volume_size, 
            n_samples=1
            )
        
        x_3d = x_3d.to(device)
        params = params.to(device)
        y_2d = y_2d.to(device)

        output = model(x_3d, params)

        #Plot the output
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(output[0, 0].detach().cpu().numpy())
        plt.colorbar()
        plt.title("Real")
        plt.subplot(1, 2, 2)
        plt.imshow(output[0, 1].detach().cpu().numpy())
        plt.colorbar()
        plt.title("Imaginary")
        plt.show()

        #Plot the
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(y_2d[0, 0].detach().cpu().numpy())
        plt.colorbar()
        plt.title("Real")
        plt.subplot(1, 2, 2)
        plt.imshow(y_2d[0, 1].detach().cpu().numpy())
        plt.colorbar()
        plt.title("Imaginary")
        plt.show()



    ### Test the model with a random volume.
    import numpy as np
    data = np.load('../../test_data/test_data.npz', allow_pickle=True)
    data = torch.tensor(data["volume"], dtype=torch.float32)

    #Resize to 64x64x64
    volume = F.interpolate(data.unsqueeze(0).unsqueeze(0), size=64, mode='trilinear', align_corners=True).squeeze().squeeze()

    #Set pixels that are 0 to 1.33
    volume[volume == 0] = 1.33

    #Set the pixels that are not 0 to range from 1.36 to 1.42 ie. min and max of the pixel values in the volume is 1.33 and 1.42
    volume[volume != 1.33] = volume[volume != 1.33] / torch.max(volume[volume != 1.33]) * 0.06 + 1.36

    # image the volume
    object = dt.Image(volume)
    optics = imb.setup_optics(
        nsize=64,
        NA=0.7, 
        wavelength=532e-9,
        resolution=100e-9, 
        magnification=1, 
        return_field=True,
        )
    image = optics['optics'].get(object, optics['limits'], optics['fields'], **optics['filtered_properties'])

    #Set the image to the 2D tensor
    y = torch.stack([torch.tensor(image.real.squeeze()), torch.tensor(image.imag.squeeze())], dim=0)

    #Normalize the parameters


    #Add channel dimension
    volume = volume.unsqueeze(0).unsqueeze(0).to(device)

    output = model(volume, torch.tensor([[0.75, 0.75, 0.75, 0.75]]).to(device))