import deeplay as dl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from typing import Optional, Sequence#, Callable, List

# Importing the necessary modules
import estimate_rotations_from_latent as erfl
import vaemod as vm

import deeptrack as dt
from deeplay.external import Adam

import numpy as np
import time
import plotting

class Tomography(dl.Application):
    def __init__(self,
                 volume_size: Optional[Sequence[int]] = (96, 96, 96),
                 vae_model: Optional[torch.nn.Module] = None,
                 imaging_model: Optional[torch.nn.Module] = None,
                 initial_volume: Optional[str] = None,  # Initial guess for volume
                 rotation_optim_case: Optional[str] = None,  # Rotation optimization case ('quaternion', 'basis')
                 optimizer=None,
                 volume_init=None,  # Initial guess for volume explicitly
                 **kwargs):
        
        # Set volume size
        self.N = volume_size[0]
        self.volume_size = volume_size
        
        # If VAE model is not passed, initialize a default VAE model
        self.vae_model = vae_model if vae_model is not None else dl.VariationalAutoEncoder(input_size=(self.volume_size[0], self.volume_size[1]), latent_dim=2)
        
        # Set the encoder and other VAE components
        self.encoder = self.vae_model.encoder
        self.fc_mu = self.vae_model.fc_mu
        
        # Set the imaging model (either passed as a module or projection function)
        self.imaging_model = imaging_model if imaging_model is not None else "projection"
        
        # Determine the device (cuda if available, else cpu)
        self._device = torch.device("cuda" if torch.cuda.is_available() else getattr(vae_model, "device", "cpu"))
        #self._device = torch.device("cpu")
        # Set initial volume if provided, otherwise default to "zeros"
        self.initial_volume = initial_volume if initial_volume is not None else "zeros"
        
        # Set the rotation optimization case, default to "quaternion"
        self.rotation_optim_case = rotation_optim_case if rotation_optim_case is not None else "quaternion"
        
        # Set the optimizer (if provided) - not used as of now...
        self.optimizer = optimizer if optimizer is not None else Adam(lr=1e-3)
        
        # Set volume initialization (if provided)
        self.volume_init = volume_init
        
        # Call the superclass constructor
        super().__init__(**kwargs)

        # Set the imaging model function if 'projection' is selected
        if self.imaging_model == "projection":
            def projection(volume):
                # Simple projection summing along one axis (dim=2)
                return torch.sum(volume, dim=-1)
            self.imaging_model = projection

        # Set the grid for rotating the volume
        x = torch.arange(self.N) - self.N / 2
        self.xx, self.yy, self.zz = torch.meshgrid(x, x, x, indexing='ij')
        self.grid = torch.stack([self.xx, self.yy, self.zz], dim=-1).to(self._device)

        # Create base 3D grid
        lin = torch.linspace(-1, 1, self.N, device=self._device)
        x, y, z = torch.meshgrid(lin, lin, lin, indexing='ij')
        self.grid_batch = torch.stack((x, y, z), dim=-1).reshape(-1, 3)  # (D*H*W, 3)

        # Placeholder
        self.normalize = False

    def initialize_parameters(self, projections, **kwargs):
        
        # Check if projections are a tensor
        if not isinstance(projections, torch.Tensor):
            projections = torch.tensor(projections)
        
        # Move projections to the device
        projections = projections.to(self._device)

        # Compute the global min/max values per channel over the entire dataset
        self.compute_global_min_max(projections)

        # Set the number of channels
        self.CH = projections.shape[1]

        if self.CH > 0:
            # Update the VAE model to handle multiple channels
            vae = vm.ConvVAE(input_shape=(self.CH, self.N, self.N), latent_dim=2)
            self.vae_model.encoder=vae.encoder
            self.vae_model.decoder=vae.decoder
            self.vae_model.fc_mu=vae.fc_mu
            self.vae_model.fc_var=vae.fc_var
            self.vae_model.fc_dec=vae.fc_dec
            self.vae_model.beta = 0.025
            
        # Normalize projections
        if 'normalize' in kwargs and kwargs['normalize']:
            # Compute the global min/max values per channel over the entire dataset
            projections = self.per_channel_normalization(projections)
            self.normalize = True

        # Train the VAE model if not already trained
        if self.vae_model.training:
            self.train_vae(projections)

        # Compute the latent space
        latent_space = self.vae_model.fc_mu(self.vae_model.encoder(projections))
        self.latent = latent_space

        # Initialize the volume
        self.initialize_volume()

        # Retrieve the initial rotation parameters
        self.rotation_initial_dict = erfl.process_latent_space(
            z=latent_space, 
            frames=projections,
            #quaternions=QGT, 
            **kwargs
            )  # Later: add axis also

        # Set the rotation parameters
        if self.rotation_optim_case == 'quaternion':
            rotation_params = self.rotation_initial_dict['quaternions']
        elif self.rotation_optim_case == 'basis':
            rotation_params = self.rotation_initial_dict['coeffs']
            self.basis = self.rotation_initial_dict['basis']
        else:
            raise ValueError("Invalid rotation optimization case. Must be 'quaternion' or 'basis'. as of now...")
        
        self.rotation_params = nn.Parameter(rotation_params.to(self._device))

        # Setting frames to the number of rotations
        self.frames = projections[:self.rotation_initial_dict["peaks"][-1].item()]

        # Normalize back to the original range
        #if self.normalize:
        #    self.frames = self.per_channel_denormalization(self.frames)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=8e-4)
        @self.optimizer.params
        def params(self):
            return self.parameters()
        
        # Fit the model to the data
        #self.fit(self.frames)

    def compute_global_min_max(self, projections):
        """
        Compute the global min/max values per channel over the entire dataset.
        """
        global_min = torch.amin(projections, dim=(0, 2, 3))
        global_max = torch.amax(projections, dim=(0, 2, 3))  

        # Set the global min/max values
        self.global_min = global_min.to(self._device)
        self.global_max = global_max.to(self._device)

    def per_channel_normalization(self, projections):
        """
        Normalize the projections per channel using precomputed global min/max scaling.
        """
        for i in range(projections.shape[1]):  # Iterate over channels
            projections[:, i] = (projections[:, i] - self.global_min[i]) / (self.global_max[i] - self.global_min[i] + 1e-6)  # Prevent division by zero
        return projections
    
    def per_channel_denormalization(self, projections):
        """
        Denormalize the projections per channel using precomputed global min/max scaling.
        """
        for i in range(projections.shape[1]):  # Iterate over channels
            projections[:, i] = projections[:, i] * (self.global_max[i] - self.global_min[i] + 1e-6) + self.global_min[i]
        return projections
    
    def train_vae(self, projections):
        """
        Train the VAE model on the given projections.
        """

        # Data loader for the VAE model x=projections and y=projections
        data_loader = DataLoader(
            TensorDataset(projections, projections), batch_size=32, shuffle=True
            )

        # Build the VAE model
        self.vae_model.build()

        # Train the VAE model
        trainer = dl.Trainer(max_epochs=250, accelerator="auto")
        trainer.fit(self.vae_model, data_loader)

        # Freeze the VAE model
        for param in self.vae_model.parameters():
            param.requires_grad = False

        # Freeze the encoder layer
        for param in self.vae_model.encoder.parameters():
            param.requires_grad = False

        # Freeze the fc_mu layer
        for param in self.vae_model.fc_mu.parameters():
            param.requires_grad = False

        # Update the VAE model and the needed components and move them to the device
        self.encoder = self.vae_model.encoder.to(self._device)
        self.fc_mu = self.vae_model.fc_mu.to(self._device)
        
    def initialize_volume(self):
        """
        Initialize the volume.

        Returns:
        - volume (torch.Tensor): Initialized volume.
        """
        if self.initial_volume == 'gaussian':
            x = torch.arange(self.N) - self.N / 2
            xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
            cloud = torch.exp(-0.001 * (xx**2 + yy**2 + zz**2))
            cloud = cloud / cloud.max()
            self.volume = nn.Parameter(cloud.to(self._device))

        elif self.initial_volume == 'zeros':
            self.volume = nn.Parameter(torch.zeros(self.N, self.N, self.N, device=self._device) + 1e-6)
        
        elif self.initial_volume == 'refraction':
            self.volume = nn.Parameter(torch.ones(self.N, self.N, self.N, device=self._device) * 1.33)

        elif self.initial_volume == 'random':
            self.volume = nn.Parameter(torch.rand(self.N, self.N, self.N, device=self._device))

        elif self.initial_volume == 'given' and self.volume_init is not None:
            self.volume = nn.Parameter(self.volume_init.to(self._device))
        
        else:
            raise ValueError("Invalid initial volume type. Must be 'gaussian', 'zeros', 'constant', 'random', or 'given'.")

    def forward(self, idx, minibatch=64):
        """
        Forward pass of the model. Returns the estimated projections for the 
        given indices by rotating the volume and imaging it.
        """
        volume = self.volume
        quaternions = self.get_quaternions(self.rotation_params)[idx]
        
        # Normalize quaternions during computation - is done in apply_rotation.
        #quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)

        batch_size = quaternions.shape[0]
        estimated_projections_batch = torch.zeros(batch_size, self.CH, self.N, self.N, device=self._device)

        # Create minibatches for rotation
        if batch_size < minibatch:
            minibatch = batch_size

        indexes = torch.arange(0, batch_size)
        b_idx = [indexes[i:i + minibatch] for i in range(0, len(indexes), minibatch)]
        volumes = torch.stack([volume.clone() for _ in range(minibatch)])

        for b in b_idx:
            #Rotate the volume(s)
            rotated_volumes = self.apply_rotation_batch(volumes[:len(b)], quaternions[b])
 
            #Check if imaging model is a nn.Module
            if isinstance(self.imaging_model, nn.Module):
                estimated_projections = self.imaging_model(rotated_volumes)

                # if two channels are present, concatenate them - for complex valued projections
                if self.CH > 1:
                    estimated_projections = torch.concatenate(
                        (estimated_projections.real, estimated_projections.imag),
                        axis=-1)
                    estimated_projections = estimated_projections.permute(0, 3, 1, 2)

                elif self.CH == 1:
                    # Check if the estimated projection is complex and take the imaginary part
                    if estimated_projections.dtype == torch.complex64:
                        estimated_projections = estimated_projections.imag
                    estimated_projections = estimated_projections.permute(0, 3, 1, 2)

                estimated_projections_batch[b] = estimated_projections
            
            else:
                raise ValueError("Imaging model must be a nn.Module.")

        return estimated_projections_batch
    
    def training_step(self, batch, batch_idx):
        idx = batch
        batch = self.frames[idx]

        yhat = self.forward(idx)  # Estimated projections

        # Normalize the estimated projections
        if self.normalize:
            yhat = self.per_channel_normalization(yhat)
   
        with torch.no_grad():
            latent_space = self.fc_mu(self.encoder(yhat))   # Estimated latent space

        proj_loss, latent_loss, rtv_loss, qv_loss, q0_loss, rtr_loss, so_loss = self.compute_loss(yhat, latent_space, batch, idx)

        # Compute the total loss
        tot_loss = proj_loss + latent_loss + rtv_loss + qv_loss + q0_loss + rtr_loss + so_loss

        loss = {
            "proj_loss": proj_loss, 
            "latent_loss": latent_loss, 
            "rtv_loss": rtv_loss, 
            "qv_loss": qv_loss, 
            "q0_loss": q0_loss,
            "rtr_loss": rtr_loss,
            "so_loss": so_loss,
            "total_loss": tot_loss
            }
        for name, v in loss.items():
            self.log(
                f"train_{name}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return tot_loss
    
    def compute_loss(self, yhat, latent_space, batch, idx):
        
        # Compute the projection loss - MAE
        proj_loss = F.l1_loss(yhat, batch)

        # Compute the latent loss - distance in latent space between the estimated and true latent space in MAE
        latent_loss = F.l1_loss(latent_space, self.latent[idx])

        # Compute the total variation regularization term
        rtv_loss = self.total_variation_regularization(self.volume)

        # This is the predicted quaternions
        quaternions_pred = self.get_quaternions(self.rotation_params)[idx]

        # Compute the quaternion validity loss
        qv_loss = self.quaternion_validity_loss(
            quaternions_pred
            )

        # Compute the q0 constraint loss if 0 is in the indices
        if torch.sum(idx == 0) > 0:
            q0_loss = self.q0_constraint_loss(
                quaternions_pred[idx == 0]
                )
        else:
            q0_loss = torch.tensor(0.0, device=self._device)

        # Compute the rotational trajectory regularization term if the indices are consecutive
        if torch.abs(idx[1:] - idx[:-1]).sum() == len(idx) - 1:
            rtr_loss = self.rotational_trajectory_regularization(
                quaternions_pred
                )
        else:
            rtr_loss = torch.tensor(0.0, device=self._device)

        # Compute the strictly over  loss
        if self.initial_volume == 'refraction':
            so_loss = self.strictly_over_loss(self.volume)
        else:
            so_loss = self.strictly_over_loss(self.volume, value=0)

        # Scale the losses
        proj_loss *= 10
        latent_loss *= 0.5
        rtv_loss *= 0.5
        so_loss *= 1
        q0_loss *= 100
        qv_loss *= 10

        return proj_loss, latent_loss, rtv_loss, qv_loss, q0_loss, rtr_loss, so_loss

    def strictly_over_loss(self, volume, value=1.33):
        """
        Computes a loss that penalizes values strictly below a value
        """
        loss = torch.sum(torch.relu(value - volume))  # Penalize values below 
        return loss / volume.numel()  # Normalize by total elements

    def total_variation_regularization(self, delta_n):
        """
        Calculate the total variation regularization term in 3D without creating large intermediate tensors.

        Args:
        - delta_n (torch.Tensor): A tensor of shape (D, H, W) or higher dimensional array.

        Returns:
        - R_TV (float): The total variation regularization term.
        """
        # Compute gradients and sum them inline to avoid intermediate tensors
        grad_x_sum = torch.sum(torch.abs(delta_n[1:, :, :] - delta_n[:-1, :, :]))  # Gradient in x-direction
        grad_y_sum = torch.sum(torch.abs(delta_n[:, 1:, :] - delta_n[:, :-1, :]))  # Gradient in y-direction
        grad_z_sum = torch.sum(torch.abs(delta_n[:, :, 1:] - delta_n[:, :, :-1]))  # Gradient in z-direction

        # Combine all gradient sums
        R_TV = (grad_x_sum + grad_y_sum + grad_z_sum) / delta_n.numel()

        return R_TV

    def quaternion_validity_loss(self, q):
        """
        Loss to enforce that quaternions remain valid (unit quaternions).

        Args:
        - q (torch.Tensor): Tensor of quaternions with shape (N, 4), where N is the number of quaternions.

        Returns:
        - loss (torch.Tensor): The quaternion validity loss.
        """
        # Compute the squared norm of the quaternion
        norm_squared = torch.sum(q**2, dim=1)  # Sum over the 4 components (q0, q1, q2, q3) for each quaternion
        
        # Compute the difference between the norm squared and 1
        diff_from_unit = norm_squared - 1
        
        # The loss is the square of the difference
        return torch.sum(diff_from_unit**2) / q.shape[0]
    
    def q0_constraint_loss(self, q):
        """
        Enforce that the q0 component of the quaternion to be [1, 0, 0, 0]. Just a simple constraint. So it stays at the starting point.
        """
        q_start = torch.tensor([1, 0, 0, 0], device=self._device)
        return torch.sum((q - q_start)**2)

    def rotational_trajectory_regularization(self, q):
        """
        Calculate the rotational trajectory regularization term.
        
        Args:
        - q (torch.Tensor): A tensor of shape (T, d) where T is the number of time steps and d is the dimensionality of q.
        Returns:
        - R_q (float): The rotational trajectory regularization term.
        """

        # First-order difference (consecutive quaternion differences)
        first_diff = q[1:] - q[:-1]

        # Second-order difference (smoothness penalty)
        second_diff = first_diff[1:] - first_diff[:-1]

        # Compute the loss
        first_order_loss = first_diff.norm(p=2, dim=1)**2  # Penalize large jumps
        second_order_loss = second_diff.norm(p=2, dim=1)**2  # Penalize abrupt changes in the rate of change

        # Combine first-order and second-order terms
        reg_terms = (torch.sum(first_order_loss) + torch.sum(second_order_loss)) / q.shape[0]

        return reg_terms

    def get_quaternions(self, rotations=None):
        """
        Get quaternions from the rotation parameters."""

        if rotations is None:
            rotations = self.rotation_params

        if self.rotation_optim_case == 'quaternion':
            return rotations
        elif self.rotation_optim_case == 'basis':
            return torch.matmul(self.basis, rotations) 

    def quaternion_to_rotation_matrix(self, q):
        """
        Convert a quaternion to a rotation matrix in a differentiable manner.

        Parameters:
        - q (torch.Tensor): Quaternions of shape (4,).

        Returns:
        - R (torch.Tensor): 3x3 rotation matrix.
        """
        qw, qx, qy, qz = q.unbind()

        # Compute the elements of the rotation matrix directly from quaternion components
        R = torch.stack([
            torch.stack([1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)], dim=-1),
            torch.stack([2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)], dim=-1),
            torch.stack([2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)], dim=-1),
        ], dim=-2)

        return R

    def apply_rotation(self, volume, q):
        """
        Rotate the object using quaternions.

        Parameters:
        - volume (torch.Tensor): The volume to rotate.
        - q (torch.Tensor): Quaternions representing rotations.

        Returns:
        - rotated_volume (torch.Tensor): Rotated volume.
        """

        # Convert quaternions to rotation matrix
        q = q / q.norm()  # Ensure unit quaternion
        R = self.quaternion_to_rotation_matrix(q)
        
        # Create a rotation grid
        grid = self.grid.view(-1, 3)
        rotated_grid = torch.matmul(grid, R.t()).view(self.N, self.N, self.N, 3)
        
        # Normalize the grid values to be in the range [-1, 1] for grid_sample
        rotated_grid = (rotated_grid / (self.N / 2)).clamp(-1, 1)
        
        # Apply grid_sample to rotate the volume
        rotated_volume = F.grid_sample(volume.unsqueeze(0).unsqueeze(0), rotated_grid.unsqueeze(0), align_corners=True)
        return rotated_volume.squeeze()

    def quaternion_to_rotation_matrix_batch(self, q):
        """Convert a batch of quaternions (B, 4) to rotation matrices (B, 3, 3)."""
        q = q / q.norm(dim=1, keepdim=True)  # Normalize quaternions batchwise
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = torch.stack([
            1 - 2*y**2 - 2*z**2,  2*x*y - 2*z*w,  2*x*z + 2*y*w,
            2*x*y + 2*z*w,  1 - 2*x**2 - 2*z**2,  2*y*z - 2*x*w,
            2*x*z - 2*y*w,  2*y*z + 2*x*w,  1 - 2*x**2 - 2*y**2
        ], dim=1).reshape(-1, 3, 3)  # Shape: (B, 3, 3)

        return R

    def apply_rotation_batch(self, volumes, quaternions):
        """
        Rotate a batch of 3D volumes using batch quaternions.

        Parameters:
        - volumes (torch.Tensor): Shape (B, D, H, W) or (B, 1, D, H, W)
        - quaternions (torch.Tensor): Shape (B, 4) (unit quaternions)

        Returns:
        - rotated_volumes (torch.Tensor): Shape (B, D, H, W)
        """
        B, D, H, W = volumes.shape if volumes.dim() == 4 else volumes.shape[0:4]
        if volumes.dim() == 4:
            volumes = volumes.unsqueeze(1)  # Convert (B, D, H, W) → (B, 1, D, H, W)

        # Convert quaternions to rotation matrices
        R = self.quaternion_to_rotation_matrix_batch(quaternions)  # Shape: (B, 3, 3)

        # Create base 3D grid
        grid = self.grid_batch.unsqueeze(0).expand(B, -1, -1)  # (B, D*H*W, 3)

        # Rotate grid batchwise
        rotated_grid = torch.bmm(grid, R.transpose(1, 2))  # (B, D*H*W, 3)
        rotated_grid = rotated_grid.view(B, D, H, W, 3)  # Reshape back

        # Ensure grid values are in range [-1, 1]
        rotated_grid = rotated_grid.clamp(-1, 1)

        # Apply grid_sample in batch mode
        rotated_volumes = F.grid_sample(volumes, rotated_grid, align_corners=True)

        return rotated_volumes.squeeze(1)  # Remove channel dimension if necessary

    def full_forward_final(self):
        """
        Forward pass of the model.

        Args:
        - volume (torch.Tensor): The volume to rotate.
        - quaternions (torch.Tensor): Quaternions representing rotations.

        Returns:
        - estimated_projections (torch.Tensor): Estimated projections.
        """
        if self.rotation_optim_case == 'basis':
            self.basis = self.basis.to(self._device)

        self.rotation_params = self.rotation_params.to(self._device)

        quaternions = self.get_quaternions(self.rotation_params).to(self._device)
        volume = self.volume.to(self._device)

        #Set the grid to the device as well...
        self.grid = self.grid.to(self._device)
        self.grid_batch = self.grid_batch.to(self._device)

        # Set the global min/max values to the device
        self.global_min = self.global_min.to(self._device)
        self.global_max = self.global_max.to(self._device)

        # Initialize the estimated projections
        estimated_projections = torch.zeros(quaternions.shape[0], self.CH, self.N, self.N, device=self._device)

        # Rotate the volume and estimate the projections
        for i in range(quaternions.shape[0]):
            rotated_volume = self.apply_rotation(volume, quaternions[i])

             # Check if imaging model is a nn.Module
            if isinstance(self.imaging_model, nn.Module):
                estimated_projection = self.imaging_model(rotated_volume)

                # Check if estimated_projections has a function _value
                if self.CH > 1:
                    estimated_projection = torch.concatenate(
                        (estimated_projection.real, estimated_projection.imag)
                        , axis=-1)
                    estimated_projection = estimated_projection.permute(2, 0, 1)

                elif self.CH == 1:
                    # Check if the estimated projection is complex and take the imaginary part
                    if estimated_projection.dtype == torch.complex64:
                        estimated_projection = estimated_projection.imag
                    estimated_projection = estimated_projection.permute(2, 0, 1)
            else:
                raise ValueError("Imaging model must be a nn.Module.")

            estimated_projections[i] = estimated_projection

        if self.normalize:
            estimated_projections = self.per_channel_normalization(
                estimated_projections
                )

        return estimated_projections
    
    def get_quaternions_final(self, rotations=None):
        """
        Get quaternions from the rotation parameters."""

        if rotations is None:
            rotations = self.rotation_params

        if self.rotation_optim_case == 'quaternion':
            rotations = rotations / rotations.norm(dim=-1, keepdim=True)
            return rotations
        elif self.rotation_optim_case == 'basis':
            rotations = torch.matmul(self.basis.to(self._device), rotations)
            rotations = rotations / rotations.norm(dim=-1, keepdim=True)
            return rotations
        
    def move_all_to_device(self, device):
        """
        Move all parameters to the given device.
        """
        self.to(device)
        self.rotation_params = self.rotation_params.to(device)
        self.volume = self.volume.to(device)
        if self.rotation_optim_case == 'basis':
            self.basis = self.basis.to(device)
        self.grid = self.grid.to(device)
        self.grid_batch = self.grid_batch.to(device)
        self.global_min = self.global_min.to(device)
        self.global_max = self.global_max.to(device)
    
    def toggle_grad(self, requires_grad):
        """
        Toggle the requires_grad attribute of all parameters.
        """
        for param in self.parameters():
            param.requires_grad = requires_grad

    def toggle_gradients_quaternion(self, requires_grad):
        """
        Toggle the requires_grad attribute of the quaternion parameters.
        """
        self.rotation_params.requires_grad = requires_grad

    def swap_rotation_axis(self):
        """ 
        Swap the rotation axis. betwen x and y rotation. 
        """
        # Swap the x and y rotation
        self.rotation_params[:, [1, 2]] = self.rotation_params[:, [2, 1]]


# Testing the code
if __name__ == "__main__":
    import simulate as sim
    import os

    image_modality_list = ['sum_projection']#, 'darkfield', 'brightfield', 'sum_projection']#, 'darkfield', 'brightfield', 'sum_projection']
    rotation_case_list = ['random_sinusoidal']
    save_folder_root = '../results'
    if not os.path.exists(save_folder_root):
        os.makedirs(save_folder_root)

    for image_modality in image_modality_list:
        for rotation_case in rotation_case_list:

            save_folder = f"{save_folder_root}/{image_modality}_{rotation_case}/"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            print(image_modality, rotation_case)
            test_object, q_gt, projections, imaging_model = sim.create_data(image_modality=image_modality, rotation_case=rotation_case, samples=100)

            # Downsample the projections 2x and downsample the object 2x
            scale = 1
            projections = F.interpolate(projections, scale_factor=scale, mode='bilinear')
            try:
                test_object = F.interpolate(test_object.unsqueeze(0).unsqueeze(0), scale_factor=scale, mode='trilinear').squeeze(0).squeeze(0)
            except:
                test_object = None

            # Assuming the projections are square and the volume is cubic
            N = projections.shape[-1]

            # Create the tomography model
            tomo = Tomography(
                volume_size=(N, N, N),
                rotation_optim_case='basis',
                initial_volume='zeros' if image_modality == 'fluorescence' else 'refraction',
                imaging_model=imaging_model
                )

            # Initialize the parameters
            tomo.initialize_parameters(projections, normalize=True)
            
            # Train the model
            N = len(tomo.frames)
            idx = torch.arange(N)

            start_time = time.time()

            # Toggle the gradients rotation params to off for initial phase
            tomo.toggle_gradients_quaternion(False)

            # First axis test.
            trainer = dl.Trainer(max_epochs=10, accelerator="auto", log_every_n_steps=10)
            trainer.fit(tomo, DataLoader(idx, batch_size=128, shuffle=True))
            loss1 = trainer.callback_metrics['train_total_loss_epoch'].item()
            vol1 = tomo.volume.clone()
            rot1 = tomo.rotation_params.clone()    

            # Swap the rotation axis.
            tomo.swap_rotation_axis()
            tomo.initialize_volume()
            tomo.move_all_to_device("cuda")
            trainer = dl.Trainer(max_epochs=10, accelerator="auto", log_every_n_steps=10)
            trainer.fit(tomo, DataLoader(idx, batch_size=128, shuffle=True))
            loss2 = trainer.callback_metrics['train_total_loss_epoch'].item()
            vol2 = tomo.volume.clone()
            rot2 = tomo.rotation_params.clone()

            # Choose the best loss and proceed with that.
            if loss1 < loss2:
                print("Loss 1 is better ie. choose optimized axis", loss1, loss2)
                tomo.volume = nn.Parameter(vol1)
                tomo.rotation_params = nn.Parameter(rot1)
            else:
                del vol1, rot1, loss1, vol2, rot2, loss2

            # Visualize the latent space and the initial rotations
            plotting.plots_initial(tomo, save_folder=save_folder, gt=q_gt.to('cpu'))

            #Toggle the gradients of the quaternion parameters
            tomo.toggle_gradients_quaternion(True)
            tomo.move_all_to_device("cuda")
            trainer = dl.Trainer(max_epochs=500, accelerator="auto", log_every_n_steps=10)
            trainer.fit(tomo, DataLoader(idx, batch_size=128, shuffle=False))
            tomo.move_all_to_device("cuda")
            print("Training time: ", (time.time() - start_time) / 60, " minutes")

            # Plot the training history
            try:
                trainer.history.plot()
            except:
                print("No history to plot...")

            # Visualize the final volume and rotations.
            plotting.plots_optim(tomo, save_folder=save_folder, gt_q=q_gt.to('cpu'), gt_v=test_object.to('cpu'))
            
            # Save volume and rotations
            torch.save(tomo.volume, f"{save_folder}volume_{image_modality}_{rotation_case}.pt")
            torch.save(tomo.rotation_params, f"{save_folder}/rotations_{image_modality}_{rotation_case}.pt")

            # print gradients
            #print(tomo.volume.grad)
            #print(tomo.rotation_params.grad)

    #Check if tomo.volume has gradients
    #print(tomo.volume.grad)
    #quaternions = tomo.get_quaternions_final()

    #volumes = torch.stack([tomo.apply_rotation(tomo.volume, q) for q in quaternions])

   
    # take time
    #k = 32
    #start = time.time()
    #x = tomo.apply_rotation_batch(volumes[:k] , quaternions[:k])
    #print("Time taken: ", time.time() - start)

    #start = time.time()
    #for i in range(k):
    #    tomo.apply_rotation(tomo.volume, quaternions[i])
        