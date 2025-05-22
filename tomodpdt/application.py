import deeplay as dl
from deeplay.external import Adam

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from typing import Optional, Sequence
import time

# Importing the necessary modules
try: 
    import tomodpdt.estimate_rotations_from_latent as erfl
    import tomodpdt.vaemod as vm
    import tomodpdt.plotting as plotting
    import tomodpdt.simulate as sim
except:
    import estimate_rotations_from_latent as erfl
    import vaemod as vm
    import plotting
    import simulate as sim


class Tomography(dl.Application):
    def __init__(self,
                 volume_size: Optional[Sequence[int]] = (96, 96, 96),
                 vae_model: Optional[torch.nn.Module] = None,
                 imaging_model: Optional[torch.nn.Module] = None,
                 initial_volume: Optional[str] = None,  # Initial guess for volume
                 rotation_optim_case: Optional[str] = None,  # Rotation optimization case ('quaternion', 'basis')
                 translation_maxmin = None,  # Max/min translation values, if None, no translation is applied.
                 optimizer = None,
                 volume_init = None,  # Initial guess for volume explicitly
                 minibatch = 64,
                 loss_weights = None,
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
        
        # Set initial volume if provided, otherwise default to "zeros"
        self.initial_volume = initial_volume if initial_volume is not None else "zeros"
        
        # Set the rotation optimization case, default to "quaternion"
        self.rotation_optim_case = rotation_optim_case if rotation_optim_case is not None else "quaternion"

        # Set the translation max/min values if provided
        if translation_maxmin is not None and isinstance(translation_maxmin, (int, float)):
            self.translation_maxmin = translation_maxmin
            self.optimize_translation = True
        else:
            self.translation_maxmin = None
            self.optimize_translation = False
        
        # Set the optimizer (if provided) - not used as of now...
        self.optimizer = optimizer if optimizer is not None else Adam(lr=1e-3)
        
        # Set volume initialization (if provided)
        self.volume_init = volume_init

        # Set the minibatch size - default to 64 - can speed up training
        self.minibatch = minibatch

        # Set the loss weights
        self.loss_weights = loss_weights if loss_weights is not None else {
            'proj_loss': 10,
            'latent_loss': 0.5,
            'rtv_loss': 0.5,
            'qv_loss': 10,
            'q0_loss': 100,
            'rtr_loss': 1,
            'rtr_trans_loss': 1,
            'so_loss': 1
            }
        
        # Raise error if loss weights don´t contain all the necessary keys
        if not all(k in self.loss_weights for k in ['proj_loss', 'latent_loss', 'rtv_loss', 'qv_loss', 'q0_loss', 'rtr_loss', 'rtr_trans_loss', 'so_loss']):
            raise ValueError("Loss weights must contain all the necessary keys.")
        
        # Call the superclass constructor
        super().__init__(**kwargs)

        # Set the imaging model function if 'projection' is selected
        if self.imaging_model == None:
            def projection(volume):
                # Simple projection summing along one axis
                return torch.sum(volume, dim=-1)
            self.imaging_model = projection

        # Set the grid for rotating the volume
        x = torch.arange(self.N) - self.N / 2
        self.xx, self.yy, self.zz = torch.meshgrid(x, x, x, indexing='ij')
        self.grid = torch.stack([self.xx, self.yy, self.zz], dim=-1).to(self._device)

        # Set the grid for batch rotation
        lin = torch.linspace(-1, 1, self.N, device=self._device)
        x, y, z = torch.meshgrid(lin, lin, lin, indexing='ij')
        self.grid_batch = torch.stack((x, y, z), dim=-1).reshape(-1, 3)  # (D*H*W, 3)

        # Placeholder
        self.normalize = False

    def initialize_parameters(self, projections, **kwargs):
        """
        Function to initialize the parameters of the model, 
        set the initial rotation parameters and other settings.
        """
        # Check if projections are a tensor
        if not isinstance(projections, torch.Tensor):
            projections = torch.tensor(projections)
        
        # Move projections to the device
        projections = projections.to(self._device)

        # Compute the global min/max values per channel over the entire dataset
        self.compute_global_min_max(projections)

        # Set the number of channels
        self.CH = projections.shape[1]

        if self.CH > 0 and self.N >= 32:
            # Update the VAE model to handle multiple channels
            vae = vm.ConvVAE(input_shape=(self.CH, self.N, self.N), latent_dim=2)
            self.vae_model.encoder = vae.encoder
            self.vae_model.decoder = vae.decoder
            self.vae_model.fc_mu = vae.fc_mu
            self.vae_model.fc_var = vae.fc_var
            self.vae_model.fc_dec = vae.fc_dec
            self.vae_model.beta = 0.025
            
        # Normalize projections
        if 'normalize' in kwargs and kwargs['normalize']:
            # Compute the global min/max values per channel over the entire dataset
            projections = self.per_channel_normalization(projections)
            self.normalize = True

        # Train the VAE model if not already trained
        if self.vae_model.training:
            self.train_vae(projections, **kwargs)

        # Compute the latent space
        latent_space = self.vae_model.fc_mu(self.vae_model.encoder(projections))
        self.latent = latent_space

        # Retrieve the initial rotation parameters
        self.rotation_initial_dict = erfl.process_latent_space(
            z=latent_space, 
            frames=projections, 
            **kwargs
            )

        # Set the rotation parameters
        if self.rotation_optim_case == 'quaternion':
            rotation_params = self.rotation_initial_dict['quaternions']
        elif self.rotation_optim_case == 'basis':
            rotation_params = self.rotation_initial_dict['coeffs']
            self.basis = self.rotation_initial_dict['basis']
        else:
            raise ValueError("Invalid rotation optimization case. Must be 'quaternion' or 'basis'. as of now...")
        
        # Move the rotation parameters to the device and make them nn.Parameters
        self.rotation_params = nn.Parameter(rotation_params.to(self._device))

        # Initialize the volume
        self.initialize_volume()

        # Initialize the translation parameters if needed
        if self.optimize_translation:
            self.initialize_translation(N=self.rotation_initial_dict["peaks"][-1].item())
        else:
            self.translation_params = None

        # Setting frames to the number of rotations
        self.frames = projections[:self.rotation_initial_dict["peaks"][-1].item()]

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=8e-4)
        @self.optimizer.params
        def params(self):
            return self.parameters()

    def compute_global_min_max(self, projections):
        """
        Compute the global min/max values per channel over the entire dataset.
        """
        # Compute the global min/max values per channel over the entire dataset
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
    
    def train_vae(self, projections, **kwargs):
        """
        Train the VAE model on the given projections.
        """

        if 'max_epochs' in kwargs:
            max_epochs = kwargs['max_epochs']
        else:
            max_epochs = 500

        # Data loader for the VAE model x=projections and y=projections
        data_loader = DataLoader(
            TensorDataset(projections, projections), batch_size=32, shuffle=True
            )

        # Build the VAE model
        self.vae_model.build()

        # Train the VAE model
        trainer = dl.Trainer(max_epochs=max_epochs, accelerator="auto")
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

    def initialize_translation(self, N):
        """
        Initialize the translation parameters.
        """
        # Initialize the translation parameters
        self.translation_params = torch.zeros(N, 3, device=self._device)
        self.translation_params = nn.Parameter(self.translation_params)

    def forward(self, idx):
        """
        Forward pass of the model. Returns the estimated projections for the 
        given indices by rotating the volume and imaging it.
        """
        volume = self.volume
        quaternions = self.get_quaternions(self.rotation_params)[idx]
        translations = self.get_translations(self.translation_params)[idx] if self.optimize_translation else None

        batch_size = quaternions.shape[0]
        estimated_projections_batch = torch.zeros(batch_size, self.CH, self.N, self.N, device=self._device)

        # Create minibatches for rotation
        if batch_size < self.minibatch:
            self.minibatch = batch_size

        indexes = torch.arange(0, batch_size)
        b_idx = [indexes[i:i + self.minibatch] for i in range(0, len(indexes), self.minibatch)]
        # volumes = torch.stack([volume.clone() for _ in range(self.minibatch)])

        for b in b_idx:
            # Apply rotations to a single volume using a batch of quaternions
            rotated_volumes = self.apply_rotation_batch(
                volume, 
                quaternions[b], 
                translations=translations[b] if translations is not None else None
            )
            
            # Check if imaging model is a nn.Module
            if isinstance(self.imaging_model, nn.Module):
                estimated_projections = self.imaging_model(rotated_volumes)

                if self.CH == 1:
                    # Check if the estimated projection is complex and take the imaginary part
                    if estimated_projections.dtype == torch.complex64:
                        estimated_projections = estimated_projections.imag
                    estimated_projections = estimated_projections.permute(0, 3, 1, 2)
                
                # If two channels are present, concatenate them - for complex valued projections
                elif self.CH > 1 and estimated_projections.dtype == torch.complex64:
                    estimated_projections = torch.concatenate(
                        (estimated_projections.real, estimated_projections.imag),
                        axis=-1)
                    estimated_projections = estimated_projections.permute(0, 3, 1, 2)

                elif self.CH > 1:
                    estimated_projections = estimated_projections.permute(0, 3, 1, 2)

                # Add the estimated projections to the batch
                estimated_projections_batch[b] = estimated_projections
            
            else:
                raise ValueError("Imaging model must be a nn.Module.")

        return estimated_projections_batch
    
    def training_step(self, batch, batch_idx):
        """
        Training step for the model. Computes the loss and logs it.
        """

        # Get the batch of frames and the corresponding indices
        idx_batch = batch
        frames_batch = self.frames[idx_batch]

        # Forward step - Estimate the projections
        yhat = self.forward(idx_batch) 

        # Normalize the estimated projections
        if self.normalize:
            yhat = self.per_channel_normalization(yhat)

        # Estimate the latent space
        with torch.no_grad():
            latent_space = self.fc_mu(self.encoder(yhat))
        
        # Compute the losses
        proj_loss, latent_loss, rtv_loss, qv_loss, q0_loss, rtr_loss, rtr_trans_loss, so_loss = self.compute_loss(
            yhat, latent_space, frames_batch, idx_batch, self.loss_weights
            )

        # Compute the total loss
        tot_loss = proj_loss + latent_loss + rtv_loss + qv_loss + q0_loss + rtr_loss + rtr_trans_loss + so_loss

        loss = {
            "total_loss": tot_loss,
            "proj_loss": proj_loss, 
            "latent_loss": latent_loss, 
            "rtv_loss": rtv_loss, 
            "qv_loss": qv_loss, 
            "q0_loss": q0_loss,
            "rtr_loss": rtr_loss,
            "rtr_trans_loss": rtr_trans_loss,
            "so_loss": so_loss,
            }
        
        # Remove these losses from the dictionary if they are exactly 0 for nicer logging/plotting
        if rtr_trans_loss == 0: 
            loss.pop('rtr_trans_loss')

        if rtr_loss == 0: 
            loss.pop('rtr_loss')

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
    
    def compute_loss(self, yhat, latent_space, frames_batch, idx_batch, loss_weights=None):
        """
        Compute the projection loss, latent loss, and other regularization terms.
        """

        # Compute the projection loss - MAE
        proj_loss = F.l1_loss(yhat, frames_batch)

        # Compute the latent loss - distance in latent space between the estimated and true latent space in MAE
        latent_loss = F.l1_loss(latent_space, self.latent[idx_batch])

        # Compute the total variation regularization term
        rtv_loss = self.total_variation_regularization(self.volume)

        # This is the predicted quaternions
        quaternions_pred = self.get_quaternions(self.rotation_params)[idx_batch]

        # This is the predicted translations
        translations_pred = self.get_translations(self.translation_params)[idx_batch] if self.optimize_translation else None

        # Compute the quaternion validity loss
        qv_loss = self.quaternion_validity_loss(
            quaternions_pred
            )

        # Compute the q0 constraint loss if 0 is in the indices
        if torch.sum(idx_batch == 0) > 0:
            q0_loss = self.q0_constraint_loss(
                quaternions_pred[idx_batch == 0]
                )
        else:
            q0_loss = torch.tensor(0.0, device=self._device)

        # Compute the rotational trajectory regularization term if the indices are consecutive and optimization case is 'quaternion' and not 'basis'
        if torch.abs(idx_batch[1:] - idx_batch[:-1]).sum() == len(idx_batch) - 1 and self.rotation_optim_case == 'quaternion':
            rtr_loss = self.rotational_trajectory_regularization(
                quaternions_pred
                )
        else:
            rtr_loss = torch.tensor(0.0, device=self._device)

        # Compute the trajectory regularization term for the translations
        if self.optimize_translation and translations_pred is not None and torch.abs(idx_batch[1:] - idx_batch[:-1]).sum() == len(idx_batch) - 1:
            rtr_trans_loss = self.rotational_trajectory_regularization(
                translations_pred
                )
        else:
            rtr_trans_loss = torch.tensor(0.0, device=self._device)

        # Compute the strictly over loss
        if self.initial_volume == 'refraction':
            so_loss = self.strictly_over_loss(self.volume)
        else:
            so_loss = self.strictly_over_loss(self.volume, value=0)

        # Scale the losses
        if loss_weights is not None and isinstance(loss_weights, dict):
            proj_loss *= loss_weights['proj_loss']
            latent_loss *= loss_weights['latent_loss']
            rtv_loss *= loss_weights['rtv_loss']
            qv_loss *= loss_weights['qv_loss']
            q0_loss *= loss_weights['q0_loss']
            rtr_loss *= loss_weights['rtr_loss']
            rtr_trans_loss *= loss_weights['rtr_trans_loss']
            so_loss *= loss_weights['so_loss']
        else:
            proj_loss *= 10
            latent_loss *= 0.5
            rtv_loss *= 0.5
            qv_loss *= 10
            q0_loss *= 100
            rtr_loss *= 1
            rtr_trans_loss *= 1
            so_loss *= 1
        
        return proj_loss, latent_loss, rtv_loss, qv_loss, q0_loss, rtr_loss, rtr_trans_loss, so_loss

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
        Faster rotational trajectory regularization term.
        
        Args:
        - q (torch.Tensor): Tensor of shape (T, d)
        
        Returns:
        - reg_terms (float): Regularization value
        """
        # Compute first and second-order differences directly
        first_diff = torch.diff(q, dim=0)  # Shape: (T-1, d)
        second_diff = torch.diff(first_diff, dim=0)  # Shape: (T-2, d)

        # Compute squared norms (avoid .norm() call for speed)
        first_order_loss = torch.sum(first_diff ** 2, dim=1)  # (T-1,)
        second_order_loss = torch.sum(second_diff ** 2, dim=1)  # (T-2,)

        # Combine and normalize
        reg_terms = (torch.sum(first_order_loss) + torch.sum(second_order_loss)) / q.size(0)

        return reg_terms

    def get_translations(self, raw_translation):
        max_translation = self.translation_maxmin if self.translation_maxmin is not None else 1.0
        return max_translation * torch.tanh(raw_translation)

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

    def apply_rotation(self, volume, q, translations=None):
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
        rotated_grid = (rotated_grid / (self.N / 2)).clamp(-1, 1).unsqueeze(0)
        
        # Apply translation if given
        if translations is not None:
            # Normalize voxel translations to [-1, 1]
            t_norm = torch.zeros(3).to(rotated_grid.device)
            t_norm[2] = 2 * translations[0] / (self.N - 1)  # z (depth)
            t_norm[1] = 2 * translations[1] / (self.N - 1)  # y (height)
            t_norm[0] = 2 * translations[2] / (self.N - 1)  # x (width)
            rotated_grid -= t_norm.view(1, 1, 1, 3)  # broadcast over (N, N, N, 3)

        rotated_grid = rotated_grid.view(1, self.N, self.N, self.N, 3).clamp(-1, 1)  # Shape: (1, N, N, N, 3)

        # Apply grid_sample to rotate the volume
        rotated_volume = F.grid_sample(
            volume.unsqueeze(0).unsqueeze(0), 
            rotated_grid, 
            align_corners=True,
            )

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

    def apply_rotation_batch(self, volume, quaternions, translations=None):
        """
        Applies a batch of rotations (and optional translations) to a single 3D volume.

        Args:
            volume (torch.Tensor): Input volume of shape (D, H, W) or (1, D, H, W).
            quaternions (torch.Tensor): Batch of rotation quaternions of shape (B, 4).
            translations (torch.Tensor or None): Optional translations of shape (B, 3),
                                                in voxel units (dz, dy, dx).

        Returns:
            torch.Tensor: Rotated volumes of shape (B, D, H, W).
        """
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)  # (1, D, H, W)
        volume = volume.unsqueeze(0)     # (1, 1, D, H, W)
        _, _, D, H, W = volume.shape

        B = quaternions.shape[0]

        # Repeat the single volume B times
        volumes = volume.expand(B, -1, -1, -1, -1)  # (B, 1, D, H, W)

        # Get rotation matrices
        R = self.quaternion_to_rotation_matrix_batch(quaternions)  # (B, 3, 3)

        # Prepare and expand grid
        grid = self.grid_batch.to(volume.device)  # (D*H*W, 3)
        grid = grid.unsqueeze(0).expand(B, -1, -1)  # (B, D*H*W, 3)

        # Rotate the grid
        rotated_grid = torch.bmm(grid, R.transpose(1, 2))  # (B, D*H*W, 3)

        # Apply translation if needed
        if translations is not None:
            t_norm = translations.clone()
            t_norm[:, 2] = 2 * translations[:, 0] / (D - 1)  # dz → z
            t_norm[:, 1] = 2 * translations[:, 1] / (H - 1)  # dy → y
            t_norm[:, 0] = 2 * translations[:, 2] / (W - 1)  # dx → x
            rotated_grid -= t_norm[:, None, :]

        # Reshape and clamp
        rotated_grid = rotated_grid.view(B, D, H, W, 3).clamp(-1, 1)

        # Sample and return
        transformed = F.grid_sample(volumes, rotated_grid, align_corners=True)
        return transformed.squeeze(1)  # (B, D, H, W)

    def full_forward_final(self, max_projections=None):
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

        # Get the quaternions
        quaternions = self.get_quaternions(self.rotation_params).to(self._device)

        # Get the translations
        translations = self.get_translations(self.translation_params).to(self._device) if self.optimize_translation else None

        # Get the volume
        volume = self.volume.to(self._device)

        #Set the grid to the device as well...
        self.grid = self.grid.to(self._device)
        self.grid_batch = self.grid_batch.to(self._device)

        # Set the global min/max values to the device
        self.global_min = self.global_min.to(self._device)
        self.global_max = self.global_max.to(self._device)

        # If max_projections is not None, set the number of projections to max_projections. Saves time and memory.
        if max_projections is not None:
            quaternions = quaternions[:max_projections]
            translations = translations[:max_projections] if translations is not None else None

        # Initialize the estimated projections
        estimated_projections = torch.zeros(quaternions.shape[0], self.CH, self.N, self.N, device=self._device)

        # Rotate the volume and estimate the projections
        for i in range(quaternions.shape[0]):
            rotated_volume = self.apply_rotation(
                volume, 
                quaternions[i], 
                translations=translations[i] if translations is not None else None
                )

            # Check if imaging model is a nn.Module
            if isinstance(self.imaging_model, nn.Module):
                estimated_projection = self.imaging_model(rotated_volume)

                if self.CH == 1:

                    # Check if the estimated projection is complex and take the imaginary part
                    if estimated_projection.dtype == torch.complex64:
                        estimated_projection = estimated_projection.imag
                    
                    estimated_projection = estimated_projection.permute(2, 0, 1)

                # If two channels are present, concatenate them - for complex valued projections
                elif self.CH > 1 and estimated_projection.dtype == torch.complex64:
                    estimated_projection = torch.concatenate(
                        (estimated_projection.real, estimated_projection.imag)
                        , axis=-1)
                    estimated_projection = estimated_projection.permute(2, 0, 1)

                # If more than 2 channels are present, permute the dimensions
                elif self.CH > 1:
                    estimated_projection = estimated_projection.permute(2, 0, 1)

            else:
                raise ValueError("Imaging model must be a nn.Module.")

            estimated_projections[i] = estimated_projection

        # Normalize the estimated projections
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
        
    def get_translations_final(self, raw_translation=None):
        """
        Get translations from the translation parameters."""

        if raw_translation is None:
            raw_translation = self.translation_params

        max_translation = self.translation_maxmin if self.translation_maxmin is not None else 1.0
        return max_translation * torch.tanh(raw_translation)
        
    def move_all_to_device(self, device):
        """
        Move all parameters to the given device.
        """
        self.to(device)
        self.rotation_params = self.rotation_params.to(device)

        if self.optimize_translation:
            self.translation_params = self.translation_params.to(device)

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
        if self.optimize_translation and self.translation_params is not None:
            self.translation_params.requires_grad = requires_grad

    def swap_rotation_axis(self):
        """ 
        Swap the rotation axis. Between x and y rotation. 
        """
        # Swap the x and y rotation
        self.rotation_params[:, [1, 2]] = self.rotation_params[:, [2, 1]]


# Testing the code
if __name__ == "__main__":
    pass



        