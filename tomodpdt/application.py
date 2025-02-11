import deeplay as dl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from typing import Optional, Sequence#, Callable, List

# Importing the necessary modules
import estimate_rotations_from_latent as erfl
import imaging_modality_brightfield_torch as imb
import vaemod as vm

import deeptrack as dt
#so = imb.setup_optics(nsize=96)
from deeplay.external import Adam

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
        self.optimizer = optimizer if optimizer is not None else Adam(lr=8e-4)
        
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

        # Placeholder
        self.normalize = False

    def initialize_parameters(self, projections, **kwargs):
        
        # Check if projections are a tensor
        if not isinstance(projections, torch.Tensor):
            projections = torch.tensor(projections)
        
        # Move projections to the device
        projections = projections.to(self._device)

        # Set the number of channels
        self.CH = projections.shape[1]

        if self.CH > 1:
            # Update the VAE model to handle multiple channels
            vae = vm.ConvVAE(input_shape=(self.CH, self.N, self.N), latent_dim=2)
            self.vae_model.encoder=vae.encoder
            self.vae_model.decoder=vae.decoder
            self.vae_model.fc_mu=vae.fc_mu
            self.vae_model.fc_var=vae.fc_var
            self.vae_model.fc_dec=vae.fc_dec
            
        # Normalize projections
        if 'normalize' in kwargs and kwargs['normalize']:
            # Compute the global min/max values per channel over the entire dataset
            self.compute_global_min_max(projections)
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
        #self.frames = self.frames.squeeze(1)

        # Set the optimizer
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=8e-4)
        @self.optimizer.params
        def params(self):
            return self.parameters()

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
        trainer = dl.Trainer(max_epochs=400, accelerator="auto")
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
            self.volume = nn.Parameter(torch.zeros(self.N, self.N, self.N, device=self._device)+1.33)

        elif self.initial_volume == 'constant':
            self.volume = nn.Parameter(torch.ones(self.N, self.N, self.N, device=self._device))

        elif self.initial_volume == 'random':
            self.volume = nn.Parameter(torch.rand(self.N, self.N, self.N, device=self._device))

        elif self.initial_volume == 'given' and self.volume_init is not None:
            self.volume = nn.Parameter(self.volume_init.to(self._device))
        
        else:
            raise ValueError("Invalid initial volume type. Must be 'gaussian', 'zeros', 'constant', 'random', or 'given'.")

    def forward(self, idx):
        """
        Forward pass of the model. Returns the estimated projections for the 
        given indices by rotating the volume and imaging it.
        """
        volume = self.volume
        quaternions = self.get_quaternions(self.rotation_params)[idx]
        
        # Normalize quaternions during computation - is done in apply_rotation.
        #quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)

        batch_size = quaternions.shape[0]
        estimated_projections = torch.zeros(batch_size, self.CH, self.N, self.N, device=self._device)

        # Rotate the volume and estimate the projections
        for i in range(batch_size):
            rotated_volume = self.apply_rotation(volume, quaternions[i])

            #Check if imaging model is a nn.Module
            if isinstance(self.imaging_model, nn.Module):
                estimated_projection = self.imaging_model(rotated_volume)

                #Check if estimated_projections has a function _value
                if hasattr(estimated_projection, '_value') and self.CH > 1:
                    estimated_projection = torch.concatenate(
                        (estimated_projection._value.real, estimated_projection._value.imag),
                        axis=-1)
                    estimated_projection = estimated_projection.permute(2, 0, 1)

                elif hasattr(estimated_projection, '_value') and self.CH == 1:
                    estimated_projection = estimated_projection._value

                    # Check if the estimated projection is complex and take the imaginary part
                    if estimated_projection.dtype == torch.complex64:
                        estimated_projection = estimated_projection.imag
                        estimated_projection = estimated_projection.permute(2, 0, 1)

                estimated_projections[i] = estimated_projection
            
            else:
                raise ValueError("Imaging model must be a nn.Module.")

            #Hardcoded for now
            # Create a detached version for NumPy-based function
            # rotated_volume_np = rotated_volume.detach().cpu().numpy()
            # rotated_volume_img = dt.Image(rotated_volume_np)
            # im = so['optics'].get(rotated_volume_img, so['limits'], so['fields'], **so['filtered_properties']).imag
            # estimated_projections[i] = torch.tensor(im, device=self._device).squeeze(-1)

        return estimated_projections
    
    def training_step(self, batch, batch_idx):
        idx = batch
        batch = self.frames[idx]

        yhat = self.forward(idx)  # Estimated projections

        # Normalize the estimated projections. Has to be done before computing the loss.
        if self.normalize:
            yhat = self.per_channel_normalization(yhat)

        with torch.no_grad():
            latent_space = self.fc_mu(self.encoder(yhat))   # Estimated latent space

        proj_loss, latent_loss, rtv_loss, qv_loss, q0_loss, rtr_loss = self.compute_loss(yhat, latent_space, batch, idx)

        # Compute the total loss
        tot_loss = proj_loss + latent_loss + rtv_loss + qv_loss + q0_loss + rtr_loss

        loss = {
            "proj_loss": proj_loss, 
            "latent_loss": latent_loss, 
            "rtv_loss": rtv_loss, 
            "qv_loss": qv_loss, 
            "q0_loss": q0_loss,
            "rtr_loss": rtr_loss,
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

        return proj_loss, latent_loss, rtv_loss, qv_loss, q0_loss, rtr_loss

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

    def gaussian_blur_projection(self, projections, sigma=1):
        """
        Apply Gaussian blur to a set of projections.

        Args:
        - projections (torch.Tensor): A tensor of shape (N, H, W) representing N projections.
        - sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
        - blurred_projections (torch.Tensor): Blurred projections.
        """
        # Create Gaussian kernel
        kernel_size = 2 * int(3 * sigma) + 1
        grid = torch.arange(kernel_size, dtype=projections.dtype, device=projections.device) - kernel_size // 2
        x, y = torch.meshgrid(grid, grid, indexing='ij')
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        # Reshape kernel to match conv2d input requirements
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        # Apply Gaussian blur
        blurred_projections = F.conv2d(projections, kernel, padding=kernel_size // 2)

        return blurred_projections

    def full_forward_final(self):
        """
        Forward pass of the model.

        Args:
        - volume (torch.Tensor): The volume to rotate.
        - quaternions (torch.Tensor): Quaternions representing rotations.

        Returns:
        - estimated_projections (torch.Tensor): Estimated projections.
        """
        self.basis = self.basis.to(self._device)
        self.rotation_params = self.rotation_params.to(self._device)

        quaternions = self.get_quaternions(self.rotation_params).to(self._device)
        volume = self.volume.to(self._device)

        #Set the grid to the device as well...
        self.grid = self.grid.to(self._device)

        # Set the global min/max values to the device
        self.global_min = self.global_min.to(self._device)
        self.global_max = self.global_max.to(self._device)

        # Initialize the estimated projections
        estimated_projections = torch.zeros(quaternions.shape[0], self.CH, self.N, self.N, device=self._device)

        # Rotate the volume and estimate the projections
        for i in range(quaternions.shape[0]):
            rotated_volume = self.apply_rotation(volume, quaternions[i])
             #Check if imaging model is a nn.Module

            if isinstance(self.imaging_model, nn.Module):
                estimated_projection = self.imaging_model(rotated_volume)

                #Check if estimated_projections has a function _value
                if hasattr(estimated_projection, '_value') and self.CH > 1:
                    estimated_projection = torch.concatenate(
                        (estimated_projection._value.real, estimated_projection._value.imag)
                        ,axis=-1)
                    estimated_projection = estimated_projection.permute(2, 0, 1)

                elif hasattr(estimated_projection, '_value') and self.CH == 1:
                    estimated_projection = estimated_projection._value

                    # Check if the estimated projection is complex and take the imaginary part
                    if estimated_projection.dtype == torch.complex64:
                        estimated_projection = estimated_projection.imag
                        estimated_projection = estimated_projection.permute(2, 0, 1)

                estimated_projections[i] = estimated_projection
            
            else:
                raise ValueError("Imaging model must be a nn.Module.")

        if self.normalize:
            estimated_projections = self.per_channel_normalization(estimated_projections)

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
        

# Testing the code
if __name__ == "__main__":
    import numpy as np
    import plotting

    from importlib import reload
    reload(plotting)

    data = np.load('../test_data/test_data_b.npz', allow_pickle=True)
    projections = data["projections"] if "projections" in data else None
    #projections = torch.tensor(projections, dtype=torch.float32).unsqueeze(1) if projections is not None else None
    # Projections is a real and imaginary part of the projections
    
    projections = torch.tensor(projections, dtype=torch.complex64).unsqueeze(1)
    #projections = torch.tensor(projections.imag, dtype=torch.float32).squeeze(-1)
    projections = torch.concat((projections.real, projections.imag), dim=1).squeeze(-1)

    test_object = torch.tensor(data["volume"], dtype=torch.float32) if "volume" in data else None
    #test_object = test_object.unsqueeze(0)

    q_gt = torch.tensor(data["quaternions"], dtype=torch.float32) if "quaternions" in data else None

    #Downsample the projections 2x and downsample the object 2x
    scale = 0.5
    projections = F.interpolate(projections, scale_factor=scale, mode='bilinear')
    
    try:
        test_object = F.interpolate(test_object.unsqueeze(0).unsqueeze(0), scale_factor=scale, mode='trilinear').squeeze(0).squeeze(0)
    except:
        test_object = None

    # Dummy Imaging model
    #imaging_model = vm.Dummy3d2d()
    optics_setup = imb.setup_optics(nsize=48)
    imaging_model = imb.imaging_model(optics_setup)

    # Assuming the projections are square and the volume is cubic
    N = projections.shape[-1]
  
    # Create the tomography model
    tomo = Tomography(volume_size=(N, N, N), rotation_optim_case='basis', initial_volume='zeros', imaging_model=imaging_model)

    # Initialize the parameters
    tomo.initialize_parameters(projections, normalize=True)
    
    # Visualize the latent space and the initial rotations
    plotting.plots_initial(tomo, gt=q_gt)

    # Train the model
    N = len(tomo.frames)
    idx = torch.arange(N)

    trainer = dl.Trainer(max_epochs=5, accelerator="auto", log_every_n_steps=10)
    trainer.fit(tomo, DataLoader(idx, batch_size=64, shuffle=False))

    # Plot the training history
    try:
        trainer.history.plot()
    except:
        print("No history to plot...")

    # Visualize the final volume and rotations.
    plotting.plots_optim(tomo, gt_q=q_gt, gt_v=test_object)


    #Check if tomo.volume has gradients
    print(tomo.volume.grad)