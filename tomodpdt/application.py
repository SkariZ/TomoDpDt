import deeplay as dl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from typing import Optional, Sequence#, Callable, List

# Importing the necessary modules
import estimate_rotations_from_latent as erfl

class Tomography(dl.Application):
    def __init__(self,
                 volume_size: Optional[Sequence[int]]=(96, 96, 96),
                 vae_model: Optional[torch.nn.Module]=None,
                 imaging_model: Optional[torch.nn.Module]=None,
                 initial_volume: Optional[str]=None, #initial guess for volume - gaussian, constant, random, given
                 rotation_optim_case: Optional[str]=None, #rotation optimization case - 'quaternion', 'basis'
                 optimizer=None,
                 volume_init=None, #initial guess for volume explicitly
                 **kwargs,
                 ):
        
        self.N = volume_size[0]
        self.volume_size = volume_size
        self.vae_model = vae_model if vae_model is not None else dl.VariationalAutoEncoder(input_size=(self.volume_size[0], self.volume_size[1]), latent_dim=2)
        self.encoder = self.vae_model.encoder
        self.fc_mu = self.vae_model.fc_mu
        self.imaging_model = imaging_model if imaging_model is not None else "projection_model"
        self._device = torch.device("cuda" if torch.cuda.is_available() else getattr(vae_model, "device", "cpu"))
        self.initial_volume = initial_volume
        self.rotation_optim_case = rotation_optim_case if rotation_optim_case is not None else "quaternion"
        self.optimizer = optimizer
        self.volume_init = volume_init
        super().__init__(**kwargs)

        # Set the imaging model
        if imaging_model == "projection_model":
            def projection(volume):
                return torch.sum(volume, dim=2)
            self.imaging_model = projection

        # Set the grid for rotating the volume
        x = torch.arange(self.N) - self.N / 2
        self.xx, self.yy, self.zz = torch.meshgrid(x, x, x, indexing='ij')
        self.grid = torch.stack([self.xx, self.yy, self.zz], dim=-1).to(self._device)

                
    def initialize_parameters(self, projections, **kwargs):
        
        # Check if projections are a tensor
        if not isinstance(projections, torch.Tensor):
            projections = torch.tensor(projections)
        
        # Move projections to the device
        projections = projections.to(self._device)

        # Normalize projections
        if 'normalize' in kwargs:
            if kwargs['normalize']:
                projections = (projections - projections.min()) / (projections.max() - projections.min())
        
        # Add a channel dimension if necessary
        if projections.dim() == 3:
            projections = projections.unsqueeze(1)

        # Train the VAE model if not already trained
        if self.vae_model.training:
            self.train_vae(projections)

        # Compute the latent space
        latent_space = self.vae_model.fc_mu(self.vae_model.encoder(projections))
        self.latent = latent_space

        # Initialize the volume
        self.volume = self.initialize_volume()

        # Retrieve the initial rotation parameters
        self.rotation_initial_dict = erfl.process_latent_space(
            z=latent_space, 
            frames=projections, 
            **kwargs
            ) # Later: add axis also

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
        self.frames = projections[:len(rotation_params)]

        # Set the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)

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
        trainer = dl.Trainer(max_epochs=10, accelerator="auto")
        trainer.fit(self.vae_model, data_loader)

        # Update the VAE model and the needed components and move them to the device
        self.encoder = self.vae_model.encoder.to(self._device)
        self.fc_mu = self.vae_model.fc_mu.to(self._device)

        # Freeze the VAE model
        for param in self.vae_model.parameters():
            param.requires_grad = False

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

        elif self.initial_volume == 'constant':
            self.volume = nn.Parameter(torch.ones(self.N, self.N, self.N, device=self._device))

        elif self.initial_volume == 'random':
            self.volume = nn.Parameter(torch.rand(self.N, self.N, self.N, device=self._device))

        elif self.initial_volume == 'given' and self.volume_init is not None:
            self.volume = nn.Parameter(self.volume_init.to(self._device))

        else:
            self.volume = nn.Parameter(torch.zeros(self.N, self.N, self.N, device=self._device))
    
    def forward(self, idx):
        """
        Forward pass of the model.
        """
        volume = self.volume
        quaternions = self.get_quaternions(self.rotation_params)[idx]

        # Normalize quaternions during computation
        quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)

        batch_size = quaternions.shape[0]
        estimated_projections = torch.zeros(batch_size, self.N, self.N, device=self._device)

        # Rotate the volume and estimate the projections
        for i in range(batch_size):
            rotated_volume = self.apply_rotation(volume, quaternions[i])
            estimated_projections[i] = self.imaging_model(rotated_volume)

        return estimated_projections
    
    def training_step(self, batch, batch_idx):
        #x,y = self.train_preprocess(batch) # batch = ground_truth projections

        yhat = self.forward(batch_idx) # Estimated projections
        latent_space = self.fc_mu(self.encoder(yhat)) # Estimated latent space

        proj_loss, latent_loss, rtv_loss = self.compute_loss(yhat, latent_space, batch, batch_idx)
        tot_loss = proj_loss + latent_loss + rtv_loss

        loss = {"proj_loss": proj_loss, "latent_loss": latent_loss, "rtv_loss":rtv_loss, "total_loss": tot_loss}
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
    
    def compute_loss(self, yhat, latent_space, batch, batch_idx):
        
        # Compute the projection loss - MAE
        proj_loss = F.l1_loss(yhat, batch)

        # Compute the latent loss - distance in latent space between the estimated and true latent space in MAE
        latent_loss = F.l1_loss(latent_space, self.latent[batch_idx])

        # Compute the total variation regularization term
        R_TV = self.total_variation_regularization(self.volume)

        return proj_loss, latent_loss, R_TV

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

    def get_quaternions(self, rotations):
        """
        Get quaternions from the rotation parameters."""

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

        # Ensure input has a channel dimension
        projections = projections.unsqueeze(1)  # Shape (N, 1, H, W)

        # Apply Gaussian blur
        blurred_projections = F.conv2d(projections, kernel, padding=kernel_size // 2)

        return blurred_projections.squeeze(1)  # Remove the channel dimension


# Testing the code
if __name__ == "__main__":

    # Create a dummy dataset
    N = 96
    projections = torch.rand(64, N, N)

    # Create the tomography model
    tomography = Tomography(volume_size=(N, N, N))

    # Initialize the parameters
    tomography.initialize_parameters(projections)

    # Perform a forward pass
    tomography(0)



