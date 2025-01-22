import deeplay as dl
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Sequence, Callable, List

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
        self.vae_model = vae_model or dl.VariationalAutoEncoder(input_size=(self.volume_size[0], self.volume_size[1]))
        self.encoder = self.vae_model.encoder
        self.fc_mu = self.vae_model.fc_mu
        self.imaging_model = imaging_model if imaging_model is not None else "projection_model"
        self.device = self.vae_model.device
        self.initial_volume = initial_volume
        self.rotation_optim_case = rotation_optim_case if rotation_optim_case is not None else "quaternion"
        self.volume_init = volume_init
        super().__init__(**kwargs)

        # Set the optimizer
        self.optimizer = optimizer or torch.nn.optimizers.Adam(lr=5e-3)

        # Set the imaging model
        if imaging_model=="projection_model":
            def projection(volume):
                return torch.sum(volume, dim=0)
            self.imaging_model = projection

        # Set the grid for rotating the volume
        x = torch.arange(self.N) - self.N / 2
        self.xx, self.yy, self.zz = torch.meshgrid(x, x, x, indexing='ij')
        self.grid = torch.stack([self.xx, self.yy, self.zz], dim=-1).to(self.device)

                
    def initialize_parameters(self, projections, **kwargs):
        
        #Normalize projections
        if not isinstance(projections, torch.Tensor):
            projections = torch.tensor(projections)
        
        if 'normalize' in kwargs:
            if kwargs['normalize']:
                projections = (projections - projections.min()) / (projections.max() - projections.min())
        
        # Add a channel dimension if necessary
        if projections.dim() == 3:
            projections = projections.unsqueeze(1)


        latent_space = self.vae.fc_mu(self.vae.encoder(projections))
        self.latent = latent_space
        self.volume = self.initialize_volume()

        # Retrieve the initial rotation parameters
        self.rotation_initial_dict = erfl(latent_space, **kwargs) #Later: add axis also

        # Set the rotation parameters
        if self.rotation_optim_case == 'quaternion':
            rotation_params = self.rotation_initial_dict['quaternions']
        elif self.rotation_optim_case == 'basis':
            rotation_params = self.rotation_initial_dict['coeffs']
            self.basis = self.rotation_initial_dict['basis']
        else:
            raise ValueError("Invalid rotation optimization case. Must be 'quaternion' or 'basis'. as of now...")
        
        self.rotation_params = nn.Parameter(rotation_params.to(self.device))
        
        # Setting frames to the number of rotations
        self.frames = projections[:len(rotation_params)]

        @self.optimizer.params
        def params(self):
            return self.parameters()

    def train_vae(self, projections):
        self.vae.fit(projections)
        for param in self.vae.Parameters():
            param.requires_grad=False
        #for param in self.vae.fc_mu.Parameters():
        #    param.requires_grad=False

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
            self.volume = nn.Parameter(cloud.to(self.device))

        elif self.initial_volume == 'constant':
            self.volume = nn.Parameter(torch.ones(self.N, self.N, self.N, device=self.device))

        elif self.initial_volume == 'random':
            self.volume = nn.Parameter(torch.rand(self.N, self.N, self.N, device=self.device))

        elif self.initial_volume == 'given' and self.volume_init is not None:
            self.volume = nn.Parameter(self.volume_init.to(self.device))

        else:
            self.volume = nn.Parameter(torch.zeros(self.N, self.N, self.N, device=self.device))
    
    def forward(self, idx):
        """
        Forward pass of the model.
        """
        volume = self.volume
        quaternions = self.get_quaternions(self.rotation_params)[idx]

        # Normalize quaternions during computation
        quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)

        batch_size = quaternions.shape[0]
        estimated_projections = torch.zeros(batch_size, self.N, self.N, device=self.device)

        # Rotate the volume and estimate the projections
        for i in range(batch_size):
            rotated_volume = self.apply_rotation(volume, quaternions[i])
            estimated_projections[i] = self.imaging_model(rotated_volume)

        return estimated_projections
    
    def training_step(self, batch, batch_idx):
        #x,y = self.train_preprocess(batch) # batch = ground_truth projections

        yhat = self.forward(batch_idx) #Estimated projections

        latent_space = self.fc_mu(self.encoder(yhat))

        proj_loss, latent_loss = self.compute_loss(yhat, latent_space, batch, batch_idx)
        tot_loss = proj_loss + latent_loss

        loss = {"proj_loss": proj_loss, "latent_loss": latent_loss, "total_loss": tot_loss}
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
    
    def compute_loss(yhat, y):
        XXX
        return XXX

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
    

# Testing the code
if __name__ == "__main__":

    # Create a dummy dataset
    N = 96
    projections = torch.rand(10, N, N)

    # Create the tomography model
    tomography = Tomography(volume_size=(N, N, N))

    # Initialize the parameters
    tomography.initialize_parameters(projections)

    # Perform a forward pass
    tomography(0)