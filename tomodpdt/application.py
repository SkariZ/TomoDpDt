import deeplay as dl
import deeptrack as dt
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import estimate_rotations_from_latent as erfl

class Tomography(dl.Application):
    def __init__(self,
                 volume_size: Optional[Sequence(int)] = (96,96,96),
                 vae_model: Optional[torch.nn.Module]=None,
                 imaging_model: Optional[torch.nn.Module] = None,
                 initial_volume: Optional[str] = None, #Or provide initial guess for volume explicitly
                 optimizer=None,
                 volume_init=None,
                 **kwargs,



                 ):
        self.N=volume_size[0]
        self.vae_model=vae_model or dl.VariationalAutoEncoder(input_size=(self.volume_size[0],self.volume_size[1]))
        self.encoder=self.vae_model.encoder
        self.fc_mu=self.vae_model.fc_mu
        self.imaging_model=imaging_model or Projections
        self.device=self.vae_model.device
        self.initial_volume=initial_volume
        self.volume_init = volume_init
        super().__init__(**kwargs)

        self.optimizer = optimizer or Adam(lr=5e-3)
    def initialize_parameters(self,projections,**kwargs):
        latent_space = self.vae.fc_mu(self.vae.encoder(projections))
        self.latent = latent_space
        self.volume = self.initialize_volume()
        self.rotation_params = erfl(latent_space,**kwargs) #Later: add axis also
        
        # Setting frames to the number of quaternions
        self.frames = projections[:len(self.quaternions)]
        
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
            cloud = cloud.to(self.device)
            self.volume = nn.Parameter(cloud)

        elif self.initial_volume == 'constant':
            self.volume = nn.Parameter(torch.ones(self.N, self.N, self.N, device=self.device))

        elif self.initial_volume == 'random':
            self.volume = nn.Parameter(torch.rand(self.N, self.N, self.N, device=self.device))

        elif self.initial_volume == 'given' and self.volume_init is not None:
            self.volume = nn.Parameter(self.volume_init.to(self.device))

        else:
            self.volume = nn.Parameter(torch.zeros(self.N, self.N, self.N, device=self.device))
    
    def forward(self,idx):
        volume=self.volume
        rotations=self.get_quaternions(self.rotation_params)[idx]
        batch_size = rotations.shape[0]
        estimated_projections = torch.zeros(batch_size, self.N, self.N, device=self.device)
        for i in range(batch_size):
            rotated_volume=self.apply_rotation(volume,rotations[i])
            estimated_projections[i]=self.imaging_model(rotated_volume)
        return estimated_projections
    
    def training_step(self,batch,batch_idx):
        #x,y = self.train_preprocess(batch) # batch = ground_truth projections

        
        
        yhat=self.forward(batch_idx) #Estimated projections

        latent_space = self.fc_mu(self.encoder(yhat))

        proj_loss,latent_loss=self.compute_loss(yhat,latent_space,batch,batch_idx)
        tot_loss=proj_loss+latent_loss
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
    def compute_loss(yhat,y):
        XXX
        return XXX

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