import deeplay as dl
from deeplay.external import Adam

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from typing import Optional, Sequence
import time
import numpy as np

# Importing the necessary modules
try: 
    import tomodpdt.estimate_rotations_from_latent as erfl
    import tomodpdt.vaemod as vm
    import tomodpdt.plotting as tp

except:
    import estimate_rotations_from_latent as erfl
    import vaemod as vm
    import plotting as tp


class Sum3d2d(nn.Module):
    def __init__(self, dim=-1):
        self.dim = dim
        self.microscopy_regime = 'sum_projection'
        super(Sum3d2d, self).__init__()

    def forward(self, x):
        return x.sum(dim=self.dim, keepdim=True)


class Tomography(dl.Application):
    """
    Deep-learning-based 3D tomography reconstruction class.

    This class implements an end-to-end framework for reconstructing 3D volumes 
    from 2D projection images using a Variational Autoencoder (VAE) and 
    rotation/translation optimization. It supports both automatic and manual 
    optimization modes, flexible initialization schemes, and differentiable 
    geometric transformations.

    The model estimates latent representations of 2D projections, initializes
    rotation parameters via latent-space processing, and reconstructs the 3D
    volume by optimizing projection consistency, smoothness, and regularization losses.

    Parameters
    ----------
    volume_size : Sequence[int], optional
        The size of the 3D reconstruction volume (default: (96, 96, 96)).
    vae_model : torch.nn.Module, optional
        Variational Autoencoder model used for latent representation of projections.
        If None, a default `dl.VariationalAutoEncoder` is used.
    imaging_model : torch.nn.Module, optional
        Forward imaging model (e.g., projection operator). Defaults to `Sum3d2d`.
    initial_volume : str, optional
        Type of initial volume. Options: {'zeros', 'gaussian', 'refraction', 'random', 'given'}.
    rotation_optim_case : str, optional
        Rotation parameterization method. Options: {'quaternion', 'basis'}.
    optimizer : deeplay.external.Adam, optional
        Optimizer used for training. Defaults to Adam with learning rate 8e-3.
    volume_init : torch.Tensor, optional
        Custom tensor for initializing the volume if `initial_volume='given'`.
    minibatch : int, optional
        Batch size used during projection estimation (default: 16).
    loss_weights : dict, optional
        Dictionary of loss weights for automatic optimization mode.
    loss_weights_manual : dict, optional
        Dictionary of loss weights for manual optimization mode.
    learning_rate_volume : float, optional
        Learning rate for volume parameters (default: 8e-4).
    learning_rate_rotation : float, optional
        Learning rate for rotation parameters (default: 5e-4).
    learning_rate_translation : float, optional
        Learning rate for translation parameters (default: 1e-3).
    automatic_optimization : bool, optional
        If True, use automatic optimization (via PyTorch Lightning); otherwise manual updates.
    **kwargs : dict
        Additional keyword arguments for the parent `deeplay.Application` class.

    Attributes
    ----------
    volume : torch.nn.Parameter
        The reconstructed 3D volume.
    rotation_params : torch.nn.Parameter
        Learnable quaternion or basis rotation parameters.
    translation_params : torch.nn.Parameter
        Learnable translation parameters.
    vae_model : torch.nn.Module
        The VAE used to extract latent representations.
    imaging_model : torch.nn.Module
        Imaging model (projection operator).
    grid, grid_batch : torch.Tensor
        Normalized voxel-space grids used for transformations.

    Methods
    -------
    initialize_parameters(projections, **kwargs)
        Initializes model parameters (normalization, VAE training, rotations, volume, etc.).
    configure_optimizers()
        Sets up optimizers and schedulers for training.
    forward(idx)
        Performs forward projection of the reconstructed volume given indices.
    training_step(batch, batch_idx)
        Executes a single training step.
    initialize_volume()
        Initializes the volume parameter.
    initialize_translation(N)
        Initializes translation parameters.
    ...
        (See code for additional helper and utility functions.)

    Examples
    --------
    >>> tomo = Tomography(volume_size=(64, 64, 64))
    >>> tomo.initialize_parameters(projections)
    >>> loss = tomo.training_step(batch_indices, 0)
    """


    def __init__(self,
                 volume_size: Optional[Sequence[int]] = (96, 96, 96),
                 vae_model: Optional[torch.nn.Module] = None,
                 imaging_model: Optional[torch.nn.Module] = None,
                 initial_volume: Optional[str] = None,  # Initial guess for volume
                 rotation_optim_case: Optional[str] = None,  # Rotation optimization case ('quaternion', 'basis')
                 optimizer = None,
                 volume_init = None,  # Initial guess for volume explicitly
                 minibatch = 16,
                 loss_weights = None,
                 loss_weights_manual = None,
                 learning_rate_volume: float = 8e-4,
                 learning_rate_rotation: float = 5e-4,
                 learning_rate_translation: float = 1e-3,
                 automatic_optimization: bool = True,
                 **kwargs):
        
        # Set volume size and dimensions
        self.volume_size = volume_size
        self.nx, self.ny, self.nz = volume_size
        
        # If VAE model is not passed, initialize a default VAE model. This will be updated later if needed.
        self.vae_model = vae_model if vae_model is not None else dl.VariationalAutoEncoder(input_size=(self.volume_size[0], self.volume_size[1]), latent_dim=2)
        
        # Set the encoder and other VAE components
        self.encoder = self.vae_model.encoder
        self.fc_mu = self.vae_model.fc_mu
        
        # Set the imaging model (either passed as a module or projection function)
        self.imaging_model = imaging_model if imaging_model is not None else Sum3d2d(dim=-1)
        
        # Determine the device (cuda if available, else cpu)
        self._device = torch.device("cuda" if torch.cuda.is_available() else getattr(vae_model, "device", "cpu"))
        
        # Set initial volume if provided, otherwise default to "zeros"
        self.initial_volume = initial_volume if initial_volume is not None else "zeros"
        
        # Set the rotation optimization case, default to "quaternion"
        self.rotation_optim_case = rotation_optim_case if rotation_optim_case is not None else "quaternion"

        # Set volume initialization (if provided)
        self.volume_init = volume_init

        # Set the minibatch size - default to 16 - can speed up training
        self.minibatch = minibatch

        # Set the loss weights for standard optimization
        self.loss_weights = loss_weights if loss_weights is not None else {
            'proj_loss': 2.0,
            'latent_loss': 0.1,
            'rtv_loss': 7.0,
            'qv_loss': 0.2,
            'q0_loss': 0.2,
            'rtr_loss': 5.0,
            'so_loss': 100.0,
            'binarization_loss': 0.1  # Only used if microscopy_regime is fluorescence
            }
        
        # Set the loss weights for automatic optimization (only "projection", rtv and rtr losses)
        self.loss_weights_manual = loss_weights_manual if loss_weights_manual is not None else {
            'proj_loss': 1.0,
            'rtv_loss': 7.0,
            'rtr_loss': 5.0,
            }
        self.lr_volume_manual = 5.0
        self.lr_q_manual = 1.0
        self.lr_t_manual = 1e3

        # Raise error if loss weights don´t contain all the necessary keys
        if not all(k in self.loss_weights for k in ['proj_loss', 'latent_loss', 'rtv_loss', 'qv_loss', 'q0_loss', 'rtr_loss', 'so_loss']):
            raise ValueError("Loss weights must contain all the necessary keys.")
        
        if not all(k in self.loss_weights_manual for k in ['proj_loss', 'rtv_loss', 'rtr_loss']):
            raise ValueError("Loss weights must contain all the necessary keys.")

        # Set the learning rates for volume, rotation, and translation
        self.learning_rate_volume = learning_rate_volume
        self.learning_rate_rotation = learning_rate_rotation
        self.learning_rate_translation = learning_rate_translation

        # This is the optimizer for the variational autoencoder
        self.optimizer = optimizer if optimizer is not None else Adam(lr=8e-3)

        # Call the superclass constructor
        super().__init__(**kwargs)

        # Store normalized voxel-space grid for single volume
        lin_x = torch.linspace(-1, 1, self.nx, device=self._device)
        lin_y = torch.linspace(-1, 1, self.ny, device=self._device)
        lin_z = torch.linspace(-1, 1, self.nz, device=self._device)

        xx, yy, zz = torch.meshgrid(lin_x, lin_y, lin_z, indexing='ij')
        grid = torch.stack([xx, yy, zz], dim=-1)
        self.grid = grid.view(self.nx, self.ny, self.nz, 3)  # (nx, ny, nz, 3)

        # Store flat normalized grid for batch processing
        grid_batch = grid.view(-1, 3)  # Shape: (nx*ny*nz, 3)
        self.grid_batch = grid_batch

        # Move grids to the device
        self.grid = self.grid.to(self._device)
        self.grid_batch = self.grid_batch.to(self._device)

        # Placeholder
        self.normalize = False

        # Set automatic optimization flag
        self.automatic_optimization = automatic_optimization

        # Flag to enable/disable on_train_batch_end operations
        self.on_train_batch_end_enabled = kwargs.get('on_train_batch_end_enabled', False)
        self.on_train_epoch_end_enabled = kwargs.get('on_train_epoch_end_enabled', True)
        self.smooth_startup = kwargs.get('smooth_startup', True) if automatic_optimization else False
        self.smooth_startup_rotations = kwargs.get('smooth_startup_rotations', 100) if automatic_optimization else 0
        self.smooth_startup_translations = kwargs.get('smooth_startup_translations', 200) if automatic_optimization else 0

        # Flags to keep track of requires_grad status
        self.volume_flag = True
        self.rotation_params_flag = True
        self.translation_params_flag = True

        # Set binarization flag for fluorescence regime
        self.binarize_volume = True if self.imaging_model.microscopy_regime == "fluorescence" else False

        # Set rtv_loss, so_loss to 0 if fluorescence regime is used
        if self.binarize_volume:
            self.loss_weights['rtv_loss'] = 0.0 # no TV regularization in fluorescence
            self.loss_weights['so_loss'] = 0.0 # no strictly over loss in fluorescence
            self.loss_weights['proj_loss'] = 1e3
            self.loss_weights['latent_loss'] = 0.5
            self.learning_rate_volume = 1e-1

            self.smooth_startup_rotations = 400
            self.smooth_startup_translations = 800
            self.sigma_update = 1.04 # Fluorescence binarization sigma update factor

            x = torch.arange(self.nx) - self.nx / 2
            y = torch.arange(self.ny) - self.ny / 2
            z = torch.arange(self.nz) - self.nz / 2
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

            self.sigma = 0.4
            self.n_spots = 15#20

            self.mesh = [
                xx.to(self._device),
                yy.to(self._device),
                zz.to(self._device)
                ]


    def initialize_parameters(self, projections, **kwargs):
        """
        Initialize model parameters:
        - Normalize projections
        - Train VAE (with padded data if necessary)
        - Compute latent space and rotation initialization
        - Initialize volume and translation parameters
        """

        # -------------------------------------------------------
        # --- 0. Input preparation and safe copies
        # -------------------------------------------------------
        if not isinstance(projections, torch.Tensor):
            projections = torch.tensor(projections)

        # Move to device
        projections = projections.to(self._device)

        # Save a clone of the *original*, unpadded projections (on CPU for safety)
        projections_orig = projections.clone().detach().cpu()

        # Compute and store global min/max values for normalization
        self.compute_global_min_max(projections)

        # Number of channels
        self.CH = projections.shape[1]

        # -------------------------------------------------------
        # --- 1. Normalization and optional field correction
        # -------------------------------------------------------
        if kwargs.get('normalize', False):
            projections = self.per_channel_normalization(projections)
            self.normalize = True

        if kwargs.get('field_normalize', False):
            projections = self.correctfield(projections)

        # -------------------------------------------------------
        # --- 2. VAE setup (pad data to multiple of 8 if needed)
        # -------------------------------------------------------
        if kwargs.get('train_vae', True):

            if self.CH > 0 and min([self.nx, self.ny]) >= 24:
                _, C, H, W = projections.shape
                self.H_orig, self.W_orig = H, W

                # Crop projections to 96*96 if possible else, 64*64 if possible, else to 32*32
                crop_arg = kwargs.get('crop_projections', True)

                if H > 96 and W > 96 and crop_arg:
                    # Crop to 96*96 centered
                    start_h = (H - 96) // 2
                    start_w = (W - 96) // 2
                    projections = projections[:, :, start_h:start_h + 96, start_w:start_w + 96]
                    H, W = projections.shape[2], projections.shape[3]

                elif H > 64 and W > 64 and crop_arg:
                    # Crop to 64*64 centered
                    start_h = (H - 64) // 2
                    start_w = (W - 64) // 2
                    projections = projections[:, :, start_h:start_h + 64, start_w:start_w + 64]
                    H, W = projections.shape[2], projections.shape[3]

                elif H > 32 and W > 32 and crop_arg:
                    # Crop to 32*32 centered
                    start_h = (H - 32) // 2
                    start_w = (W - 32) // 2
                    projections = projections[:, :, start_h:start_h + 32, start_w:start_w + 32]
                    H, W = projections.shape[2], projections.shape[3]
                #else:
                #    raise ValueError("Projections are too small for VAE training. Minimum size is 32x32.")
                
                # The size
                self.H_vae, self.W_vae = H, W

                # Estimate beta
                if self.normalize:
                    self.vae_model.beta = 0.001 if 'fluorescence' in self.imaging_model.microscopy_regime else 0.025
                else:
                    r_loss = torch.mean(torch.abs(projections - torch.mean(projections)))
                    kl_loss = 0.5 * torch.mean(torch.sum(
                        1 + torch.zeros(C, 2) - torch.zeros(C, 2).exp() - torch.zeros(C, 2).pow(2), dim=1
                    ))
                    ratio = r_loss / (kl_loss + 1e-8)
                    self.vae_model.beta = 1e-5 if ratio > 1 else 1e-6

                # Keyword for beta override
                if 'vae_beta' in kwargs:
                    self.vae_model.beta = kwargs['vae_beta']

                # Build VAE to match padded dimensions
                vae = vm.ConvVAE(
                    input_shape=(self.CH, self.H_vae, self.W_vae),
                    latent_dim=2,
                    output_activation='sigmoid' if self.normalize else 'linear'
                )
                self.vae_model.encoder = vae.encoder
                self.vae_model.decoder = vae.decoder
                self.vae_model.fc_mu = vae.fc_mu
                self.vae_model.fc_var = vae.fc_var
                self.vae_model.fc_dec = vae.fc_dec
                if not self.normalize:
                    self.vae_model.reconstruction_loss = torch.nn.L1Loss()
            else:
                # No padding needed
                self.H_vae, self.W_vae = projections.shape[2:]


        # -------------------------------------------------------
        # --- 3. Train VAE (only on padded data)
        # -------------------------------------------------------
        vae_success = False
        vae_attempts = 0
        max_vae_attempts = 3

        while not vae_success and vae_attempts < max_vae_attempts:
            vae_attempts += 1
            try:
                if self.vae_model.training:
                    self.train_vae(projections, **kwargs)

                # -------------------------------------------------------
                # --- 4. Latent space & rotation initialization
                # -------------------------------------------------------
                latent_space = self.vae_model.fc_mu(self.vae_model.encoder(projections))
                self.latent = latent_space

                # Try latent-space initialization
                self.rotation_initial_dict = erfl.process_latent_space(
                    z=latent_space,
                    frames=projections,
                    **kwargs
                )
                print("✅ Rotation initialization from latent space successful.")
                vae_success = True
                break  # stop loop once successful

            except Exception as e:
                print(f"VAE attempt {vae_attempts} failed: {e}")
                if vae_attempts < max_vae_attempts:
                    time.sleep(2)  # wait 2 seconds before retrying

                    # Build VAE to match padded dimensions
                    vae = vm.ConvVAE(
                        input_shape=(self.CH, self.H_vae, self.W_vae),
                        latent_dim=2,
                        output_activation='sigmoid' if self.normalize else 'linear'
                    )
                    self.vae_model.encoder = vae.encoder
                    self.vae_model.decoder = vae.decoder
                    self.vae_model.fc_mu = vae.fc_mu
                    self.vae_model.fc_var = vae.fc_var
                    self.vae_model.fc_dec = vae.fc_dec
                    if not self.normalize:
                        self.vae_model.reconstruction_loss = torch.nn.L1Loss()

                else:
                    print("VAE repeatedly failed — switching to cross-correlation initialization.")
                    self.rotation_initial_dict = erfl.process_cross_correlation(
                        frames=projections,
                        **kwargs
                    )
                    print("✅ Rotation initialization via cross-correlation successful.")

        # -------------------------------------------------------
        # --- 5. Set rotation parameters
        # -------------------------------------------------------
        try:
            if self.rotation_optim_case == 'quaternion':
                rotation_params = self.rotation_initial_dict['quaternions']
            elif self.rotation_optim_case == 'basis':
                rotation_params = self.rotation_initial_dict['coeffs']
                self.basis = self.rotation_initial_dict['basis']
            else:
                raise ValueError("Invalid rotation optimization case. Must be 'quaternion' or 'basis'.")

            self.rotation_params = nn.Parameter(rotation_params.to(self._device))

            # Determine number of frames needed for optimization
            N_frames_needed = self.rotation_initial_dict["peaks"][-1].item()

        except Exception as e:
            N_frames_needed = projections.shape[0]
            self.rotation_params = nn.Parameter(torch.zeros(N_frames_needed, 4 if self.rotation_optim_case == 'quaternion' else 3, device=self._device))

            # Create a dummy latent space with zeros
            self.latent = torch.zeros(N_frames_needed, 2, device=self._device)

        # Throw error if automatic_optimizations is True but VAE training is skipped
        if self.automatic_optimization and not self.vae_model.training:
            raise RuntimeError("VAE training is required for automatic optimization.")

        # -------------------------------------------------------
        # --- 6. Initialize volume
        # -------------------------------------------------------
        self.initialize_volume()

        # -------------------------------------------------------
        # --- 7. Initialize translation parameters as zeros
        # -------------------------------------------------------
        self.initialize_translation(N_frames_needed)
        
        # -------------------------------------------------------
        # --- 8. Restore self.frames to ORIGINAL (unpadded) shape
        # -------------------------------------------------------
        self.frames = projections_orig[:N_frames_needed].to(self._device)
        
        # -------------------------------------------------------
        # --- 9. Optional optimizer registration
        # -------------------------------------------------------
        @self.optimizer.params
        def params(self):
            return self.parameters()

        # -------------------------------------------------------
        # --- 10. Background correction initialization
        # -------------------------------------------------------
        #self.V0 = self.imaging_model(self.volume * 0).detach()
        #if self.V0.dtype == torch.complex64:
        #    self.V0_phase = torch.median(torch.angle(self.V0))
        #    self.V0 = self.V0 * torch.exp(-1j * self.V0_phase)

    def configure_optimizers(self):
        param_groups = []

        # --- Volume parameters ---
        if getattr(self, "binarize_volume", False):
            param_groups.append({
                "params": self.mus,
                "lr": self.learning_rate_volume
            })
        else:
            if hasattr(self, "volume") and self.volume.requires_grad:
                param_groups.append({'params': [self.volume], 'lr': self.learning_rate_volume})

        # --- Rotation parameters ---
        if hasattr(self, "rotation_params") and self.rotation_params.requires_grad:
            param_groups.append({'params': [self.rotation_params], 'lr': self.learning_rate_rotation})

        # --- Translation parameters ---
        if hasattr(self, "translation_params") and self.translation_params.requires_grad:
            param_groups.append({'params': [self.translation_params], 'lr': self.learning_rate_translation})

        # --- Sanity check ---
        if not param_groups:
            raise ValueError("No parameters to optimize. Check requires_grad flags.")

        optimizer = torch.optim.Adam(param_groups)

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.75, patience=20, threshold=1e-3, min_lr=5e-7
            ),
            'monitor': 'train_total_loss',
        }

        return [optimizer], [scheduler]

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
    
    def correctfield(self, field, n_iter=5):
        """
        Correct field to have a mean phase of 0 and a mean absolute value of 1.
        """

        if field.dtype == torch.float32:
            field = field.to(torch.complex64)

        f_new = field.clone()

        # Normalize with mean of absolute value.
        f_new = f_new / torch.mean(torch.abs(f_new))

        for _ in range(n_iter):
            f_new = f_new * torch.exp(-1j * torch.median(torch.angle(f_new)))

        return f_new

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
        Initialize the volume with shape (nx, ny, nz) from self.volume_size.
        """
        nx, ny, nz = self.volume_size  # get actual volume dimensions

        #if self.binarize_volume:
            #self.volume = nn.Parameter(torch.rand(nx, ny, nz, device=self._device) * 1e-1)  # small random initialization
        
        if self.binarize_volume and self.initial_volume != 'given':
            # Create 3D coordinate grids
            x = torch.arange(nx, device=self._device) - nx / 2
            y = torch.arange(ny, device=self._device) - ny / 2
            z = torch.arange(nz, device=self._device) - nz / 2
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

            # Initialize Gaussian centers (mus)
            mus = min(nx, ny, nz) * (torch.rand(self.n_spots, 3, device=self._device) - 0.5)
            self.mus = [
                nn.Parameter(mus[:, 0]),
                nn.Parameter(mus[:, 1]),
                nn.Parameter(mus[:, 2]),
            ]

            # Compute Gaussian cloud volume
            dx = xx[None] - self.mus[0][:, None, None, None]
            dy = yy[None] - self.mus[1][:, None, None, None]
            dz = zz[None] - self.mus[2][:, None, None, None]

            cloud = torch.sum(
                torch.exp(-self.sigma * (dx**2 + dy**2 + dz**2)),
                dim=0
            )

            # Normalize to 0 to 1
            cloud = cloud / cloud.max()
            cloud = torch.clamp(cloud, 0, 0.1)

            self.volume = cloud

        elif self.initial_volume == 'gaussian':
            x = torch.arange(nx) - nx / 2
            y = torch.arange(ny) - ny / 2
            z = torch.arange(nz) - nz / 2
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
            cloud = torch.exp(-0.001 * (xx**2 + yy**2 + zz**2))
            cloud = cloud / cloud.max()
            cloud = torch.clamp(cloud, 0, 0.1)
            self.volume = nn.Parameter(cloud.to(self._device))

        elif self.initial_volume == 'zeros':
            self.volume = nn.Parameter(torch.zeros(nx, ny, nz, device=self._device))

        elif self.initial_volume == 'refraction':
            self.volume = nn.Parameter(torch.ones(nx, ny, nz, device=self._device) * 1.33)

        elif self.initial_volume == 'random':
            self.volume = nn.Parameter(torch.rand(nx, ny, nz, device=self._device))

        elif self.initial_volume == 'given' and self.volume_init is not None:
            self.volume = nn.Parameter(self.volume_init.to(self._device))

        # Override with given volume if specified
        if self.initial_volume == 'given' and self.volume_init is not None:
            self.volume = nn.Parameter(self.volume_init.to(self._device))

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

        # --- 1. Fetch parameters ---
        quaternions = self.get_quaternions(self.rotation_params)[idx]
        translations = self.get_translations(self.translation_params)[idx]

        # --- 2. Retrieve and condition the volume ---
        volume = self.get_volume()

        batch_size = quaternions.shape[0]
        estimated_projections_batch = torch.zeros(
            batch_size, self.CH, self.nx, self.ny, device=self._device
        )

        # --- 3. Minibatching setup ---
        if batch_size < self.minibatch:
            self.minibatch = batch_size

        indexes = torch.arange(0, batch_size)
        b_idx = [indexes[i:i + self.minibatch] for i in range(0, len(indexes), self.minibatch)]

        # --- 4. Loop through mini-batches ---
        for b in b_idx:
            # Apply rotations to a single volume using a batch of quaternions
            rotated_volumes = self.apply_rotation_batch(
                volume=volume, 
                quaternions=quaternions[b], 
                translations=translations[b] if translations is not None else None
            )
            
            # --- 5. Forward through the imaging model ---
            if isinstance(self.imaging_model, nn.Module):
                estimated_projections = self.imaging_model(rotated_volumes)

                # Handle single-channel (e.g., brightfield or fluorescence intensity)
                if self.CH == 1:
                    if estimated_projections.dtype == torch.complex64:
                        estimated_projections = estimated_projections.imag
                
                # Handle two-channel complex projections
                elif self.CH > 1 and estimated_projections.dtype == torch.complex64:
                    #estimated_projections = estimated_projections * torch.exp(-1j * self.V0_phase)
                    #estimated_projections = estimated_projections - self.V0 + 1
                    estimated_projections = torch.cat(
                        (estimated_projections.real, estimated_projections.imag), dim=-1
                    )

                # Ensure (B, C, H, W) layout
                if estimated_projections.dim() == 4 and estimated_projections.shape[1] != self.CH:
                    estimated_projections = estimated_projections.permute(0, 3, 1, 2)
                if estimated_projections.dim() == 3 and estimated_projections.shape[1] != self.CH:
                    estimated_projections = estimated_projections.unsqueeze(1)

                # Store the batch projections
                estimated_projections_batch[b] = estimated_projections
            
            else:
                raise ValueError("Imaging model must be a nn.Module.")

        return estimated_projections_batch

    def training_step(self, batch, batch_idx):
        """
        Training step for the model. Computes the loss and logs it.
        """

        # Get indices and corresponding frames
        idx_batch = batch
        frames_batch = self.frames[idx_batch]

        # Safely unpad to original size.
        if hasattr(self, 'H_orig') and hasattr(self, 'W_orig'):
            if frames_batch.shape[2:] != (self.H_orig, self.W_orig):
                frames_batch = self.unpad_to_original(frames_batch)
        
        if self.smooth_startup:
            # If global_step is below 100 set the rotation_params to not require gradients
            if self.global_step < self.smooth_startup_rotations and self.rotation_params.requires_grad == True and self.rotation_params_flag == True:
                self.rotation_params.requires_grad = False
            elif self.global_step >= self.smooth_startup_rotations and self.rotation_params.requires_grad == False and self.rotation_params_flag == True:
                self.rotation_params.requires_grad = True

            # If global_step is below 200 set the translation_params to not require gradients
            if self.global_step < self.smooth_startup_translations and self.translation_params.requires_grad == True and self.translation_params_flag == True:
                self.translation_params.requires_grad = False
            elif self.global_step >= self.smooth_startup_translations and self.translation_params.requires_grad == False and self.translation_params_flag == True:
                self.translation_params.requires_grad = True

        # Forward pass: estimate projections
        yhat = self.forward(idx_batch)

        # Normalize predictions if required
        if self.normalize:
            yhat = self.per_channel_normalization(yhat)

        if self.automatic_optimization == True:

            # Prepare VAE input (pad yhat if necessary)
            yhat_vae = self.pad_for_vae(yhat, self.H_vae, self.W_vae)
        
            # Compute latent space from VAE
            with torch.no_grad():
                latent_space = self.fc_mu(self.encoder(yhat_vae))

            # Compute all losses
            proj_loss, latent_loss, rtv_loss, qv_loss, q0_loss, rtr_loss, so_loss = self.compute_loss_old(
                yhat, latent_space, frames_batch, idx_batch, self.loss_weights
            )

            # Total loss
            tot_loss = proj_loss + latent_loss + rtv_loss + qv_loss + q0_loss + rtr_loss + so_loss

            # Log losses
            loss_dict = {
                "total_loss": tot_loss,
                "proj_loss": proj_loss,
                "latent_loss": latent_loss,
                "rtv_loss": rtv_loss,
                "rtr_loss": rtr_loss,
                "qv_loss": qv_loss,
                "q0_loss": q0_loss,
                "so_loss": so_loss,
            }

            # Binarization loss for fluorescence regime
            if self.imaging_model.microscopy_regime == 'fluorescence':

                # Update sigma every 50 steps
                if self.global_step % 50 == 0 and self.global_step > 0 and self.sigma < 1.25:
                    self.sigma = self.sigma * self.sigma_update

                # Recompute projection loss for fluorescence regime
                tot_loss = tot_loss - proj_loss  # Remove projection loss
                proj_loss_fluorescence = self.projection_loss_fluorescence(yhat, frames_batch) * self.loss_weights['proj_loss']
                tot_loss += proj_loss_fluorescence

                # Remove old proj_loss and add new one
                del loss_dict["proj_loss"]
                loss_dict["proj_loss"] = proj_loss_fluorescence

                volume = self.get_volume()
                binarization_loss = self.compute_binarization_loss(volume) * self.loss_weights['binarization_loss']
                tot_loss += binarization_loss
                loss_dict["binarization_loss"] = binarization_loss

                # Compute RTV loss for fluorescence regime
                rtv_loss_fluorescence = self.rtv_loss_fluorescence(volume) * 1e-4
                tot_loss += rtv_loss_fluorescence
                loss_dict["rtv_loss_fluorescence"] = rtv_loss_fluorescence

                # Update total loss in the dictionary
                del loss_dict["total_loss"]
                loss_dict["total_loss"] = tot_loss

            # Only keep non-zero losses
            loss_dict = {k: v for k, v in loss_dict.items() if v.item() > 0}

            # Log all losses
            for name, value in loss_dict.items():
                self.log(
                    f"train_{name}",
                    value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
            return tot_loss
        
        # Manual optimization
        elif self.automatic_optimization == False:
            
            # Compute all losses
            proj_loss, rtv_loss, rtr_loss = self.compute_loss(
                yhat, frames_batch, self.loss_weights_manual
            )

            # Total loss
            tot_loss = proj_loss + rtv_loss + rtr_loss

            # Log losses
            loss_dict = {
                "total_loss": tot_loss,
                "proj_loss": proj_loss,
                "rtv_loss": rtv_loss,
                "rtr_loss": rtr_loss,
            }

            # Only keep non-zero losses
            loss_dict = {k: v for k, v in loss_dict.items() if v.item() > 0}

            for name, value in loss_dict.items():
                self.log(
                    f"train_{name}",
                    value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
            
            # Backward
            tot_loss.backward()

            # --- Manual parameter updates ---
            with torch.no_grad():
                # dRI update (clipped)
                self.volume -= self.lr_volume_manual * self.volume.grad
                self.volume.clamp_(0., 1.)

                if self.rotation_params.requires_grad == True:
                    # Q update (renormalize quaternions)
                    if self.rotation_optim_case == 'quaternion':
                        self.rotation_params -= self.lr_q_manual * self.rotation_params.grad
                        self.rotation_params[:] = self.rotation_params / self.rotation_params.norm(dim=1, keepdim=True)

                    # Q update (basis)
                    elif self.rotation_optim_case == 'basis':
                        self.rotation_params -= self.lr_q_manual * self.rotation_params.grad
                        # Recompute quaternions from basis
                        quaternions = self.get_quaternions(self.rotation_params)
                        quaternions_norm = quaternions / quaternions.norm(dim=1, keepdim=True)

                        # Generate the basis functions. Solve the least squares problem to find the coefficients
                        coeffs = torch.linalg.lstsq(self.basis, quaternions_norm).solution
                        # Update the rotation parameters with the new coefficients
                        self.rotation_params.data = coeffs

                if self.translation_params.requires_grad == True:
                    # T update (translations)
                    self.translation_params -= self.lr_t_manual * self.translation_params.grad

                # Zero gradients
                self.volume.grad = None
                self.rotation_params.grad = None
                self.translation_params.grad = None

                return tot_loss
        else:
            raise ValueError("Invalid automatic_optimization value. Must be True or False.")

    def pad_for_vae(self, tensor, H_padded, W_padded):
        """
        Pads a tensor to the desired VAE input size.
        Args:
            tensor (torch.Tensor): Shape (B, C, H, W)
            H_padded (int): Desired height
            W_padded (int): Desired width
        Returns:
            torch.Tensor: Padded tensor
        """
        H, W = tensor.shape[2], tensor.shape[3]
        pad_h = H_padded - H
        pad_w = W_padded - W
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    def unpad_to_original(self, tensor):
        """Crop tensor symmetrically back to original (unpadded) size."""
        _, _, H, W = tensor.shape
        pad_h = (H - self.H_orig)
        pad_w = (W - self.W_orig)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return tensor[:, :, pad_top:H - pad_bottom, pad_left:W - pad_right]

    def projection_loss(self, yhat, frames_batch):
        """
        Compute the projection loss using Mean Squared Error (MSE).
        """

        # If more than one channel is present make it a complex valued projection
        if self.CH == 2 and yhat.dtype == torch.float32:
            yhat = yhat[:, 0] + 1j * yhat[:, 1]
            frames_batch = frames_batch[:, 0] + 1j * frames_batch[:, 1]

        return torch.square(torch.abs(yhat - frames_batch)).mean()

    def projection_loss_fluorescence(self, yhat, frames_batch):
        """
        Compute the projection loss using Mean Absolute Error (MAE) for fluorescence regime.
        """

        f_norm = frames_batch / torch.std(frames_batch, dim=(1,2,3), keepdim=True)
        y_norm = yhat / torch.std(yhat, dim=(1,2,3), keepdim=True)

        return torch.abs(f_norm-y_norm).mean()
    
    def rtv_loss_fluorescence(self, volume):
        """
        Compute the RTV loss for fluorescence regime.
        """

        return torch.abs(torch.sum(volume)-torch.sum(volume**2)).mean()
    
    def compute_loss(self, yhat, frames_batch, loss_weights=None):
        """
        Compute the projection loss, latent loss, and all regularization terms.
        Uses MSE instead of L1 for projections and latent space.
        Rotation/translation trajectory regularization is computed on the full trajectory.
        """

        # === Projection loss (MSE) ===
        proj_loss = self.projection_loss(yhat, frames_batch)

        # === TV regularization on volume ===
        if self.volume.requires_grad:
            rtv_loss = self.total_variation_regularization(self.volume)
        else:
            rtv_loss = torch.tensor(0.0, device=self._device)

        # === Rotational trajectory regularization (computed on full trajectory, not batch only) ===
        if self.rotation_params.requires_grad and self.rotation_optim_case == 'quaternion':
            quaternions_full = self.get_quaternions(self.rotation_params)   # full trajectory
            rtr_loss = self.rotational_trajectory_regularization(quaternions_full, loss_weights['rtr_loss'], loss_weights['rtr_loss'])
        else:
            rtr_loss = torch.tensor(0.0, device=self._device)

        # === Scale the losses ===
        if loss_weights is not None and isinstance(loss_weights, dict):
            proj_loss *= loss_weights['proj_loss']
            rtv_loss *= loss_weights['rtv_loss']
            rtr_loss *= 1.0


        return proj_loss, rtv_loss, rtr_loss

    def compute_loss_old(self, yhat, latent_space, frames_batch, idx_batch, loss_weights=None):
        """
        Compute the projection loss, latent loss, and all regularization terms.
        Uses MSE instead of L1 for projections and latent space.
        Rotation/translation trajectory regularization is computed on the full trajectory.
        """

        # === Projection loss (MSE) ===
        proj_loss = self.projection_loss(yhat, frames_batch)

        # === Latent loss (MSE) ===
        if self.loss_weights['latent_loss'] > 0:
            latent_loss = F.mse_loss(latent_space, self.latent[idx_batch])

        # === TV regularization on volume ===
        if self.volume.requires_grad and self.loss_weights['rtv_loss'] > 0:
            rtv_loss = self.total_variation_regularization(self.volume)
        else:
            rtv_loss = torch.tensor(0.0, device=self._device)

        # === Quaternion validity loss ===
        if self.rotation_params.requires_grad and self.loss_weights['qv_loss'] > 0:
            quaternions_pred = self.get_quaternions(self.rotation_params)[idx_batch]
            qv_loss = self.quaternion_validity_loss(quaternions_pred)
        else:
            qv_loss = torch.tensor(0.0, device=self._device)

        # === q0 constraint loss (only if idx_batch contains 0) ===
        if self.rotation_params.requires_grad and torch.sum(idx_batch == 0) > 0 and self.loss_weights['q0_loss'] > 0:
            q0_loss = self.q0_constraint_loss(
                quaternions_pred[idx_batch == 0]
            )
        else:
            q0_loss = torch.tensor(0.0, device=self._device)

        # === Rotational trajectory regularization (computed on full trajectory, not batch only) ===
        if self.rotation_params.requires_grad and self.rotation_optim_case == 'quaternion' and self.loss_weights['rtr_loss'] > 0:
            quaternions_full = self.get_quaternions(self.rotation_params)   # full trajectory
            rtr_loss = self.rotational_trajectory_regularization(quaternions_full)
        else:
            rtr_loss = torch.tensor(0.0, device=self._device)

        # === Strictly over loss on volume ===
        if self.volume.requires_grad and self.loss_weights['so_loss'] > 0:
            so_loss = self.strictly_over_loss(self.volume, value=0)
        else:
            so_loss = torch.tensor(0.0, device=self._device)

        # === Scale the losses ===
        if loss_weights is not None and isinstance(loss_weights, dict):
            proj_loss *= self.loss_weights['proj_loss']
            latent_loss *= self.loss_weights['latent_loss']
            rtv_loss *= self.loss_weights['rtv_loss']
            qv_loss *= self.loss_weights['qv_loss']
            q0_loss *= self.loss_weights['q0_loss']
            rtr_loss *= self.loss_weights['rtr_loss']
            so_loss *= self.loss_weights['so_loss']

        return proj_loss, latent_loss, rtv_loss, qv_loss, q0_loss, rtr_loss, so_loss

    def strictly_over_loss(self, volume, value=1.33):
        """
        Computes a loss that penalizes values strictly below a value
        """
        loss = torch.sum(torch.relu(value - volume))  # Penalize values below 
        return loss / volume.numel()  # Normalize by total elements

    def total_variation_regularization(self, delta_n, beta=1e-10):
        """
        Calculate the total variation regularization term in 3D without creating large intermediate tensors.

        Args:
        - delta_n (torch.Tensor): A tensor of shape (D, H, W) or higher dimensional array.

        Returns:
        - R_TV (float): The total variation regularization term.
        """
        # Compute gradients and sum them inline to avoid intermediate tensors
        # grad_x_sum = torch.sum(torch.abs(delta_n[1:, :, :] - delta_n[:-1, :, :]))  # Gradient in x-direction
        # grad_y_sum = torch.sum(torch.abs(delta_n[:, 1:, :] - delta_n[:, :-1, :]))  # Gradient in y-direction
        # grad_z_sum = torch.sum(torch.abs(delta_n[:, :, 1:] - delta_n[:, :, :-1]))  # Gradient in z-direction

        grad_x = torch.diff(delta_n, dim=0, append=delta_n[-1, None])
        grad_y = torch.diff(delta_n, dim=1, append=delta_n[:, -1, None])
        grad_z = torch.diff(delta_n, dim=2, append=delta_n[:, :, -1, None])

        # Combine all gradient sums
        R_TV = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + beta).sum() / delta_n.numel()#(grad_x_sum + grad_y_sum + grad_z_sum) / delta_n.numel()

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

    def rotational_trajectory_regularization(self, q, λ1=1.0, λ2=1.0):
        """
        Faster rotational trajectory regularization term.
        
        Args:
        - q (torch.Tensor): Tensor of shape (T, d)
        
        Returns:
        - reg_terms (float): Regularization value
        """
        qd = torch.diff(q, axis=0)
        qdd = torch.diff(qd, axis=0)
        
        return (λ1 * torch.sum(torch.square(qd)) + λ2 * torch.sum(torch.square(qdd))) / q.shape[0]

    def compute_binarization_loss(self, volume, λ_bin=1.0):
        return λ_bin * torch.mean(volume * (1 - volume))

    def get_volume(self):
        """
        Get the volume from the volume parameters."""

        if hasattr(self, "binarize_volume") and self.binarize_volume:

            xx = self.mesh[0]
            yy = self.mesh[1]
            zz = self.mesh[2]

            dx = xx[None] - self.mus[0][:, None, None, None]
            dy = yy[None] - self.mus[1][:, None, None, None]
            dz = zz[None] - self.mus[2][:, None, None, None]

            volume = torch.sum(torch.exp(- self.sigma * (dx**2 + dy**2 + dz**2)), dim=0)

            # Clamp volume to [0, 1]
            #volume = torch.clamp(volume, 0, 1)
            # Scale volume to [0, 1]
            volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-6)

            # Normalize volume to have a fixed total sum
            # volume = volume / torch.sum(volume) #* self.n_spots / 2.0

            return volume
        else:
            return self.volume

    def get_translations(self, raw_translation):

        """
        Get translations from the translation parameters."""

        if raw_translation is None:
            return None
        else:
            return raw_translation

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

        q = q / (q.norm())  # Normalize quaternion

        # Compute the elements of the rotation matrix directly from quaternion components
        R = torch.stack([
            torch.stack([1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)], dim=-1),
            torch.stack([2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)], dim=-1),
            torch.stack([2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)], dim=-1),
        ], dim=-2)

        return R

    def quaternion_to_rotation_matrix_batch(self, q):
        """Convert a batch of quaternions (B, 4) to rotation matrices (B, 3, 3)."""

        q = q / (q.norm(dim=1, keepdim=True))  # Normalize quaternions batchwise
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = torch.stack([
            1 - 2*y**2 - 2*z**2,  2*x*y - 2*z*w,  2*x*z + 2*y*w,
            2*x*y + 2*z*w,  1 - 2*x**2 - 2*z**2,  2*y*z - 2*x*w,
            2*x*z - 2*y*w,  2*y*z + 2*x*w,  1 - 2*x**2 - 2*y**2
        ], dim=1).reshape(-1, 3, 3)  # Shape: (B, 3, 3)

        return R

    def apply_rotation(self, volume, q, translations=None):

        R = self.quaternion_to_rotation_matrix(q)  # (3,3)

        # Prepare volume: add batch and channel dims if needed
        if volume.dim() == 3:
            volume = volume.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

        # Get volume shape
        _, _, D, H, W = volume.shape

        # Flatten normalized voxel-space grid
        grid = self.grid.view(-1, 3)  # (N^3, 3)

        # If translation provided, normalize and subtract
        if translations is not None:
            t_norm = torch.zeros(3, device=grid.device)
            t_norm[0] = 2 * translations[0] / (D - 1)  # dz normalized
            t_norm[1] = 2 * translations[1] / (H - 1)  # dy normalized
            t_norm[2] = 2 * translations[2] / (W - 1)  # dx normalized

            grid -= t_norm.view(1, 3)

        # Rotate grid by R
        rotated_grid = torch.matmul(grid, R.t())  # (N^3, 3)

        # Reshape and clamp
        rotated_grid = rotated_grid.view(1, D, H, W, 3).clamp(-1, 1)

        rotated_volume = F.grid_sample(volume, rotated_grid, align_corners=True)

        return rotated_volume.squeeze(0).squeeze(0)

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
        volume = volume.unsqueeze(0)  # (1, 1, D, H, W)
        _, _, D, H, W = volume.shape

        B = quaternions.shape[0]

        # Repeat the single volume B times
        volumes = volume.expand(B, -1, -1, -1, -1)  # (B, 1, D, H, W)

        # Get rotation matrices batch: (B, 3, 3)
        R = self.quaternion_to_rotation_matrix_batch(quaternions)

        # Prepare and expand normalized flat grid on correct device: (N³, 3)
        grid = self.grid_batch.to(volume.device)  # (N³, 3)
        grid = grid.unsqueeze(0).expand(B, -1, -1).clone()  # (B, N³, 3)

        # Apply translation if provided (convert voxel units to normalized coords)
        if translations is not None:
            t_norm = translations.clone()
            # Normalize each translation component to [-1, 1]
            t_norm[:, 0] = 2 * translations[:, 0] / (D - 1)  # dz → z axis
            t_norm[:, 1] = 2 * translations[:, 1] / (H - 1)  # dy → y axis
            t_norm[:, 2] = 2 * translations[:, 2] / (W - 1)  # dx → x axis
            grid -= t_norm[:, None, :]  # Broadcast over all points

        # Rotate the grid points by batch rotation matrices: (B, N³, 3)
        rotated_grid = torch.bmm(grid, R.transpose(1, 2))

        # Reshape rotated grid back to (B, D, H, W, 3) for grid_sample
        rotated_grid = rotated_grid.view(B, D, H, W, 3).clamp(-1, 1)

        # Use grid_sample to sample volumes at rotated (and translated) coordinates
        transformed = F.grid_sample(volumes, rotated_grid, align_corners=True)

        # Return rotated volumes as (B, D, H, W)
        return transformed.squeeze(1)

    def full_forward_final(self, max_projections=None, rand_idx=False, idx=None):
        """
        Forward pass of the model.

        Args:
        - volume (torch.Tensor): The volume to rotate.
        - quaternions (torch.Tensor): Quaternions representing rotations.

        Returns:
        - estimated_projections (torch.Tensor): Estimated projections.
        """

        # Determine number of projections
        N_f = self.frames.shape[0]

        # Determine indexes to process
        if idx is not None:
            if isinstance(idx, (list, np.ndarray)):
                indexes = torch.tensor(idx, device=self._device, dtype=torch.long)

            elif isinstance(idx, torch.Tensor):
                indexes = idx.to(self._device)

        elif rand_idx:
            indexes = torch.arange(0, N_f, device=self._device)
            indexes = indexes[torch.randperm(indexes.shape[0])]

        else:
            indexes = torch.arange(0, N_f, device=self._device)

        if max_projections is not None and N_f > max_projections:
            indexes = indexes[:max_projections]

        # Safe check so that indexes are within bounds
        indexes = indexes[indexes < N_f]

        # Forward pass to get estimated projections
        estimated_projections = self.forward(
            idx=indexes
            )
    
        # Normalize the estimated projections
        if self.normalize:
            estimated_projections = self.per_channel_normalization(
                estimated_projections
                )

        return estimated_projections
    
    def get_quaternions_final(self, rotations=None):
        """
        Get quaternions from the rotation parameters.
        """

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
        Get translations from the translation parameters.
        """

        return self.get_translations(raw_translation if raw_translation is not None else self.translation_params)

    def move_all_to_device(self, device):
        """
        Safely move all model components (parameters, buffers, and extra tensors)
        to the specified device.

        Handles both brightfield (continuous) and fluorescence (binarized) modes.
        """
        # --- Move all registered parameters & buffers automatically ---
        super().to(device)

        # --- Fluorescence regime (binarized volume via logits) ---
        if getattr(self, "imaging_model", None) is not None:

            # Continuous volume parameter (e.g., refractive index or absorption)
            if hasattr(self, "volume") and isinstance(self.volume, torch.nn.Parameter):
                # Already moved by self.to(device), no need to reassign
                pass
            elif hasattr(self, "volume"):
                self.volume = self.volume.to(device)

        # --- Rotation & translation parameters ---
        # Only move manually if they are not Parameters (already handled by .to)
        if hasattr(self, "rotation_params") and not isinstance(self.rotation_params, torch.nn.Parameter):
            self.rotation_params = self.rotation_params.to(device)
        if hasattr(self, "translation_params") and not isinstance(self.translation_params, torch.nn.Parameter):
            self.translation_params = self.translation_params.to(device)

        # --- Basis functions (not parameters) ---
        if getattr(self, "rotation_optim_case", None) == "basis":
            if hasattr(self, "basis"):
                self.basis = self.basis.to(device)

        # --- Grids & normalization constants ---
        for attr in ["grid", "grid_batch", "global_min", "global_max"]:
            if hasattr(self, attr):
                tensor = getattr(self, attr)
                if isinstance(tensor, torch.Tensor):
                    setattr(self, attr, tensor.to(device))

    def toggle_grad(self, requires_grad: bool):
        """
        Toggle requires_grad for all model parameters.
        """
        for param in self.parameters():
            param.requires_grad = requires_grad

    def toggle_gradients_rotation_translation(self, requires_grad: bool):
        """
        Toggle gradients for both rotation and translation parameters.
        """
        if hasattr(self, "rotation_params"):
            self.rotation_params.requires_grad = requires_grad
            self.rotation_params_flag = requires_grad
        if hasattr(self, "translation_params"):
            self.translation_params.requires_grad = requires_grad
            self.translation_params_flag = requires_grad

    def toggle_gradients_quaternion(self, requires_grad: bool):
        """
        Toggle gradients for quaternion (rotation) parameters only.
        """
        if hasattr(self, "rotation_params"):
            self.rotation_params.requires_grad = requires_grad
            self.rotation_params_flag = requires_grad

    def toggle_gradients_translation(self, requires_grad: bool):
        """
        Toggle gradients for translation parameters only.
        """
        if hasattr(self, "translation_params"):
            self.translation_params.requires_grad = requires_grad
            self.translation_params_flag = requires_grad

    def toggle_gradients_volume(self, requires_grad: bool):
        """
        Toggle gradients for the volume (or logits, for fluorescence).
        """

        if hasattr(self, "volume"):
            self.volume.requires_grad = requires_grad
            self.volume_flag = requires_grad

    def swap_rotation_axis(self):
        """ 
        Swap the rotation axis. Between x and y rotation. 
        """
        # Swap the x and y rotation
        self.rotation_params[:, [1, 2]] = self.rotation_params[:, [2, 1]]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Called after optimizer step

        if self.automatic_optimization and self.on_train_batch_end_enabled:
            with torch.no_grad():
                # Clip the volume to valid range
                if hasattr(self, "volume") and self.volume.requires_grad and self.imaging_model.microscopy_regime != 'fluorescence':
                    self.volume.clamp_(0.0, 1.0)

                # Normalize rotation parameters (quaternions)
                if hasattr(self, "rotation_params") and self.rotation_params.requires_grad:
                    self.rotation_params[:] = (
                        self.rotation_params / self.rotation_params.norm(dim=1, keepdim=True)
                    )

    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch.
        """
        if self.automatic_optimization and self.on_train_epoch_end_enabled:
            with torch.no_grad():  # prevents autograd tracking

                # --- Optional: clamp volume if needed ---
                #if hasattr(self, "volume") and self.volume.requires_grad:
                #    if self.initial_volume == 'refraction':
                #         self.volume.clamp_(1.0, 2.0)
                #    else:
                #         self.volume.clamp_(0.0, 1.0)

                # --- Handle quaternion rotations ---
                if self.rotation_optim_case == 'quaternion' and self.rotation_params.requires_grad:
                    predicted = self.get_quaternions(self.rotation_params)
                    predicted = predicted / predicted.norm(dim=-1, keepdim=True)

                    # align first quaternion to identity
                    P0 = predicted[0]
                    q_rel = self.quat_conjugate(P0.unsqueeze(0))
                    Q_aligned = self.quat_multiply(q_rel, predicted)
                    Q_aligned = Q_aligned / Q_aligned.norm(dim=-1, keepdim=True)

                    # safely overwrite tensor values
                    self.rotation_params.copy_(Q_aligned)

                # --- Handle basis-based rotations ---
                elif self.rotation_optim_case == 'basis' and self.rotation_params.requires_grad:
                    predicted = self.get_quaternions(self.rotation_params)
                    predicted = predicted / torch.norm(predicted, dim=1, keepdim=True)

                    # align first quaternion to identity
                    P0 = predicted[0]
                    q_rel = self.quat_conjugate(P0.unsqueeze(0))
                    Q_aligned = self.quat_multiply(q_rel, predicted)
                    Q_aligned = Q_aligned / Q_aligned.norm(dim=-1, keepdim=True)

                    # recompute coefficients from basis
                    coeffs = torch.linalg.lstsq(self.basis, Q_aligned).solution

                    # safely overwrite rotation params
                    self.rotation_params.copy_(coeffs)

    def quat_conjugate(self, q):
        # q: (...,4) tensor of unit quaternions (w,x,y,z)
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return torch.stack([w, -x, -y, -z], dim=-1)
    
    def quat_multiply(self, q, r):
        # q, r: (..., 4) tensors of quaternions in (w, x, y, z) format
        w1, x1, y1, z1 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        w2, x2, y2, z2 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

# Testing the code
if __name__ == "__main__":
    pass



        