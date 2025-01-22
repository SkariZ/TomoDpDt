import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class OptimizeVolume(nn.Module):
    def __init__(
            self,
            forward_model,
            initial_guesser,
            quaternions_init,
            projections,
            optimize_quaternions=True,
            dim=2,
            initial_volume='gaussian',
            volume_init=None):
        super(OptimizeVolume, self).__init__()

        self.N = forward_model.N
        self.forward_model = forward_model
        self.initial_guesser = initial_guesser
        self.projections = projections
        self.dim = dim
        self.initial_volume = initial_volume
        self.volume_init = volume_init
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize quaternions
        self.quaternions = quaternions_init.to(self.device)
        if optimize_quaternions:
            self.quaternions = nn.Parameter(self.quaternions)

        # ground truth z for perception loss
        self.z_gt = self.initial_guesser.z[:len(self.quaternions)].detach()

        # Initialize volume
        self.initialize_volume()

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

        if self.initial_volume == 'fbp':
            
            try:
                import tomopy
                import os
                os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            except ImportError:
                raise ImportError('tomopy is required for FBP initialization.')

            if self.projections.dim() == 4 and self.projections.size(1) == 1:
                self.projections = self.projections.squeeze(1)

            # Get the cumulative angles from the cumulative quaternions
            angles = 2 * torch.arccos(self.quaternions[:, 0])

            # Full rotation
            try:
                idx_full = torch.where(angles == 2*np.pi)[0][0]
            except IndexError:
                idx_full = len(angles)

            angles = angles[:idx_full]

            # Convert to numpy
            angles = angles.cpu().detach().numpy()
            
            # Find the center of rotation
            center = tomopy.find_center(self.projections[:idx_full].cpu().detach().numpy(), angles)

            # Perform FBP reconstruction
            volume_fbp = tomopy.recon(self.projections[:idx_full].cpu().detach().numpy(), angles, center=center, algorithm='gridrec')

            # Rearrange axes: (0, 1, 2) -> (1, 2, 0)
            volume_fbp = np.transpose(volume_fbp, axes=(1, 2, 0))

            # Flip along the new 0th axis
            volume_fbp = np.flip(volume_fbp, axis=0)

            # Normalize the volume
            volume_fbp = volume_fbp / volume_fbp.max()

            # Convert to tensor and move to device
            self.volume = nn.Parameter(torch.tensor(volume_fbp, device=self.device))

        elif self.initial_volume == 'ones':
            self.volume = nn.Parameter(torch.ones(self.N, self.N, self.N, device=self.device))
        
        elif self.initial_volume == 'zeros':
            self.volume = nn.Parameter(torch.zeros(self.N, self.N, self.N, device=self.device))

        elif self.initial_volume == 'random':
            self.volume = nn.Parameter(torch.rand(self.N, self.N, self.N, device=self.device))

        elif self.initial_volume == 'given' and self.volume_init is not None:
            self.volume = nn.Parameter(self.volume_init.to(self.device))

        else:
            self.volume = nn.Parameter(torch.zeros(self.N, self.N, self.N, device=self.device))
    
    def forward(self, quaternions):
        """
        Forward pass of the model.

        Returns:
        - estimated_projections (torch.Tensor): Estimated 2D projections of the volume.
        """
        # Normalize quaternions during computation
        normalized_quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
        estimated_projections = self.forward_model(self.volume, normalized_quaternions)
        
        return estimated_projections

    def translate_xy_projection_upsample(projections, shift, upsample_factor=4):
        """
        Translate the projections in the x-y plane by fractional shifts using upsampling.

        Args:
        - projections (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        - shift (tuple of float): Fractional shift (shift_x, shift_y).
        - upsample_factor (int): Factor to upsample the projections.

        Returns:
        - shifted_projections (torch.Tensor): Translated projections with fractional shifts.
        """
        batch_size, channels, height, width = projections.shape
        shift_x, shift_y = shift

        # Upsample projections
        new_height, new_width = height * upsample_factor, width * upsample_factor
        upsampled_projections = F.interpolate(projections, size=(new_height, new_width), mode='bilinear', align_corners=True)

        # Compute integer shifts at the upsampled resolution
        int_shift_x = int(round(shift_x * upsample_factor))
        int_shift_y = int(round(shift_y * upsample_factor))

        # Apply integer shift using torch.roll
        shifted_upsampled_projections = torch.roll(
            torch.roll(upsampled_projections, shifts=int_shift_x, dims=3),
            shifts=int_shift_y, dims=2
        )

        # Downsample back to original resolution
        shifted_projections = F.interpolate(shifted_upsampled_projections, size=(height, width), mode='bilinear', align_corners=True)

        return shifted_projections

    def rotational_trajectory_regularization(self, q, lambda_q=1e-2):
        """
        Calculate the rotational trajectory regularization term.
        
        Args:
        - q (torch.Tensor): A tensor of shape (T, d) where T is the number of time steps and d is the dimensionality of q.
        - lambda_q (float): Scalar regularization strength for the trajectory regularization.
        
        Returns:
        - R_q (float): The rotational trajectory regularization term.
        """
        return lambda_q * torch.sum((q[1:] - q[:-1]).norm(p=2, dim=1)**2) / q.shape[0]

    def total_variation_regularization(self, delta_n, lambda_TV=1e-2):
        """
        Calculate the total variation regularization term in 3D without creating large intermediate tensors.

        Args:
        - delta_n (torch.Tensor): A tensor of shape (D, H, W) or higher dimensional array.
        - lambda_TV (float): Scalar regularization strength for the total variation regularization.

        Returns:
        - R_TV (float): The total variation regularization term.
        """
        # Compute gradients and sum them inline to avoid intermediate tensors
        grad_x_sum = torch.sum(torch.abs(delta_n[1:, :, :] - delta_n[:-1, :, :]))  # Gradient in x-direction
        grad_y_sum = torch.sum(torch.abs(delta_n[:, 1:, :] - delta_n[:, :-1, :]))  # Gradient in y-direction
        grad_z_sum = torch.sum(torch.abs(delta_n[:, :, 1:] - delta_n[:, :, :-1]))  # Gradient in z-direction

        # Combine all gradient sums
        R_TV = lambda_TV * (grad_x_sum + grad_y_sum + grad_z_sum) / delta_n.numel()

        return R_TV

    def quaternion_validity_loss(self, q, lambda_q_valid=1e-2):
        """
        Loss to enforce that quaternions remain valid (unit quaternions).
        
        Args:
        - q (torch.Tensor): Tensor of quaternions with shape (N, 4), where N is the number of quaternions.
        - lambda_q_valid (float): Regularization strength for quaternion validity. A higher value enforces the constraint more strictly.
        
        Returns:
        - loss (torch.Tensor): The quaternion validity loss.
        """
        # Compute the squared norm of the quaternion
        norm_squared = torch.sum(q**2, dim=1)  # Sum over the 4 components (q0, q1, q2, q3) for each quaternion
        
        # Compute the difference between the norm squared and 1
        diff_from_unit = norm_squared - 1
        
        # The loss is the square of the difference
        return lambda_q_valid * torch.sum(diff_from_unit**2) / q.shape[0]
    
    def q0_constraint_loss(self, q, lambda_q0=1e-2):
        """
        Enforce that the q0 component of the quaternion to be [1, 0, 0, 0]. Just a simple constraint. So it stays at the starting point.
        """
        q_start = torch.tensor([1, 0, 0, 0], device=self.device)
        return lambda_q0 * torch.sum((q - q_start)**2)

    def perception_loss_vae(self, projections):

        _, mu, logvar = self.initial_guesser.model_vae(projections)
        zp = self.initial_guesser.model_vae.reparameterize(mu, logvar)

        return torch.linalg.norm(zp - self.z_gt, axis=1).mean()

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

    def fit(
        self, 
        projections=None, 
        criterion=None, 
        optimizer=None, 
        scheduler=None, 
        epochs=200, 
        batch_size=1000, 
        learning_rate=8e-3,
        loss_weights=[1, 10, 0.1, 0.1, 10],
        gaussian_blur=False,
        gaussian_range=[1e-5, 2],
        quaternion_optim_wait=0,
        volume_optim_wait=0,
            ):
        """
        Fit the model in a batch-wise manner.

        Parameters:
        - projections (torch.Tensor): 2D projections of the volume.
        - criterion (torch.nn): Loss function for projection error.
        - optimizer (torch.optim): Optimizer to use.
        - scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        - epochs (int): Number of epochs to train.
        - batch_size (int): Batch size for training.
        - learning_rate (float): Learning rate for the optimizer.
        - loss_weights (list): Weights for the different loss terms in the following order: [MAE, Rotational, TV, q_valid, q0]
        - gaussian_blur (bool): Whether to apply Gaussian blur to the projections.
        - gaussian_range (list): Range of standard deviations for the Gaussian blur.
        - quaternion_optim_wait (int): Number of epochs to wait before optimizing quaternions.
        """

        if projections is not None:
            self.projections = projections

        # Match projections with quaternions
        if self.projections.shape[0] > self.quaternions.shape[0]:
            print('More projections than quaternions. Truncating projections to match quaternions.')
            self.projections = self.projections[:self.quaternions.shape[0]]

        # Transform the projections to a tensor
        if not torch.is_tensor(self.projections):
            self.projections = torch.tensor(self.projections, device=self.device)

        # Remove channel dimension if it exists it is in the form of (N,1,x,y)
        if self.projections.dim() == 4 and self.projections.size(1) == 1:
            self.projections = self.projections.squeeze(1)

        # Default loss function, optimizer, and scheduler
        if criterion is None:
            criterion = nn.L1Loss()
        if optimizer is None:
            print(f'Using Adam optimizer with learning rate {learning_rate}')
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        if scheduler is None:
            print('Using ReduceLROnPlateau scheduler with patience 50')
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.75, patience=50, min_lr=1e-7, threshold=5e-3
            )

        # Calculate the number of batches
        num_batches = (self.projections.shape[0] + batch_size - 1) // batch_size  # Ceiling division

        # Apply Gaussian blur to the projections
        if gaussian_blur:
            sigmas = torch.linspace(gaussian_range[0], gaussian_range[1], epochs)
        
        # Freeze the quaternions for a certain number of epochs
        if quaternion_optim_wait > 0:
            self.quaternions.requires_grad = False

        # Freeze the volume for a certain number of epochs    
        if volume_optim_wait > 0:
            self.volume.requires_grad = False

        # Initialize lists to store losses
        self.loss_total = []
        self.loss_mae = []
        self.loss_rot = []
        self.loss_tv = []
        self.loss_q_valid = []
        self.loss_q0 = []
        
        # Print the initial setup
        print(f"Number of batches: {num_batches}")
        print(f"Number of epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Loss weights: {loss_weights}")
        print(f"Optimizing quaternions: {self.quaternions.requires_grad}")
        print(f"Using Gaussian blur: {gaussian_blur}")
        print(f"Quaternions optimization wait: {quaternion_optim_wait}")
    
        # Training loop
        for epoch in range(epochs):
            
            # Enable gradients for quaternions after a certain number of epochs
            if epoch == quaternion_optim_wait:
                self.quaternions.requires_grad = True
                print('Optimizing quaternions now.')
            
            # Enable gradients for volume after a certain number of epochs
            if epoch == volume_optim_wait:
                self.volume.requires_grad = True
                print('Optimizing volume now.')

            epoch_loss = 0.0
            mae_loss, rot_loss, tv_loss, q_valid_loss, q0_loss = 0.0, 0.0, 0.0, 0.0, 0.0
            for batch_idx in range(num_batches):
                # Extract the batch of projections and quaternions
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, self.projections.shape[0])
                
                batch_projections = self.projections[start_idx:end_idx]

                # Apply Gaussian blur to the projections if needed
                if gaussian_blur:
                    batch_projections = self.gaussian_blur_projection(
                        batch_projections, sigma=sigmas[epoch]
                        )

                batch_quaternions = self.quaternions[start_idx:end_idx]

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                estimated_projections = self.forward(batch_quaternions)

                # Calculate losses
                loss_mae = criterion(
                    estimated_projections, batch_projections
                    ) * loss_weights[0]
                # Rotational trajectory regularization loss
                loss_rot = self.rotational_trajectory_regularization(
                    batch_quaternions, lambda_q=10
                    ) * loss_weights[1]
                # TV loss
                loss_tv = self.total_variation_regularization(
                    self.volume, lambda_TV=0.1
                    ) * loss_weights[2]
                # Q_valid loss
                loss_q_valid = self.quaternion_validity_loss(
                    batch_quaternions, lambda_q_valid=0.1
                    ) * loss_weights[3]

                if batch_idx == 0:
                    loss_q0 = self.q0_constraint_loss(
                        batch_quaternions[0], lambda_q0=10
                        ) * loss_weights[4]
                else:
                    loss_q0 = torch.tensor(0.0)

                # Combine losses
                loss = loss_mae + loss_rot + loss_tv + loss_q_valid + loss_q0

                epoch_loss += loss.item() / num_batches
                mae_loss += loss_mae.item() / num_batches
                rot_loss += loss_rot.item() / num_batches
                tv_loss += loss_tv.item() / num_batches
                q_valid_loss += loss_q_valid.item() / num_batches
                q0_loss += loss_q0.item() / num_batches
                
                # Backpropagation and optimization step
                loss.backward()
                optimizer.step()

            # Scheduler step at the end of the epoch
            scheduler.step(epoch_loss)

            print(
                f"Epoch {epoch+1}/{epochs}, Total loss: {epoch_loss:.5f}, "
                f"MAE loss: {loss_mae:.5f}, Rotational loss: {loss_rot:.5f}, "
                f"TV loss: {loss_tv:.5f}, Quaternion validity loss: {loss_q_valid:.5f}, "
                f"q0 constraint loss: {loss_q0:.5f}"
            )

            self.loss_total.append(epoch_loss)
            self.loss_mae.append(mae_loss)
            self.loss_rot.append(rot_loss)
            self.loss_tv.append(tv_loss)
            self.loss_q_valid.append(q_valid_loss)
            self.loss_q0.append(q0_loss)

            # Every 100th epoch print current learning rate
            if (epoch+1) % 100 == 0:
                for param_group in optimizer.param_groups:
                    print(f"Current learning rate: {param_group['lr']}")

            # Free up memory
            del loss, loss_mae, loss_rot, loss_tv, loss_q_valid, loss_q0, estimated_projections

    def get_volume_np(self):
        """
        Get the optimized volume.

        Returns:
        - volume (torch.Tensor): Optimized volume.
        """
        return self.volume.detach().cpu().numpy()

    def get_projections_np(self):
        """
        Get the estimated projections.

        Returns:
        - estimated_projections (torch.Tensor): Estimated 2D projections of the volume.
        """
        return self.forward(self.quaternions).detach().cpu().numpy()