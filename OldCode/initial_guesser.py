import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from scipy.signal import savgol_filter
import scipy.signal as signal

class InitialGuesser(nn.Module):
    def __init__(self,  
                 frames,
                 model_vae=None,
                 n_epochs_vae=1000,
                 max_n_projections=500,
                 step_size=1,
                 normalize=True,
                 initial_axes='x',
                 tune_rotation_params=True,
                 peaks_period_range=[20, 100]
                 ):
        super(InitialGuesser, self).__init__()

        self.frames = frames
        self.model_vae = model_vae
        self.n_epochs_vae = n_epochs_vae
        self.max_n_projections = max_n_projections
        self.step_size = step_size
        self.normalize = normalize
        self.initial_axes = initial_axes
        self.tune_rotation_params = tune_rotation_params
        self.peaks_period_range = peaks_period_range

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set the number of frames
        self.frames = self.frames[::self.step_size][:self.max_n_projections]

        # Transform the frames to a tensor if it is not
        if not isinstance(self.frames, torch.Tensor):
            self.frames = torch.tensor(self.frames)
        
        # Set the dimensions
        self.xdim = self.frames[0].shape[1]
        self.ydim = self.frames[0].shape[0]

        #If the model is not provided, create a new one
        if self.model_vae is None:
            self.model_vae = ConvVAE(input_shape=(1, self.xdim, self.ydim), latent_dim=2, dropout=0)

        # Set the optimizer
        self.optimizer = torch.optim.Adam(self.model_vae.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.9, patience=100, min_lr=1e-7)

        # Move to device
        self.to_device()

        # Normalize the frames
        if self.normalize:
            self.normalize_frames()

        # Parameters
        self.angle = []
        self.angle_rotations = []
        self.peaks = []
        self.z = None
        self.mu = None
        self.logvar = None
        self.x_recon = None
        self.quaternions = None
        self.res = None

    def pipeline(self):
        """
        Run the pipeline.
        """

        print('Training VAE...')
        self.train_vae()

        print('Finding peaks...')
        self.find_peaks()
        print("Peaks found at: ", self.peaks)

        print('Getting axis angle...')
        self.get_axis_angle()

        print('Converting axis angle to quaternion...')
        self.axis_angle_to_quaternion_torch_batch(self.angle_rotations[:, 1:], self.angle_rotations[:, 0])

        # Setting frames to the number of quaternions
        self.frames = self.frames[:len(self.quaternions)]

        # Set requires grad to false for model
        self.model_vae.requires_grad_(False)
        

    def to_device(self):
        """
        Move to device.
        """
        self.frames = self.frames.to(self.device)
        self.model_vae.to(self.device)
    
    def normalize_frames(self, th=0.1, mask_out=False):
        """
        Normalize the frames.
        """

        # Normalize the projections
        self.frames = (self.frames - self.frames.min()) / (self.frames.max() - self.frames.min())

        if mask_out:
            # Set background pixels to 0 - Subtract the mean image aloing the time axis
            a = self.frames.std(axis=0)

            a = a - a.min()
            a = a / a.max()
            a = a > th
            a = a.float()

            # Set the pixels that are not part of the object to 0
            self.frames = self.frames * a

        # Add a channel dimension if it is not present
        if len(self.frames.shape) == 3:
            self.frames = self.frames.unsqueeze(1)

    def train_vae(self):
        batch_size = 64

        losses = []
        kl_losses = []
        recon_losses = []
        for epoch in range(self.n_epochs_vae):

            # Shuffle the data
            idx = torch.randperm(self.frames.shape[0])
            data = self.frames[idx]

            loss_tmp = 0
            kl_loss_tmp = 0
            recon_loss_tmp = 0

            for i in range(0, data.shape[0], batch_size):

                # Forward pass
                self.optimizer.zero_grad()

                x = data[i:i+batch_size]
                x_recon, mu, logvar = self.model_vae(x)
                loss, recon_loss, kl_loss = self.model_vae.loss(
                    x, x_recon, mu, logvar, weight_kl=1e-3
                    )

                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss)

                loss_tmp += loss.item()
                kl_loss_tmp += kl_loss.item()
                recon_loss_tmp += recon_loss.item()

            losses.append(loss_tmp/batch_size)
            kl_losses.append(kl_loss_tmp/batch_size)
            recon_losses.append(recon_loss_tmp/batch_size)

            # Print the loss
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{self.n_epochs_vae}, Loss: {loss_tmp/batch_size}, KL Loss: {kl_loss_tmp/batch_size}, Recon Loss: {recon_loss_tmp/batch_size}')

        # When model is trained
        x_recon, mu, logvar = self.model_vae(self.frames)
        z = self.model_vae.reparameterize(mu, logvar)

        self.z = z.detach()
        self.mu = mu.detach()
        self.logvar = logvar.detach()
        self.x_recon = x_recon.detach()

    def get_angle_interpolated(self):
        """
        Angle interpolation.
        """

        # Ensure peaks are sorted and unique
        self.peaks = sorted(set(self.peaks))

        # Define full range of timesteps
        total_timesteps = max(self.peaks) + 1
        angles = torch.zeros(total_timesteps)

        # Assign full rotation angles at keyframes
        full_rotation_count = len(self.peaks)
        for i, t in enumerate(self.peaks):
            angles[t] = i * 2 * torch.pi  # Full rotations (0, 2π, 4π, ...)

        # Interpolate between keyframes
        for i in range(1, full_rotation_count):
            start, end = self.peaks[i - 1], self.peaks[i]
            if end > start:  # Avoid division by zero
                angles[start:end] = torch.linspace(angles[start], angles[end], end - start)

        self.angle = angles[:-1]

    def get_axis_angle(self):
        """
        """
        self.angle_rotations = torch.zeros(max(self.peaks), 4)

        # Get the angle rotation
        self.get_angle_interpolated()

        # Get the angle rotations
        self.angle_rotations[:, 0] = torch.tensor(self.angle)

        if self.initial_axes == 'x':
            # set all axis element to [ang, 1, 0, 0]
            self.angle_rotations[:, 2:] = 0
            self.angle_rotations[:, 1] = 1
        elif self.initial_axes == 'y':
            # set all axis element to [ang, 0, 1, 0]
            self.angle_rotations[:, [1, 3]] = 0
            self.angle_rotations[:, 2] = 1
        elif self.initial_axes == 'z':
            # set all axis element to [ang, 0, 0, 1]
            self.angle_rotations[:, 1:3] = 0
            self.angle_rotations[:, 3] = 1

    def axis_angle_to_quaternion_torch_batch(self, axes, angles):
        """
        Convert batch of axis-angle representations to quaternions using PyTorch.

        Parameters:
        - axes (torch.Tensor): Tensor of shape (N, 3) where N is the number of rotations.
        - angles (torch.Tensor): Tensor of shape (N,) representing the rotation angles.

        Returns:
        - quaternions (torch.Tensor): Tensor of shape (N, 4) with quaternions (q_w, q_x, q_y, q_z).
        """
        norms = torch.norm(axes, dim=1, keepdim=True)
        axes = axes / norms  # Normalize the axes

        half_angles = angles / 2
        qw = torch.cos(half_angles)
        sin_half_angles = torch.sin(half_angles)
        q_xyz = axes * sin_half_angles.unsqueeze(1)  # Broadcast sin values to multiply with axes

        # Combine quaternion components
        self.quaternions = torch.cat((qw.unsqueeze(1), q_xyz), dim=1)  # Shape (N, 4)
        
    def compute_distances(self, z):
        """
        Compute distances from the first point in the tensor to all other points.

        Parameters:
        - z (torch.Tensor): Tensor of shape (N, 2), where each row represents a 2D point.

        Returns:
        - dists (torch.Tensor): Tensor of distances from the first point to all other points.
        """
        if z.shape[1] != 2:
            raise ValueError("Input tensor must have shape (N, 2).")

        d0 = z[0]  # Reference point (first row)
        dists = torch.sqrt(((z - d0) ** 2).sum(dim=1))

        dists = dists / dists.max()

        return dists

    def find_peaks(self, window_length=11, polyorder=2, max_peaks=7, min_peaks=2):
        
        # Compute the distances. The closer to 1, the more similar the points are
        res = 1 - self.compute_distances(self.z).cpu().numpy()
        
        # Convert results to a NumPy array and apply a weighting factor. As the distance might degrade somewhat
        res = np.array(res)
        res = res * np.linspace(0.95, 1, len(res))

        # Smooth the result using a Savitzky-Golay filter
        res = savgol_filter(res, window_length=window_length, polyorder=polyorder)
        res = res / max(res)  # Normalize the result

        # Search for peaks using varying distances
        height = 0.8 * np.max(res)
        distance_range = (self.peaks_period_range[0], self.peaks_period_range[1], 10)
    
        for dist in range(*distance_range):
            peaks = signal.find_peaks(res, distance=dist, height=height, prominence=0.7)[0]
            if min_peaks < len(peaks) < max_peaks:
                break
        
        # Add the first peak at index 0 and return the peaks
        peaks = np.append(0, peaks)

        # If number of peaks is less than the minimum, add the last peak
        if len(peaks) < min_peaks:
            peaks = np.append(peaks, len(res) - 1)

        self.peaks = peaks
        self.res = res


class ConvVAE(nn.Module):
    def __init__(
            self,
            input_shape,
            latent_dim,
            conv_channels=[16, 32, 64],
            conv_kernels=[4, 4, 4],
            conv_strides=[2, 2, 2],
            conv_paddings=[1, 1, 1],
            dense_dim=128,
            activation='lrelu',
            output_activation='linear',
            dropout=0.1,
            ):
        super(ConvVAE, self).__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.conv_channels = conv_channels
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.conv_paddings = conv_paddings
        self.dense_dim = dense_dim
        self.activation = activation
        self.output_activation = output_activation
        self.dropout = dropout

        self.encoder = self.build_encoder()
        self.fc_mu = nn.Linear(self.flattened_size(), latent_dim)
        self.fc_log_var = nn.Linear(self.flattened_size(), latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size())
        self.decoder = self.build_decoder()

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'lrelu':
            return nn.LeakyReLU(0.1)
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            return nn.Identity()

    def flattened_size(self):
        dummy_input = torch.zeros(1, *self.input_shape)
        dummy_output = self.encoder(dummy_input)
        return int(torch.prod(torch.tensor(dummy_output.shape[1:])))
        
    def build_encoder(self):
        """
        Build the encoder.
        """
        encoder = []
        in_channels = self.input_shape[0]
        for i in range(len(self.conv_channels)):
            encoder.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.conv_channels[i],
                    kernel_size=self.conv_kernels[i],
                    stride=self.conv_strides[i],
                    padding=self.conv_paddings[i],
                )
            )
            encoder.append(self.get_activation(self.activation))
            encoder.append(nn.Dropout(self.dropout))
            in_channels = self.conv_channels[i]

        return nn.Sequential(*encoder)
    
    def build_decoder(self):
        """
        Build the decoder.
        """
        decoder = []
        in_channels = self.conv_channels[-1]
        in_size = self.input_shape[1] // (2 ** len(self.conv_channels))
        for i in range(len(self.conv_channels) - 1, -1, -1):
            decoder.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=self.conv_channels[i],
                    kernel_size=self.conv_kernels[i],
                    stride=self.conv_strides[i],
                    padding=self.conv_paddings[i],
                )
            )
            decoder.append(self.get_activation(self.activation))
            decoder.append(nn.Dropout(self.dropout))
            in_channels = self.conv_channels[i]

        decoder.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.input_shape[0],
            kernel_size=3,
            padding=1,
        ))
        decoder.append(self.get_activation(self.output_activation))
        
        return nn.Sequential(*decoder)
    
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def forward(self, x):
        """
        Forward pass.
        """
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        z = self.reparameterize(mu, log_var)
        z = self.fc_decode(z).view(-1, self.conv_channels[-1], self.input_shape[1] // (2 ** len(self.conv_channels)), self.input_shape[2] // (2 ** len(self.conv_channels)))
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def loss(self, x, x_hat, mu, log_var, weight_kl=1e-2):
        """
        Calculate the loss.
        """
        recon_loss = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)

        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())*weight_kl

        total_loss = recon_loss + kl_loss

        return total_loss, recon_loss, kl_loss