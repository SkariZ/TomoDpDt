import torch
import torch.nn as nn

class NeuralMicroscope(nn.Module):
    def __init__(self, num_params=6, xs=64, ys=64, dropout=0.1):  # Dynamically set xs, ys
        super(NeuralMicroscope, self).__init__()
        self.xs, self.ys = xs, ys  # Store for later use
        self.dropout = dropout

        # 3D Feature Extraction (keeping xs, ys unchanged)
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1), nn.BatchNorm3d(64), nn.LeakyReLU(0.1), nn.Dropout3d(self.dropout),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.1), nn.Dropout3d(self.dropout),
            nn.Conv3d(64, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.1), nn.Dropout3d(self.dropout),
            nn.Conv3d(32, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.1), nn.Dropout3d(self.dropout),
            nn.Conv3d(32, 1, kernel_size=3, padding=1), nn.LeakyReLU(0.1), nn.Dropout3d(self.dropout)
        )

        # Adaptive pooling to remove depth but keep (xs, ys)
        self.global_avg_pooling = nn.AdaptiveAvgPool3d((1, None, None))  # Keeps (xs, ys)

        # Continuous Parameter Processing - Dynamic xs, ys output!
        self.param_fc = nn.Sequential(
            nn.Linear(num_params, 32), nn.LeakyReLU(0.1), nn.Dropout(self.dropout),
            nn.Linear(32, 32), nn.LeakyReLU(0.1), nn.Dropout(self.dropout),
            nn.Linear(32, xs * ys), nn.Dropout(self.dropout)  # Output size depends on input spatial size
        )

        # Fusion Layer (Concatenation instead of multiplication)
        self.fusion_fc = nn.Conv2d(64 + 1, 64, kernel_size=3, padding=1)  # 1 extra channel from params

        # Output Refinement (to 2-channel image)
        self.output_refinement = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1), 
            nn.LeakyReLU(0.1), 
            nn.Dropout2d(self.dropout),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), 
            nn.LeakyReLU(0.1), 
            nn.Dropout2d(self.dropout),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), 
            nn.LeakyReLU(0.1), 
            nn.Dropout2d(self.dropout),
            nn.Conv2d(32, 2, kernel_size=3, padding=1)  # 2-channel output
        )

    def forward(self, volume, continuous_params):
        """
        Args:
            volume: 3D tensor of shape [Batch, 1, xs, ys, zs]
            continuous_params: [Batch, num_params] (e.g., magnification, wavelength)
        """
        batch_size, _, xs, ys, zs = volume.shape

        # Process 3D volume
        volume_features = self.feature_extractor(volume)  # Keep (xs, ys)
        #volume_features = self.global_avg_pooling(volume_features)  # Reduce depth to 1
        volume_features = volume_features.squeeze(1)  # Shape: [Batch, 64, xs, ys]
        #print(volume_features.shape)

        # Process continuous parameters into (xs, ys)
        param_features = self.param_fc(continuous_params)  # Shape: (B, xs * ys)
        param_features = param_features.view(batch_size, 1, xs, ys)  # Reshape to (B, 1, xs, ys)

        # Concatenate instead of element-wise multiplication
        fused_features = torch.cat([volume_features, param_features], dim=1)  # Shape: [Batch, 65, xs, ys]

        # Refinement to 2-channel 2D output
        fused_features = self.fusion_fc(fused_features)  # Keep (xs, ys)
        output = self.output_refinement(fused_features)  # [Batch, 2, xs, ys]

        return output

# Discriminator for  deciding if the 2d image is real or fake, conditioned on the parameters

class Discriminator(nn.Module):
    def __init__(self, num_params=6, xs=64, ys=64, dropout=0.1):
        super(Discriminator, self).__init__()
        self.xs, self.ys = xs, ys  # Store for later use
        self.dropout = dropout

        # 2D Feature Extraction (keeping xs, ys unchanged)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1), nn.Dropout2d(self.dropout),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.1), nn.Dropout2d(self.dropout),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.1), nn.Dropout2d(self.dropout),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.1), nn.Dropout2d(self.dropout),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 1, kernel_size=3, padding=1), nn.LeakyReLU(0.1), nn.Dropout2d(self.dropout)
        )

        # Continuous Parameter Processing
        self.param_fc = nn.Sequential(
            nn.Linear(num_params, 32), nn.LeakyReLU(0.1), nn.Dropout(self.dropout),
            nn.Linear(32, xs//4 * ys//4), nn.LeakyReLU(0.1), nn.Dropout(self.dropout)  # Output size depends on input spatial size
        )

        # Fusion Layer (Concatenation instead of multiplication)
        self.fusion_fc = nn.Conv2d(2, 16, kernel_size=3, padding=1)#

        # Output Refinement (to 2-channel image)
        self.output_refinement = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1), 
            nn.LeakyReLU(0.1), 
            nn.Dropout2d(self.dropout),
            nn.Flatten(),
            nn.Linear(16 * xs//4 * ys//4, 1),  # 1-channel output
            nn.Sigmoid()
        )

    def forward(self, image, continuous_params):

        batch_size, _, xs, ys = image.shape

        # Process 2D image
        image_features = self.feature_extractor(image)  # Goes to 64->16
        
        # Process continuous parameters into (xs//4, ys//4)
        param_features = self.param_fc(continuous_params)
        param_features = param_features.view(batch_size, 1, xs//4, ys//4)

        # Concatenate instead of element-wise multiplication
        fused_features = torch.cat([image_features, param_features], dim=1)

        # Refinement to 2-channel 2D output
        fused_features = self.fusion_fc(fused_features)

        output = self.output_refinement(fused_features)

        return output



def test_neural_microscope():
    # Define input size
    batch_size = 8
    xs, ys, zs = 64, 64, 64  # Example 3D volume size
    num_params = 1  # Magnification, wavelength, etc.
    
    # Create a random 3D volume (Batch, Channels, xs, ys, zs)
    volume = torch.randn(batch_size, 1, xs, ys, zs)

    # Create random continuous parameters (Batch, num_params)
    continuous_params = torch.randn(batch_size, num_params)

    # Initialize model
    model = NeuralMicroscope(num_params=num_params)

    # Forward pass
    output = model(volume, continuous_params)

    # Check output shape
    assert output.shape == (batch_size, 2, xs, ys), f"Unexpected output shape: {output.shape}"
    
    print("✅ Test passed! Output shape:", output.shape)

    # Initialize discriminator
    discriminator = Discriminator(num_params=num_params)

    # Forward pass
    output = discriminator(output, continuous_params)

    # Check output shape
    assert output.shape == (batch_size, 1), f"Unexpected output shape: {output.shape}"

    print("✅ Test passed! Output shape:", output.shape)


# Conditional GAN with Neural Microscope
# The Neural Microscope model is a conditional GAN that generates a 2D image from a 3D volume and continuous parameters.

# The model consists of two main components:
# 1. NeuralMicroscope: Generates a 2D image from a 3D volume and continuous parameters.
# 2. Discriminator: Decides if the 2D image is real or fake, conditioned on the parameters.

def trainv2():
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Hyperparameters
    latent_dim = 16  # Size of the noise vector
    num_params = 6   # Number of continuous parameters
    epochs = 10000   # Training iterations
    batch_size = 8   # Batch size
    lr = 0.0002      # Learning rate

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = NeuralMicroscope(num_params + latent_dim).to(device)
    discriminator = Discriminator(num_params).to(device)

    # Loss function and optimizers
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    l1_loss = nn.L1Loss()  # Optional: L1 loss for smoothness
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Labels
    real_label = 1.0
    fake_label = 0.0

    # Create a dummy dataset
    images = torch.randn(200, 2, 64, 64)  # 2-channel images
    volumes = torch.randn(200, 1, 64, 64, 64)  # 3D volumes
    continuous_params = torch.randn(200, num_params)  # Continuous parameters

    from torch.utils.data import DataLoader, TensorDataset
    # Create a custom dataset
    dataset = TensorDataset(volumes, continuous_params, images)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training Loop
    for epoch in range(epochs):
        for i, (real_3d_volumes, params, real_images,) in enumerate(dataloader):  
            real_3d_volumes = real_3d_volumes.to(device)  # Shape: [B, 1, xs, ys, zs]
            real_images = real_images.to(device)  # Shape: [B, 2, xs, ys]
            params = params.to(device)  # Shape: [B, num_params]

            # Generate random noise (z)
            z = torch.randn(batch_size, latent_dim, device=device)  # Shape: [B, latent_dim]

            ### Train Discriminator ###
            d_optimizer.zero_grad()
            
            # Forward pass real images
            real_labels = torch.full((batch_size, 1), real_label, device=device)
            real_preds = discriminator(real_images, params)  # Shape: [B, 1]
            d_real_loss = criterion(real_preds, real_labels)  # BCE loss

            # Generate fake images
            fake_input = torch.cat([params, z], dim=1)  # Combine params + noise
            fake_images = generator(real_3d_volumes, fake_input)  # Fake images
            
            # Forward pass fake images
            fake_labels = torch.full((batch_size, 1), fake_label, device=device)
            fake_preds = discriminator(fake_images.detach(), params)
            d_fake_loss = criterion(fake_preds, fake_labels)

            # Backprop for Discriminator
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            ### Train Generator ###
            g_optimizer.zero_grad()

            # Forward pass fake images (again)
            fake_preds = discriminator(fake_images, params)  # Try to fool D
            g_loss = criterion(fake_preds, real_labels)  # Wants D to classify fake as real

            # Optional: Add L1 loss for smooth image output
            g_l1 = l1_loss(fake_images, real_images) * 10  # Weight = 10
            g_loss += g_l1

            # Backprop for Generator
            g_loss.backward()
            g_optimizer.step()

        # Print losses every few epochs
        if epoch % 1 == 0:
            print(f"Epoch [{epoch}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    print("Training complete!")


def train():
    # Initialize Neural Microscope
    num_params = 4  # Magnification, wavelength, resolution, NA
    model = NeuralMicroscope(num_params=num_params)

    # Initialize Discriminator
    discriminator = Discriminator(num_params=num_params)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(discriminator.parameters()), lr=0.001)
    

    num_epochs = 10
    total_steps = 1000
    batch_size = 16

    # Create a dummy dataset
    images = torch.randn(200, 2, 64, 64)  # 2-channel images
    volumes = torch.randn(200, 1, 64, 64, 64)  # 3D volumes
    continuous_params = torch.randn(200, num_params)  # Continuous parameters

    from torch.utils.data import DataLoader, TensorDataset
    # Create a custom dataset
    dataset = TensorDataset(volumes, continuous_params, images)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        for i, (volume, continuous_params, real_images) in enumerate(train_loader):
            # Generate fake images
            fake_images = model(volume, continuous_params)

            # Train Discriminator
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)
            real_outputs = discriminator(real_images, continuous_params)
            fake_outputs = discriminator(fake_images, continuous_params)
            d_loss_real = criterion(real_outputs, real_labels)
            d_loss_fake = criterion(fake_outputs, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            optimizer.zero_grad()
            d_loss.backward()
            optimizer.step()

            # Train Neural Microscope
            fake_images = model(volume, continuous_params)
            outputs = discriminator(fake_images, continuous_params)
            g_loss = criterion(outputs, real_labels) 
            optimizer.zero_grad()
            g_loss.backward()
            optimizer.step()

            # Print loss
            if i % 1 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{total_steps}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")

    print("Training complete!")

if __name__ == "__main__":
    # Run the test
    test_neural_microscope()