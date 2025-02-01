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
            nn.Linear(32, xs * ys), nn.Dropout(self.dropout)  # Output size depends on input spatial size
        )

        # Fusion Layer (Concatenation instead of multiplication)
        self.fusion_fc = nn.Conv2d(64 + 1, 64, kernel_size=3, padding=1)  # 1 extra channel from params

        # Output Refinement (to 2-channel image)
        self.output_refinement = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1), 
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

if __name__ == "__main__":
    # Run the test
    test_neural_microscope()