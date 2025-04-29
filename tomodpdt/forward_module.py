import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardModelSimple(nn.Module):
    def __init__(
            self, 
            N=64,
            dim=2):
        super(ForwardModelSimple, self).__init__()

        self.N = N
        self.dim = dim

        x = torch.arange(self.N) - self.N / 2
        self.xx, self.yy, self.zz = torch.meshgrid(x, x, x, indexing='ij')
        self.grid = torch.stack([self.xx, self.yy, self.zz], dim=-1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to_device()

    def to_device(self):
        """
        Move the grid to the device.
        """

        self.grid = self.grid.to(self.device)

    def forward(self, volume, quaternions):
        """
        Forward pass of the model.

        Parameters:
        - volume (torch.Tensor): The volume to project.
        - quaternions (torch.Tensor): Quaternions representing rotations.

        Returns:
        - estimated_projections (torch.Tensor): Estimated 2D projections of the volume.
        """

        batch_size = quaternions.shape[0]
        estimated_projections = torch.zeros(batch_size, self.N, self.N, device=self.device)

        for i in range(batch_size):
            rotated_volume = self.apply_rotation(volume, quaternions[i])
            estimated_projections[i] = self.project(rotated_volume)
        
        return estimated_projections

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

    def project(self, volume):
        """
        Project the volume to a 2D plane.

        Parameters:
        - volume (torch.Tensor): The volume to project.

        Returns:
        - projection (torch.Tensor): 2D projection of the volume.
        """

        projection = torch.sum(volume, dim=self.dim)
        return projection

    def full_projection(self, volume, quaternions):
        """
        Compute the full projection for a set of quaternions.

        Parameters:
        - volume (torch.Tensor): The volume to project.
        - quaternions (torch.Tensor): Set of quaternions for projection.

        Returns:
        - projections (torch.Tensor): 2D projections of the volume.
        """

        projections = torch.zeros(len(quaternions), volume.shape[1], volume.shape[2], device=self.device)

        for i in range(len(quaternions)):
            rotated_volume = self.apply_rotation(volume, quaternions[i])
            projections[i] = self.project(rotated_volume)
        
        return projections