import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardModelSimple(nn.Module):
    """
    Forward model for projecting a 3D volume after applying quaternion-based rotations.
    Works with arbitrary 3D volume shapes (nx, ny, nz).
    """

    def __init__(self, 
                 nx: int = 0, 
                 ny: int = 0, 
                 nz: int = 0,
                 N: int = 0,
                 dim: int = 2,
                 device: torch.device = None):
        """
        Parameters
        ----------
        nx, ny, nz : int
            Size of the 3D volume in x, y, z directions.
        N : int
            If nx, ny, nz are not provided, use N for all dimensions.
        dim : int, optional
            Axis to project along (default = 2).
        """
        super().__init__()

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Handle default size
        if (nx <= 0 or ny <= 0 or nz <= 0) and N > 0:
            nx = ny = nz = N

        self.nx, self.ny, self.nz = nx, ny, nz
        self.dim = dim

        # Create voxel coordinates centered around 0
        x = torch.linspace(-1, 1, nx)
        y = torch.linspace(-1, 1, ny)
        z = torch.linspace(-1, 1, nz)
        grid = torch.stack(torch.meshgrid(z, y, x, indexing='ij'), dim=-1)  # (nz, ny, nx, 3)
        self.grid = grid.to(self.device)  # Store the grid for later use

    def forward(self, volume: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Rotate and project the 3D volume.
        """
        batch_size = quaternions.shape[0]
        out_h, out_w = self.ny, self.nx  # 2D projection size matches input xy
        estimated_projections = torch.zeros(batch_size, out_h, out_w, device=self.device)

        for i in range(batch_size):
            rotated_volume = self.apply_rotation_translation(volume, quaternions[i])
            estimated_projections[i] = self.project(rotated_volume)

        return estimated_projections

    def apply_rotation_translation(self, volume, q, translations=None):

        q = q / q.norm()
        R = self.quaternion_to_rotation_matrix(q)  # (3,3)
        R = R.to(volume.device)

        # Prepare volume: add batch and channel dims if needed
        if volume.dim() == 3:
            volume = volume.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

        # Get volume shape
        _, _, D, H, W = volume.shape

        # Flatten normalized voxel-space grid
        grid = self.grid.view(-1, 3)  # (N^3, 3)
        grid = grid.to(volume.device)

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

    @staticmethod
    def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
        """
        Convert a quaternion to a 3x3 rotation matrix.
        """
        qw, qx, qy, qz = q.unbind()
        return torch.stack([
            torch.stack([1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)], dim=-1),
            torch.stack([2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)], dim=-1),
            torch.stack([2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)], dim=-1)
        ], dim=-2)

    def project(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Project the volume onto a 2D plane by summing along 'dim'.
        """
        return torch.sum(volume, dim=self.dim)

    def full_projection(self, volume: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Compute projections for multiple quaternions.
        """
        out_h, out_w = self.ny, self.nx
        projections = torch.zeros(len(quaternions), out_h, out_w, device=self.device)

        for i, q in enumerate(quaternions):
            rotated_volume = self.apply_rotation(volume, q)
            projections[i] = self.project(rotated_volume)

        return projections
