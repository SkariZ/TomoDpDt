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
                 dim: int = 2):
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

        # Handle default size
        if nx <= 0 or ny <= 0 or nz <= 0 and N>0:
            nx = ny = nz = N

        self.nx, self.ny, self.nz = nx, ny, nz
        self.dim = dim

        # Create voxel coordinates centered around 0
        x = torch.arange(nx) - nx / 2
        y = torch.arange(ny) - ny / 2
        z = torch.arange(nz) - nz / 2

        # Meshgrid in (x,y,z) order, indexing='ij' ensures correct axis alignment
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')  # shape: (nz, ny, nx)

        # Stack coordinates in (z,y,x) order for consistency with grid_sample
        grid = torch.stack([xx, yy, zz], dim=-1).float()  # (nz, ny, nx, 3)

        # Normalize to [-1,1] for grid_sample
        grid[..., 0] = grid[..., 0] / (nx / 2)
        grid[..., 1] = grid[..., 1] / (ny / 2)
        grid[..., 2] = grid[..., 2] / (nz / 2)

        self.register_buffer("base_grid", grid.view(-1, 3).clamp(-1, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move base_grid to the device
        self.base_grid = self.base_grid.to(self.device)

    def forward(self, volume: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Rotate and project the 3D volume.
        """
        batch_size = quaternions.shape[0]
        out_h, out_w = self.ny, self.nx  # 2D projection size matches input xy
        estimated_projections = torch.zeros(batch_size, out_h, out_w, device=self.device)

        for i in range(batch_size):
            rotated_volume = self.apply_rotation(volume, quaternions[i])
            estimated_projections[i] = self.project(rotated_volume)

        return estimated_projections

    def apply_rotation(self, volume: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Rotate the volume using quaternion q.
        """
        q = q / q.norm()
        R = self.quaternion_to_rotation_matrix(q)  # (3,3)

        rotated_grid = torch.matmul(self.base_grid, R.t()).view(self.nz, self.ny, self.nx, 3)
        rotated_grid = rotated_grid.clamp(-1, 1)

        vol = volume
        if vol.dim() == 3:
            vol = vol.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

        rotated_volume = F.grid_sample(vol, rotated_grid.unsqueeze(0),
                                       align_corners=True, padding_mode='zeros')
        return rotated_volume.squeeze(0).squeeze(0)

    def apply_rotation_translation(self, volume, q, translations=None):
        """
        Rotate and optionally translate the volume.
        translations: tensor of shape (3,) in voxel units (x, y, z).
        """
        q = q / q.norm()
        R = self.quaternion_to_rotation_matrix(q)

        grid = self.base_grid.clone()  # (nx*ny*nz, 3)

        if translations is not None:
            tx, ty, tz = translations
            grid[:, 0] -= tx / (self.nx / 2)
            grid[:, 1] -= ty / (self.ny / 2)
            grid[:, 2] -= tz / (self.nz / 2)

        rotated_grid = torch.matmul(grid, R.t()).view(self.nz, self.ny, self.nx, 3)
        rotated_grid = rotated_grid.clamp(-1, 1)

        vol = volume
        if vol.dim() == 3:
            vol = vol.unsqueeze(0).unsqueeze(0)

        rotated_volume = F.grid_sample(vol, rotated_grid.unsqueeze(0), align_corners=True)
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
