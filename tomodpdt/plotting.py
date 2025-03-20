import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


def plot_latent_space(z, save_folder=None, dpi=200):
    """
    Plots the latent space in 2D and 3D.

    Parameters:
    z (numpy array): Latent space array.
    save_folder (str, optional): Path to save the plot. Defaults to None.
    dpi (int, optional): Resolution of the saved figure. Defaults to 100.
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # 2D plot
    axes[0].set_title("Latent space")
    scatter = axes[0].scatter(z[:, 0], z[:, 1], c=np.arange(z.shape[0]))
    axes[0].scatter(z[0, 0], z[0, 1], c='r')  # Start point
    fig.colorbar(scatter, ax=axes[0])
    # Set labels
    axes[0].set_xlabel('z1')
    axes[0].set_ylabel('z2')
    
    # 3D plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title("Latent space (3D)")
    ax.scatter(z[:, 0], z[:, 1], np.arange(z.shape[0]))
    ax.scatter(z[0, 0], z[0, 1], 0, c='r')
    # Set labels
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_zlabel('Frame No.')
    
    plt.tight_layout()
    
    if save_folder is not None:
        plt.savefig(save_folder + 'latent_space_combined.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    
    plt.show()


def plot_reconstructed_vs_gt(recon_pred, recon_gt, save_name='recon_vs_gt', column_headers=["Reconstructed", "Ground Truth"], save_folder=None, dpi=200):
    """
    Plots reconstructed and ground truth frames side by side.

    Parameters:
    recon_pred (numpy array): Reconstructed frames.
    recon_gt (numpy array): Ground truth frames.
    save_name (str, optional): Name of the saved plot. Defaults to 'recon_vs_gt'.
    column_headers (list, optional): Labels for the columns. Defaults to ["Reconstructed", "Ground Truth"].
    save_folder (str, optional): Path to save the plot. Defaults to None.
    dpi (int, optional): Resolution of the saved figure. Defaults to 100.
    """

    fig, ax = plt.subplots(3, 6, figsize=(12, 6))  # 3 rows, 6 columns
    plt.suptitle(f"{column_headers[0]} vs {column_headers[1]} Frames", fontsize=14)

    # Add column headers
    for j, label in zip([1, 4], column_headers):
        ax[0, j].set_title(label, fontsize=16, fontweight="bold")

    for i in range(3):
        for j in range(3):
            # Reconstructed frames (Left 3 columns)
            ax[i, j].imshow(recon_pred[i * 3 + j, 0], cmap="gray")
            ax[i, j].set_title(f'Recon {i * 3 + j}')
            ax[i, j].axvline(x=recon_pred.shape[2] // 2, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax[i, j].axis('off')

            # Ground truth frames (Right 3 columns)
            ax[i, j + 3].imshow(recon_gt[i * 3 + j, 0], cmap="gray")
            ax[i, j + 3].set_title(f'GT {i * 3 + j}')
            ax[i, j + 3].axvline(x=recon_gt.shape[2] // 2, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax[i, j + 3].axis('off')

    if save_folder is not None:
        plt.savefig(save_folder + f'{save_name}.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_sinogram(frames, slice_n=None, save_name="sinogram", save_folder=None, dpi=100):
    """
    Plots the sinogram of a series of frames.

    Parameters:
    frames (numpy array): 2D array representing the frames (n_frames, xsize, ysize, channel).
    slice_axis (str, optional): Axis along which to slice the object. Defaults to 'x'.
    slice (int, optional): Index of the slice along the specified axis. Defaults to None.
    save_name (str, optional): Name of the saved plot. Defaults to "sinogram".
    save_folder (str, optional): Path to save the plot. Defaults to None.
    dpi (int, optional): Resolution of the saved figure. Defaults to 100.
    """

    if slice_n is None:
        slice_n = frames.shape[1] // 2

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plt.suptitle(save_name)

    # Plot sinogram along x and y axes
    ax[0].set_title("Sinogram along x-axis")
    ax[0].imshow(frames[:, slice_n, :, 0].T, cmap='gray', aspect='auto')
    ax[1].set_title("Sinogram along y-axis")
    ax[1].imshow(frames[:, :, slice_n, 0].T, cmap='gray', aspect='auto')

    if save_folder is not None:
        plt.savefig(save_folder + f'{save_name}.png', dpi=dpi, bbox_inches='tight', pad_inches=0)


def plot_sum_object(object, save_name="object", save_folder=None, dpi=100):
    """
    Plots the sum projections of a predicted object along different axes.

    Parameters:
    object (numpy array): 3D array representing the predicted object.
    save_name (str, optional): Name of the saved plot. Defaults to "object".
    save_folder (str, optional): Path to save the plot. Defaults to None.
    dpi (int, optional): Resolution of the saved figure. Defaults to 100.
    """
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    plt.suptitle(f"Summation views of {save_name}")

    # Plot sum projections along different axes
    ax[0].set_title("Sum along x-axis")
    im = ax[0].imshow(object.sum(0))
    ax[1].set_title("Sum along y-axis")
    ax[1].imshow(object.sum(1))
    ax[2].set_title("Sum along z-axis")
    ax[2].imshow(object.sum(2))

    # Add a colorbar
    fig.colorbar(im, ax=ax)

    if save_folder is not None:
        plt.savefig(save_folder + f'{save_name}.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    
    plt.show()


def plot_quaternions(quaternions, save_name="quaternion", save_folder=None, dpi=100):
    """
    Plots quaternion components over frames.

    Parameters:
    quaternions (numpy array): 2D array representing the quaternion components over frames.
    """
    plt.figure(figsize=(7, 4))
    plt.plot(quaternions[:, 0].cpu(), label=r'$q_0$', linewidth=2)
    plt.plot(quaternions[:, 1].cpu(), label=r'$q_1$', linewidth=2)
    plt.plot(quaternions[:, 2].cpu(), label=r'$q_2$', linewidth=2)
    plt.plot(quaternions[:, 3].cpu(), label=r'$q_3$', linewidth=2)
    
    plt.legend()
    plt.xlabel('Frame')
    plt.ylabel('Quaternion')
    plt.title('Quaternions over time')
    
    if save_folder is not None:
        plt.savefig(save_folder + f'{save_name}.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    
    plt.show()


def plot_grid33_frames(projections, title='frames', save_name='frames_grid', save_folder=None, dpi=100):
    """
    Plots a 3x3 grid of frames.

    Parameters:
    projections (numpy array or torch.Tensor): Array containing frame projections.
    """
    fig, ax = plt.subplots(3, 3, figsize=(6, 6))
    plt.suptitle(title)
    
    for i in range(3):
        for j in range(3):
            ax[i, j].imshow(projections[i * 3 + j, 0], cmap='gray')
            ax[i, j].set_title(f'Frame {i * 3 + j}')
            ax[i, j].axis('off')
    
    if save_folder is not None:
        plt.savefig(save_folder + f'{save_name}.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.show()


def plots_initial(tomo, save_folder=None, gt=None, dpi=250):
    
    z = tomo.latent.detach().cpu().numpy()

    # Plot the latent space
    plot_latent_space(z, save_folder=save_folder, dpi=dpi)

    # Plot the smoothed distances and peaks
    smoothed_dists = tomo.rotation_initial_dict['smoothed_distances'].cpu().numpy()
    peaks = tomo.rotation_initial_dict['peaks'].cpu().numpy()

    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])  # 1:2 ratio

    # **Plot 1: Smoothed Distances (Narrow)**
    ax0 = plt.subplot(gs[0])
    ax0.plot(smoothed_dists)
    ax0.scatter(peaks, smoothed_dists[peaks], c='r', label="Peaks")
    ax0.set_title("Smoothed Distances and Peaks")
    ax0.set_xlabel("Time Step")
    ax0.set_ylabel("Smoothed Distance")
    ax0.legend()

    # **Plot 2: Quaternions (Wider)**
    ax1 = plt.subplot(gs[1:])
    q1 = tomo.rotation_initial_dict['quaternions'].cpu().numpy()

    ax1.plot(q1[:, 0], label=r'$q_0$', linewidth=2)
    ax1.plot(q1[:, 1], label=r'$q_1$', linewidth=2)
    ax1.plot(q1[:, 2], label=r'$q_2$', linewidth=2)
    ax1.plot(q1[:, 3], label=r'$q_3$', linewidth=2)
    ax1.axvline(x=len(q1), color='black', linestyle='--', linewidth=3, label='End of q1')

    if gt is not None:
        gt = gt.cpu().numpy()
        ax1.plot(gt[:, 0], '--', label=r'$q_0$ (GT)', linewidth=2)
        ax1.plot(gt[:, 1], '--', label=r'$q_1$ (GT)', linewidth=2)
        ax1.plot(gt[:, 2], '--', label=r'$q_2$ (GT)', linewidth=2)
        ax1.plot(gt[:, 3], '--', label=r'$q_3$ (GT)', linewidth=2)

    ax1.set_title("Initial Guess vs True Quaternion Components")
    ax1.legend()

    # **Save if needed**
    if save_folder is not None:
        plt.savefig(save_folder + 'combined_plot_wider.png', dpi=dpi, bbox_inches='tight', pad_inches=0)

    plt.show()

    tomo.vae_model.to(tomo.frames.device)
    recon_vae_pred = tomo.vae_model(tomo.frames[:9])[0].cpu()
    recon_vae_gt = tomo.frames[:9].cpu()

    # Plot the reconstructed vs ground truth frames
    plot_reconstructed_vs_gt(recon_vae_pred, recon_vae_gt, save_name='recon_vs_gt_initial', save_folder=save_folder, dpi=dpi)


def plots_optim(tomo, save_folder=None, gt_q=None, gt_v=None, dpi=250, plot_3d=True):
    
    predicted_object = tomo.volume.detach().cpu()
    projections_pred = tomo.full_forward_final().detach().cpu().numpy()
    projections_gt = tomo.frames.detach().cpu().numpy()
    quaternions_pred = tomo.get_quaternions_final().detach().cpu().numpy()
    
    # Plot the Predicted_objects axes
    plot_sum_object(predicted_object.numpy(), save_name="predicted_object", save_folder=save_folder, dpi=dpi)
    
    if gt_v is not None:
        plot_sum_object(gt_v.numpy(), save_name="gt_object", save_folder=save_folder, dpi=dpi)

    # Plot the projections
    plot_reconstructed_vs_gt(projections_pred, projections_gt, save_name='projections', column_headers=["Predicted", "Ground Truth"], save_folder=save_folder, dpi=dpi)

    # Plot the quaternions
    plt.figure(figsize=(7, 4))
    plt.plot(quaternions_pred[:, 0], label=r'$q_0$', linewidth=2)
    plt.plot(quaternions_pred[:, 1], label=r'$q_1$', linewidth=2)
    plt.plot(quaternions_pred[:, 2], label=r'$q_2$', linewidth=2)
    plt.plot(quaternions_pred[:, 3], label=r'$q_3$', linewidth=2)
    # Add a vertical line where q1 ends
    plt.axvline(x=len(quaternions_pred), color='black', linestyle='--', linewidth=3, label='End of q1')
    if gt_q is not None:
        plt.plot(gt_q[:, 0], '--', label=r'$q_0$', linewidth=2)
        plt.plot(gt_q[:, 1], '--', label=r'$q_1$', linewidth=2)
        plt.plot(gt_q[:, 2], '--', label=r'$q_2$', linewidth=2)
        plt.plot(gt_q[:, 3], '--', label=r'$q_3$', linewidth=2)
    plt.legend()
    plt.title("Predicted vs. True Quaternion Components")
    if save_folder is not None:
        plt.savefig(save_folder + 'quaternions.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.show()

    # Difference between predicted and true quaternions
    if gt_q is not None:
        diff = quaternions_pred - gt_q[:len(quaternions_pred)].numpy()
        plt.figure(figsize=(7, 4))
        plt.plot(diff[:, 0], label=r'$q_0$', linewidth=2)
        plt.plot(diff[:, 1], label=r'$q_1$', linewidth=2)
        plt.plot(diff[:, 2], label=r'$q_2$', linewidth=2)
        plt.plot(diff[:, 3], label=r'$q_3$', linewidth=2)
        plt.legend()
        plt.title("Difference Predicted vs. True Quaternion Components")
        if save_folder is not None:
            plt.savefig(save_folder + 'quaternions_diff.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.show()

    # 2x2 grid of scatter plots for each component of the quaternion
    if gt_q is not None:
        fig, ax = plt.subplots(2, 2, figsize=(6, 6))
        fig.suptitle("Scatter plots of predicted vs. true quaternion components")
        ax[0, 0].scatter(quaternions_pred[:, 0], gt_q[:len(quaternions_pred), 0], color='darkblue', alpha=0.8)
        ax[0, 0].set_title(r'$q_0$')
        ax[0, 1].scatter(quaternions_pred[:, 1], gt_q[:len(quaternions_pred), 1], color='darkblue', alpha=0.8)
        ax[0, 1].set_title(r'$q_1$')
        ax[1, 0].scatter(quaternions_pred[:, 2], gt_q[:len(quaternions_pred), 2], color='darkblue', alpha=0.8)
        ax[1, 0].set_title(r'$q_2$')
        ax[1, 1].scatter(quaternions_pred[:, 3], gt_q[:len(quaternions_pred), 3], color='darkblue', alpha=0.8)
        ax[1, 1].set_title(r'$q_3$')
        if save_folder is not None:
            plt.savefig(save_folder + 'quaternions_scatter.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.show()

    timesteps = np.arange(quaternions_pred.shape[0])  # Time indices

    # Plot each quaternion component
    fig, axes = plt.subplots(4, 1, figsize=(7, 4), sharex=True)

    labels = ['w', 'x', 'y', 'z']
    for i in range(4):
        axes[i].plot(timesteps, quaternions_pred[:, i], label='Predicted', linestyle='--', alpha=0.7)
        if gt_q is not None:
            axes[i].plot(timesteps, gt_q[:len(quaternions_pred), i], label='Ground Truth', alpha=0.9)
        axes[i].set_ylabel(labels[i])
        axes[i].legend()
    axes[-1].set_xlabel("Time Step")
    plt.suptitle("Quaternion Components Over Time")
    if save_folder is not None:
        plt.savefig(save_folder + 'quaternions_over_time.png', dpi=dpi, bbox_inches='tight', pad_inches=0)

    plt.show()

    # Convert to Euler angles (XYZ convention)
    #euler_pred = R.from_quat(quaternions_pred).as_euler('xyz', degrees=True)
    #if gt_q is not None:
    #    euler_true = R.from_quat(gt_q[:len(quaternions_pred)]).as_euler('xyz', degrees=True)


    # Unwrap to prevent discontinuities
    #euler_pred = np.unwrap(euler_pred, axis=0)
    #if gt_q is not None:
    #    euler_true = np.unwrap(euler_true, axis=0)

    # Time indices
    #timesteps = np.arange(quaternions_pred.shape[0])

    # Plot each Euler component
    #fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    #labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']

    #for i in range(3):
    #    axes[i].plot(timesteps, euler_pred[:, i], label='Predicted', linestyle='--', alpha=0.7)
    #    if gt_q is not None:
    #        axes[i].plot(timesteps, euler_true[:, i], label='Ground Truth', alpha=0.9)
    #    axes[i].set_ylabel(labels[i])
    #    axes[i].legend()

    #axes[-1].set_xlabel("Time Step")
    #plt.suptitle("Quaternion to Euler Trajectories (Fixed Discontinuities)")
    #if save_folder is not None:
    #    plt.savefig(save_folder + 'euler_over_time.png', dpi=300, bbox_inches='tight', pad_inches=0)
    #plt.show()

    if plot_3d:
        # 3D plot of the predicted object
        visualize_3d_volume(predicted_object.numpy(), save_folder=save_folder)

        # 3D plot of the ground truth object
        if gt_v is not None:
            visualize_3d_volume(gt_v.numpy())


def visualize_3d_volume(volume, pad_remove=2, save_folder=None, sigma=0.8, surface_count=15, opacity=0.5, bgcolor='black', camera_position=(1.25, 1.25, 1.25)):
    """
    Visualizes a 3D volume as an isosurface using Plotly.

    Parameters:
        volume (numpy.ndarray): 3D numpy array (volume) to visualize.
        surface_count (int): Number of isosurfaces to display (default is 15).
        opacity (float): Opacity of the isosurface (default is 0.5).
        bgcolor (str): Background color of the plot (default is 'black').
        camera_position (tuple): Position of the camera in the scene (default is (1.2, 1.2, 1.2)).

    """
    import plotly.graph_objects as go
    from scipy import ndimage

    if pad_remove > 0:
        # Pad the volume with zeros to avoid clipping the isosurface at the edges
        volume = volume[pad_remove:-pad_remove, pad_remove:-pad_remove, pad_remove:-pad_remove]

    # Generate the x, y, and z coordinate grids for the volume
    x = np.arange(volume.shape[0]).repeat(volume.shape[1] * volume.shape[2])  # Repeats each x-value across the grid
    y = np.tile(np.arange(volume.shape[1]).repeat(volume.shape[2]), volume.shape[0])  # Repeats each y-value within z slices
    z = np.tile(np.arange(volume.shape[2]), volume.shape[0] * volume.shape[1])  # Repeats z-values across the full grid

    # Gaussian smoothing for better visualization
    volume = ndimage.gaussian_filter(volume, sigma=sigma)

    # Determine dynamic isosurface bounds based on the volume
    isomin = volume.min() + 0.1 * (volume.max() - volume.min())
    isomax = volume.max() - 0.1 * (volume.max() - volume.min())

    # Create the figure with the 3D isosurface
    fig = go.Figure(data=go.Isosurface(
        x=x,
        y=y,
        z=z,
        value=volume.flatten(),  # Flatten the volume to 1D for visualization
        isomin=isomin,  # Adjust to the volume's range
        isomax=isomax,
        surface_count=surface_count,  # Number of surfaces to display
        opacity=opacity,  # Opacity of the isosurface
        colorscale="Viridis",  # Color scheme
        caps=dict(x_show=False, y_show=False, z_show=False)  # Hide caps for better 3D visualization
    ))

    # Adjust the layout for a more fitting view
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=camera_position[0], y=camera_position[1], z=camera_position[2])  # Adjust the camera's position
            ),
            xaxis=dict(range=[0, volume.shape[0]]),
            yaxis=dict(range=[0, volume.shape[1]]),
            zaxis=dict(range=[0, volume.shape[2]]),
            bgcolor=bgcolor  # Set the background color to black (or any color)
        ),
        margin=dict(l=0, r=0, b=0, t=0)  # Reduce the margins around the plot
    )

    # Make the image overall smaller
    fig.update_layout(width=500, height=500)

    fig.show()

    if save_folder is not None:
        fig.write_html(save_folder + '3d_volume.html')
