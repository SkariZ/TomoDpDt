import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D

def plots_initial(tomo, gt=None):
    
    z = tomo.latent.detach().cpu().numpy()

    plt.figure(figsize=(4, 4))
    plt.title("Latent space")
    plt.scatter(z[:, 0], z[:, 1], c=np.arange(z.shape[0]))
    plt.scatter(z[0, 0], z[0, 1], c='r')  # start_point
    plt.colorbar()
    plt.show()

    # 3D plot
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z[:, 0], z[:, 1], np.arange(z.shape[0]))
    ax.scatter(z[0, 0], z[0, 1], c='r')
    plt.show()

    # Plot the smoothed distances and peaks
    smoothed_dists = tomo.rotation_initial_dict['smoothed_distances'].cpu().numpy()
    peaks = tomo.rotation_initial_dict['peaks'].cpu().numpy()

    plt.figure(figsize=(4, 4))
    plt.plot(smoothed_dists)
    plt.scatter(peaks, smoothed_dists[peaks], c='r')
    plt.show()

    # Plot the quaternions
    q1 = tomo.rotation_initial_dict['quaternions'].cpu().numpy()

    plt.figure(figsize=(7, 4))
    plt.plot(q1[:, 0], label=r'$q_0$', linewidth=2)
    plt.plot(q1[:, 1], label=r'$q_1$', linewidth=2)
    plt.plot(q1[:, 2], label=r'$q_2$', linewidth=2)
    plt.plot(q1[:, 3], label=r'$q_3$', linewidth=2)
    # Add a vertical line where q1 ends
    plt.axvline(x=len(q1), color='black', linestyle='--', linewidth=3, label='End of q1')
    if gt is not None:
        plt.plot(gt[:, 0], '--', label=r'$q_0$', linewidth=2)
        plt.plot(gt[:, 1], '--', label=r'$q_1$', linewidth=2)
        plt.plot(gt[:, 2], '--', label=r'$q_2$', linewidth=2)
        plt.plot(gt[:, 3], '--', label=r'$q_3$', linewidth=2)
    plt.legend()
    plt.title("Initial Guess vs True Quaternion Components")
    plt.show()

def plots_optim(tomo, gt_q=None, gt_v=None):
    
    predicted_object = tomo.volume.detach().cpu()
    projections_pred = tomo.full_forward().detach().cpu().numpy()
    quaternions_pred = tomo.get_quaternions().detach().cpu().numpy()

    # Plot the Predicted_objects axes
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    #Title
    ax[0].set_title("Sum along x-axis")
    ax[0].imshow(predicted_object.sum(0))
    ax[1].set_title("Sum along y-axis")
    ax[1].imshow(predicted_object.sum(1))
    ax[2].set_title("Sum along z-axis")
    ax[2].imshow(predicted_object.sum(2))
    plt.show()

    if gt_v is not None:
        fig, ax = plt.subplots(1, 3, figsize=(10, 3))
        #Title
        ax[0].set_title("Sum along x-axis")
        ax[0].imshow(gt_v.sum(0))
        ax[1].set_title("Sum along y-axis")
        ax[1].imshow(gt_v.sum(1))
        ax[2].set_title("Sum along z-axis")
        ax[2].imshow(gt_v.sum(2))
        plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(6, 6))
    plt.suptitle("Predicted projections")
    for i in range(3):
        for j in range(3):
            im = ax[i, j].imshow(projections_pred[np.random.randint(0, projections_pred.shape[0])])
            fig.colorbar(im, ax=ax[i, j])
    plt.show()

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
    plt.title("Initial Guess vs True Quaternion Components")
    plt.show()

    # Difference between predicted and true quaternions
    if gt_q is not None:
        diff = quaternions_pred - gt_q[:len(quaternions_pred)].numpy()
        plt.figure(figsize=(7, 4))
        plt.plot(diff[:, 0], label=r'$q_0$', linewidth=2)
        plt.plot(diff[:, 1], label=r'$q_1$', linewidth=2)
        plt.plot(diff[:, 2], label=r'$q_2$', linewidth=2)
        plt.plot(diff[:, 3], label=r'$q_3$', linewidth=2)
        #Add aline at 0
        plt.axhline(y=0, color='black', linestyle='--', linewidth=3)
        plt.legend()
        plt.title("Difference between Predicted and True Quaternion Components")
        plt.show()

    # 2x2 grid of scatter plots for each component of the quaternion
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax[0, 0].scatter(quaternions_pred[:, 0], gt_q[:len(quaternions_pred), 0])
    ax[0, 0].set_title(r'$q_0$')
    ax[0, 1].scatter(quaternions_pred[:, 1], gt_q[:len(quaternions_pred), 1])
    ax[0, 1].set_title(r'$q_1$')
    ax[1, 0].scatter(quaternions_pred[:, 2], gt_q[:len(quaternions_pred), 2])
    ax[1, 0].set_title(r'$q_2$')
    ax[1, 1].scatter(quaternions_pred[:, 3], gt_q[:len(quaternions_pred), 3])
    ax[1, 1].set_title(r'$q_3$')
    plt.show()
