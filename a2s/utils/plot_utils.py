import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_input_output_distributions(X_train, X_test, Y_train, Y_test):
    """
    Plot overlaid distributions of unscaled input and output data.
    """
    if X_train.shape[1] == 3: # learned forward mapping
        # Reshape inputs to (n_trajectories, n_z, 3)
        n_z = Y_train.shape[1] // 3  # Assuming X_train is of shape (n_trajectories, n_z * 3)
        Y_train_reshaped = Y_train.reshape((Y_train.shape[0], n_z, 3))
        Y_test_reshaped = Y_test.reshape((Y_test.shape[0], n_z, 3))

        # Extract the last entry along z dimension (tip data)
        tip_train = Y_train_reshaped[:, -1, :]  # shape: (n_trajectories, 3)
        tip_test = Y_test_reshaped[:, -1, :]    # shape: (n_trajectories, 3)

        # Create dataframes
        col_names = ["act1", "act2", "act3", "tip_x", "tip_y", "tip_z"]
        df_train = pd.DataFrame(np.hstack((X_train, tip_train)), columns=col_names)
        df_test = pd.DataFrame(np.hstack((X_test, tip_test)), columns=col_names)
    elif Y_train.shape[1] == 3: # learned inverse mapping
        n_z = X_train.shape[1] // 3  # Assuming X_train is of shape (n_trajectories, n_z * 3)
        X_train_reshaped = X_train.reshape((X_train.shape[0], n_z, 3))
        X_test_reshaped = X_test.reshape((X_test.shape[0], n_z, 3))

        # Extract the last entry along z dimension (tip data)
        tip_train = X_train_reshaped[:, -1, :]  # shape: (n_trajectories, 3)
        tip_test = X_test_reshaped[:, -1, :]    # shape: (n_trajectories, 3)

        # Create dataframes
        col_names = ["tip_x", "tip_y", "tip_z", "act1", "act2", "act3"]
        df_train = pd.DataFrame(np.hstack((tip_train, Y_train)), columns=col_names)
        df_test = pd.DataFrame(np.hstack((tip_test, Y_test)), columns=col_names)

    # Set up plot grid
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    input_cols = col_names[:3]
    output_cols = col_names[3:]

    # Plot inputs (top row)
    for i, col in enumerate(input_cols):
        sns.histplot(df_train[col], ax=axes[0, i], bins=50, kde=True, color="blue", label="Train", stat="density", alpha=0.5)
        sns.histplot(df_test[col], ax=axes[0, i], bins=50, kde=True, color="orange", label="Test", stat="density", alpha=0.5)
        axes[0, i].set_title(col)
        axes[0, i].legend()

    # Plot outputs (bottom row)
    for i, col in enumerate(output_cols):
        sns.histplot(df_train[col].dropna(), ax=axes[1, i], bins=50, kde=True, color="blue", label="Train", stat="density", alpha=0.5)
        sns.histplot(df_test[col].dropna(), ax=axes[1, i], bins=50, kde=True, color="orange", label="Test", stat="density", alpha=0.5)
        axes[1, i].set_title(col)
        axes[1, i].legend()

    fig.suptitle("Train vs Test Distributions: Inputs (Top) and Outputs (Bottom)", fontsize=14)
    plt.tight_layout()
    plt.show(block=False)

def plot_pytorch_training(history, save_path=None, params=None):
    """
    Plot the training history, including train loss, validation loss, and optionally metrics.

    Args:
        history (dict): Dictionary containing training history with the following keys:
            - "train_losses" (list): Training loss values for each epoch.
            - "valid_losses" (list): Validation loss values for each epoch.
            - "epochs" (list): List of epoch numbers.
        save_path (str, optional): Path to save the plot as a PNG file. If None, the plot is not saved.
        params (dict, optional): Dictionary containing additional training parameters, such as metrics.

    Behavior:
        - Creates a plot with training and validation losses over epochs.
        - Uses a logarithmic scale for the y-axis to improve visibility of loss trends.
        - Optionally includes metrics if provided in the `params` dictionary.

    Output:
        - Displays the plot.
        - Saves the plot to `save_path + "/training_curve.png"` if `save_path` is provided.

    Example:
        >>> plot_pytorch_training(history, save_path="results", params={"metrics": ["l2 relative error"]})
    """
    train_losses = history["train_losses"]
    valid_losses = history["valid_losses"]
    # metrics_history = history["metrics_history"]
    # learning_rates = history["learning_rates"]
    epochs = history["epochs"]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot training and validation losses
    plt.plot(epochs, train_losses, label="Train Loss", linestyle="-")
    plt.plot(epochs, valid_losses, label="Val Loss", linestyle="-")
    plt.yscale("log")  # Use logarithmic scale for better visibility

    # Add labels, legend, and title
    plt.xlabel("Epochs")
    plt.ylabel("Loss / Metric")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    if not save_path is None:
        plt.savefig(save_path + "/training_curve.png")

def plot_prediction(Y_test, Y_pred, X_test=None, X_pred=None, sample_idx=42, save_path=None, nz=100):
    """
    TYPES: FNN, DON,
    """

    ns = Y_test.shape[0] // nz
    Y_pred = Y_pred.reshape(ns, nz, 3)
    Y_test = Y_test.reshape(ns, nz, 3)
    z = np.linspace(0, 1, Y_test.shape[1])  # nz evenly spaced points between 0 and 1

    # Scatter plot: True vs. Predicted for all samples at a few specific z-locations
    z_locations = [0, int(nz/2), nz-1]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle("True vs. Predicted at Multiple z-Locations", fontsize=16)

    for i, z_loc in enumerate(z_locations):
        for j in range(3):  # Loop over the 3 vector components
            ax = axes[j, i]
            ax.scatter(Y_test[:, z_loc, j],Y_pred[:, z_loc, j], alpha=0.6, label=f"z={z_loc}")
            ax.plot(
                [Y_test[:, z_loc, j].min(), Y_test[:, z_loc, j].max()],
                [Y_test[:, z_loc, j].min(), Y_test[:, z_loc, j].max()],
                "k--", label="Ideal"
            )
            ax.set_xlabel("True Value")
            ax.set_ylabel("Predicted Value")
            ax.set_title(f"r[{j+1}] at z={z_loc}")
            ax.legend()
            ax.grid(True)
    plt.tight_layout()  # Adjust layout to fit the title
    plt.show(block=False)
    plt.pause(0.001)
    if not save_path is None:
        plt.savefig(save_path + "/test_results.png")
    # plt.close(fig)

    # 3D plot: r(z) curve for a single sample
    Y_test_sample = Y_test[sample_idx]  # Shape: (nz, 3)
    Y_pred_sample = Y_pred[sample_idx]  # Shape: (nz, 3)

    # Create a combined figure with 3D plot and 2D plots
    fig = plt.figure(figsize=(14, 10))

    # 3D plot: r(z) curve for a single sample
    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')  # Right column
    ax_3d.plot(Y_test_sample[:, 0], Y_test_sample[:, 1], Y_test_sample[:, 2], label="Ground Truth", linestyle="-", color="blue")
    ax_3d.plot(Y_pred_sample[:, 0], Y_pred_sample[:, 1], Y_pred_sample[:, 2], label="Prediction", linestyle="--", color="orange")

    # Mark the starting point r(z=0) with a black circle
    ax_3d.scatter(Y_test_sample[0, 0], Y_test_sample[0, 1], Y_test_sample[0, 2], color="black", label="Start (r(z=0))", s=50)
    # Mark the endpoint r(z=1) with a red circle
    ax_3d.scatter(Y_test_sample[-1, 0], Y_test_sample[-1, 1], Y_test_sample[-1, 2], color="red", label="End (r(z=1))", s=50)

    # Adjust the orientation: invert the z-axis to make (0, 0, 1) the highest point
    ax_3d.set_zlim(ax_3d.get_zlim()[::-1])  # Reverse the z-axis limits

    # Add labels, legend, and title
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.set_title(f"3D Curve of r(z) for Sample {sample_idx}")
    ax_3d.legend()

    # 2D plots: r1(z), r2(z), and r3(z) over z for a single sample
    components = ["r1(z)", "r2(z)", "r3(z)"]
    for i in range(3):
        ax_2d = fig.add_subplot(3, 2, 2 * i + 1)  # Left column
        ax_2d.plot(z, Y_test_sample[:, i], label="Ground Truth", linestyle="-", color="blue")
        ax_2d.plot(z, Y_pred_sample[:, i], label="Prediction", linestyle="--", color="orange")
        ax_2d.set_xlabel("z")
        ax_2d.set_ylabel(components[i])
        ax_2d.legend()
        ax_2d.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    if not save_path is None:
        plt.savefig(save_path + f"/sample_{sample_idx}_prediction.png")
    # plt.close(fig)


