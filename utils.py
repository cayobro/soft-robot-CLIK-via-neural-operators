import numpy as np
import torch
import os
import joblib
import a2s
import matplotlib.pyplot as plt
import json

def load_model(orientation=False):
    don = dict()
    experiment_name = "a2s_don/run_001"
    repo_dir = 'XXX' # TODO: adapt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(repo_dir, "training-results", experiment_name)

    ### ==== Load model, metadata, and scalers ====
    metadata_file = os.path.join(run_dir, "metadata.json")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    don['best_model'] = a2s.DeepONet(params=metadata)

    state_dict_file = os.path.join(run_dir, "best_model")
    don['best_model'].load_state_dict(torch.load(state_dict_file, map_location=device))
    don['best_model'].to(device)
    don['best_model'].eval()

    don['scalerX'] = joblib.load(run_dir + "/scalerX.save") 
    don['scalerY'] = joblib.load(run_dir + "/scalerY.save") 
    don['scalerZ'] = joblib.load(run_dir + "/scalerZ.save")
    return don

def load_shape_library(orientation=False):
    base_path= 'XXX' #TODO: Path to dataset
    gamma_data = np.load(base_path + "gamma.npz")
    z_data = np.load(base_path + "z.npz", allow_pickle=True)
    if orientation: r_data = np.load(base_path + "with_orientation_" + "r.npz", allow_pickle=True)
    else: r_data = np.load(base_path + "r.npz", allow_pickle=True)
    gamma = gamma_data["gamma"]
    r = r_data["r"]
    z_raw = z_data["z"]
    z = z_raw[0,:]
    return r, gamma, z

def scale(data, scaler, inverse=False):
    mean = torch.tensor(scaler.mean_, dtype=torch.float32)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32)
    if inverse:
        scaled_data = data * scale + mean
    else:
        scaled_data = (data - mean) / scale 
        scaled_data = scaled_data.unsqueeze(0)
    return scaled_data

def split_and_visualize_jacobian(jacobian, visualize=False):
    output_names = ['r_x', 'r_y', 'r_z']
    input_names = ['gamma1', 'gamma2', 'gamma3']
    
    jac_dict = {}
    for j, out_name in enumerate(output_names):
        for k, in_name in enumerate(input_names):
            jac_dict[f'dr_{out_name}_d{in_name}'] = jacobian[:, j, k]
    
    if visualize:
        jac_np = jacobian.detach().cpu().numpy()
        locations = np.arange(jac_np.shape[0])
        fig, axs = plt.subplots(3, 3, figsize=(10, 7), sharex=True)
        for j in range(3):
            for k in range(3):
                axs[j, k].plot(locations, jac_np[:, j, k])
                axs[j, k].set_title(f'∂r_{j+1}/∂γ_{k+1}')
                axs[j, k].set_xlabel('z index')
        plt.tight_layout()
        plt.show()
    return jac_dict

def cc_tip_point(q):
    """
    Choose a target point that is actually feasible for the segment.
    q: scalar tensor (curvature), q != 0
    returns: tensor([x, y]) tip position
    """
    R = 1.0 / torch.abs(q)            # radius
    cy = torch.sign(q) * R            # circle center y-coordinate

    theta = q                         # since L = 1, theta = q * L = q
    x = R * torch.sin(theta)
    y = cy - R * torch.cos(theta)

    return torch.stack([x, y])

def pick_shape(idx, orientation=False):
    shape_lib = dict()
    shape_lib['r'], shape_lib['gamma'], shape_lib['z'] = load_shape_library(orientation=orientation)
    pick_shape = torch.tensor(shape_lib['r'][idx,:,:], dtype=torch.float32)
    pick_gamma = torch.tensor(shape_lib['gamma'][idx,:], dtype=torch.float32)
    z = torch.tensor(shape_lib['z'], dtype=torch.float32)
    return pick_shape, pick_gamma, z

def plot_shape(rs, zs, task=None, z_star=None):
    """
    Plot multiple shapes in the same figure.

    Parameters:
        rs (list of np.ndarray): List of predicted shapes, where each r is of shape (n_points, 3).
        zs (list of np.ndarray): List of corresponding z values, where each z is of shape (n_points,).
    """
    fig = plt.figure()
    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')  # Right column
    ax_2d = [fig.add_subplot(3, 2, 1), fig.add_subplot(3, 2, 3), fig.add_subplot(3, 2, 5)]

    # Iterate over the list of r's and z's
    for i, (r, z) in enumerate(zip(rs, zs)):
        if isinstance(r, torch.Tensor):
            r = r.detach().cpu().numpy()
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
    
        # Plot the 3D trajectory
        ax_3d.plot(r[:, 0], r[:, 1], r[:, 2], label=f"Shape {i+1}", linestyle="-")

        # Mark the starting and ending points
        ax_3d.scatter(r[0, 0], r[0, 1], r[0, 2], color="black", s=20)
        if z_star is None:
            ax_3d.scatter(r[-1, 0], r[-1, 1], r[-1, 2], color="red", s=20)
        else:
            if i>1:
                ax_3d.scatter(r[z_star[i-2], 0], r[z_star[i-2], 1], r[z_star[i-2], 2], color="red", s=10)

        # Plot the 2D components for each r
        components = ["r1(z)", "r2(z)", "r3(z)"]
        for j in range(3):
            ax_2d[j].plot(z, r[:, j], label=f"r {i+1}", linestyle="-") #, marker='.')
            ax_2d[j].set_xlabel("z")
            ax_2d[j].set_ylabel(components[j])
            # ax_2d[j].legend()
            ax_2d[j].grid(True)

    if task is not None:
        x0 = task
        # point = task[1]
        if isinstance(x0, torch.Tensor):
            x0 = x0.detach().cpu().numpy()
        # if isinstance(point, torch.Tensor):
            # point = point.detach().cpu().numpy()
        ax_3d.scatter(x0[0], x0[1], x0[2], color="green", s=20, label="Task Point x0")
        # ax_3d.scatter(point[0], point[1], point[2], color="blue", s=20, label="Closest Point on Shape")

    # Adjust the orientation: invert the z-axis to make (0, 0, 1) the highest point
    ax_3d.set_zlim(ax_3d.get_zlim()[::-1])  # Reverse the z-axis limits
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

def get_color(color_name):
    colors = {
        "cardinal": [157/255, 34/255, 53/255],
        "palo": [0/255, 106/255, 82/255],
        "palo verde": [39/255, 153/255, 137/255],
        "olive": [143/255, 153/255, 62/255],
        "bay": [111/255, 162/255, 135/255],
        "sky": [66/255, 152/255, 181/255],
        "lagunita": [0/255, 124/255, 146/255],
        "poppy": [233/255, 131/255, 0],
        "plum": [98/255, 0, 89/255],
        "illuminating": [254/255, 197/255, 29/255],
        "spirited": [224/255, 79/255, 57/255],
        "brick": [101/255, 28/255, 50/255],
        "archway": [93/255, 75/255, 60/255]
    }
    return colors.get(color_name.lower(), [0, 0, 0])  # Default to black if color not found
