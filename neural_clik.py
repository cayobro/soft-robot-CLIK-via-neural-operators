import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import *
from tasks import *
from neural_kinematics import kinematics_lambda_with_jacobian

columnwidth = 6
textwidth = 8

def clik(t, task, gamma, operator_network, z, target_point):
    shape, jacobian = kinematics_lambda_with_jacobian(operator_network, gamma, z, full_jacobians=False)

    if task['task_variable'] == 'position':
        if task['closest_point'] is None: x, z_star = task_phi_position_opt(shape, target_point)
        else: x, z_star = task_phi_position_fixed(shape, target_point, task['closest_point'])
    elif task['task_variable'] == 'distance':
        raise ValueError('Invalid task variable')
        
    # === CLIK algorithm ===
    jacobian_composite = jacobian[z_star,:,:] 
    j_inv = torch.linalg.inv(jacobian_composite) 
    weight = 8 * torch.diag(torch.tensor([1,1,1], dtype=torch.float32)) # weight matrix for each dimension
    dgamma = - torch.matmul(torch.matmul(j_inv, weight), x.unsqueeze(-1))  # Move to reduce distance

    return dgamma, shape.detach().numpy(), z_star.detach().numpy(), x.detach().numpy()

if __name__ == "__main__":
    ## === Choose a task ===

    ## Positioning the closest point on the shape to the target point 
    task = {'task_variable': 'position', 
            'closest_point': None} 
    
    ## Or: Positioning the tip (or another fixed point on the shape)
    # s_bar = torch.tensor(-1)
    # task = {'task_variable': 'position',
    #         'closest_point': s_bar} 
    
    # Load shape library and pick points from there
    # (makes it easier to compare if we want to position the tip, because then we can compare to the ground truth)
    feasible_shape, pick_gamma, z = pick_shape(idx=568)
    if task['closest_point'] is None: 
        # Choose target point on feasible shape
        target_point = torch.tensor(feasible_shape[70, :], dtype=torch.float32)
    else: 
        # Choose target point on the shape, and specify the index of that point (z_bar) for the fixed-point task
        target_point = torch.tensor(feasible_shape[task['closest_point'], :], dtype=torch.float32)
    
    # or : Manual point
    # target_point = torch.tensor([0.002, 0.002, 0.08], dtype=torch.float32)


    ## === Load operator network ===
    don = load_model()


    ## === Initialize CLIK ===
    t_steps = 1000
    dt = 0.001

    gamma0, shape0 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32), torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=1)

    gammas = np.zeros((t_steps + 1, 3))
    gammas_unclamped = np.zeros((t_steps + 1, 3))
    shapes = np.zeros((t_steps + 1, len(z), 3))
    z_stars = np.zeros(t_steps, dtype=np.int64) 
    xs = np.zeros((t_steps, 3))
    time = np.arange(t_steps) * dt

    gammas[0] = gamma0.numpy()
    gammas_unclamped[0] = gamma0.numpy()
    shapes[0] = shape0.numpy()

    gamma = gamma0.clone()

    ## === Run the simulation ===
    for t in range(t_steps):
        dot_gamma, shape, z_star, x = clik(t, task, gamma, don, z, target_point) 
        gamma = gamma + dt * dot_gamma.squeeze()  

        gammas_unclamped[t+1] = gamma.detach().numpy()
        gamma = torch.clamp(gamma, min=-1.6, max=0.0)
        gammas[t+1] = gamma.detach().cpu().numpy()
        
        shapes[t+1] = shape
        z_stars[t] = z_star
        xs[t] = x


    # === Plotting ===
    fig = plt.figure(figsize=(textwidth, columnwidth * 0.8))
    ax = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 3), fig.add_subplot(1, 2, 2, projection='3d')]

    colors = [get_color('illuminating'), get_color('bay'), get_color('plum')]
    labels = ['$x$', '$y$', '$z$']

    for i in range(3):
        ax[0].plot(time, gammas[1:, i], label=f'$\gamma_{{{i+1}}}$', color=colors[i])
    ax[0].set_ylabel('actuation')
    ax[0].legend()

    for i in range(3):
        ax[1].plot(time, xs[:, i], label=labels[i], color=colors[i])
    ax[1].set_ylabel('distance')
    ax[1].set_xlabel('time')
    ax[1].legend()

    if task['closest_point'] is None:
        ax_min = ax[1].twinx()
        ax_min.set_ylabel('index $s_*$')
        ax_min.plot(time, z_stars, color=get_color('poppy'), label='$s_*$')
        ax_min.legend(loc='lower center')

    for shape, z_star in zip(shapes[::10], z_stars[::10]):
        ax[2].plot(shape[:, 0], shape[:, 1], shape[:, 2], 
                color=get_color('lagunita'), alpha=0.4, linewidth=1)
        ax[2].plot(shape[z_star, 0], shape[z_star, 1], shape[z_star, 2], 
                color=get_color('poppy'), marker='.', markersize=2)

    ax[2].plot(shapes[0][:, 0], shapes[0][:, 1], shapes[0][:, 2], 
            '--', color='black', linewidth=1.5)
    ax[2].plot(shapes[-1][:, 0], shapes[-1][:, 1], shapes[-1][:, 2], 
            '-', color='black', linewidth=1.5)

    target_np = target_point.detach().cpu().numpy()
    ax[2].plot(target_np[0], target_np[1], target_np[2], 
            color=get_color('cardinal'), markersize=8, marker='.')

    ax[2].scatter([], [], color=get_color('cardinal'), marker='.', s=8, label='Target')
    ax[2].scatter([], [], color=get_color('poppy'), marker='.', s=2, label='$s_*$')

    ax[2].set_zlim(ax[2].get_zlim()[::-1])
    ax[2].set_xlabel('$x$'); ax[2].set_ylabel('$y$'); ax[2].set_zlabel('$z$')
    ax[2].legend()
    fig.tight_layout()


