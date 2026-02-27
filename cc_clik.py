import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import *
from tasks import *
from cc_kinematics import cc_kinematics_pi, cc_kinematics_jacobian

columnwidth = 6
textwidth = 8

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


def clik(t, task, q, s, target_point):
    shape, a = cc_kinematics_pi(q, s, L)
    jacobian = cc_kinematics_jacobian(q, s, L)
    
    if task['task_variable'] == 'distance':
        if task['closest_point'] is None: x, z_star = task_phi_distance_opt(shape, target_point)
        else: x, z_star = task_phi_distance_fixed(shape, target_point, task['closest_point'])
    else: raise ValueError("Invalid task variable")
    
    # === CLIK algorithm ===
    j_g1 = torch.matmul(jacobian[z_star], (shape[z_star] - target_point)) 
    j_inv = 1/j_g1
    weight = 10
    dgamma = weight * j_inv * ( - x)
    return dgamma, shape, z_star, x


if __name__ == "__main__":
    ## === Choose a task ===
    
    ## Positioning the closest point on the shape to the target point 
    task = {'task_variable': 'distance',
            'closest_point': None}

    ## Or: Positioning the tip (or another fixed point on the shape)
    # task = {'task_variable': 'distance',
    #         'closest_point': torch.tensor(-1)} 

    s = torch.linspace(0, 1, steps=100)
    if task['closest_point'] is None: 
        # Choose target point in workspace
        target_point = torch.tensor([0.2, 0.3], dtype=torch.float32) 
    else: 
        # Choose target point on the shape, and specify the index of that point (z_bar) for the fixed-point task
        target_point = cc_tip_point(torch.tensor(2.0)); z_bar = -1 


    ## === Initialize CLIK ===
    t_steps = 1000
    dt = 0.001
    L = 1 # length of the segment
   
    q0, shape0 = torch.tensor(0.0, dtype=torch.float32), torch.stack([s, torch.zeros_like(s)], dim=1)
    qs = np.zeros((t_steps + 1, 1))
    shapes = np.zeros((t_steps + 1, len(s), 2))
    z_stars = np.zeros(t_steps, dtype=np.int64) 
    xs = np.zeros((t_steps, 1))
    time = np.arange(t_steps) * dt

    qs[0] = q0.numpy()
    shapes[0] = shape0.numpy()
    q = q0.clone()

    ## === Run the simulation ===
    i = 0
    for t in range(t_steps):
        dot_q, shape, z_star, x = clik(t, task, q, s, target_point) 
        q = q + dt * dot_q.squeeze()  # Update gamma using Euler's method

        qs[i+1] = q.detach().numpy()
        shapes[i+1] = shape.detach().numpy()
        z_stars[i] = z_star.detach().numpy()
        xs[t] = x
        i = i + 1


# === Plotting ===
fig, (ax1, ax3) = plt.subplots(2,1, height_ratios=[1, 3])
ax1.plot(time, qs[1:], color=get_color('palo'))
ax1.set_ylabel('curvature $q$')
ax_min = ax1.twinx()
ax_min.plot(time, z_stars, color=get_color('poppy'), label='$s_*$')
ax_min.set_ylabel('index $s_*$')
ax1.plot([], [], color=get_color('poppy'), label='$s_*$')  # dummy for legend
ax1.plot([], [], color=get_color('palo'), label='$q$')  # dummy for legend
ax1.legend(loc='lower right')

i = 0
for shape, z in zip(shapes[1::10], z_stars[::10]):
    ax3.plot(shape[:,0], shape[:,1], color=get_color('lagunita'), alpha=0.5, linewidth=1) # , c=cmap(time[i])
    ax3.plot(shape[z,0], shape[z,1], color=get_color('poppy'), marker='.', markersize=2)
    i = i + 1
ax3.plot(shape0[:,0], shape0[:,1], '--', color='black', linewidth=1.5, label='Initial')
ax3.plot(shapes[-1][:,0], shapes[-1][:,1], '-', color='black', linewidth=1.5, label='Final')
ax3.plot(target_point[0].numpy(), target_point[1].numpy(), color=get_color('cardinal'), marker='.', markersize=8)
ax3.scatter([], [], color=get_color('cardinal'), marker='.', s=8, label='Target Point')  # dummy for legend
ax3.scatter([], [], color=get_color('poppy'), marker='.', s=2, label='$s_*$')  # dummy for legend
ax3.plot([], [], color=get_color('lagunita'), alpha=0.4, linewidth=1)  # dummy for legend
ax3.set_xlabel('$x$'); ax3.set_ylabel('$y$'); ax3.legend()
fig.set_size_inches(columnwidth, columnwidth)
fig.tight_layout()

plt.show()
