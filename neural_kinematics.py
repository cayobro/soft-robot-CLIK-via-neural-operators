import torch
import matplotlib.pyplot as plt

from utils import *

def kinematics_lambda(don, gamma, z):
    gamma = torch.tile(gamma, (len(z), 1)) 
    gamma = scale(gamma, don['scalerX'], inverse=False) 

    z = z.reshape(-1, 1)
    z = scale(z, don['scalerZ'], inverse=False)  

    with torch.no_grad(): # Forward pass
        output = don['best_model'](gamma, z)
    r = scale(output, don['scalerY'], inverse=True)

    return r

def kinematics_lambda_with_jacobian(don, gamma, z, full_jacobians=False):
    z = z.reshape(-1, 1)
    z = scale(z, don['scalerZ'], inverse=False)  
    z = z.requires_grad_(full_jacobians)  # grad only if full jac
    
    if full_jacobians:
        gamma_tiled = torch.tile(gamma, (len(z), 1))
        gamma_tiled = scale(gamma_tiled, don['scalerX'], inverse=False)
        gamma_tiled = gamma_tiled.requires_grad_()

        def model_forward(gamma_in, z_in):
            output = don['best_model'](gamma_in, z_in)
            r = scale(output, don['scalerY'], inverse=True)
            return r

        r = model_forward(gamma_tiled, z)
        jacobian = torch.autograd.functional.jacobian(model_forward, (gamma_tiled, z)) # full jacobian w.r.t (gamma, z)
    
    else:
        gamma = gamma.requires_grad_()
        
        def model_forward(gamma_in):
            # Tile and scale inside for untiled input
            gamma_tiled = torch.tile(gamma_in, (len(z), 1))
            gamma_tiled = scale(gamma_tiled, don['scalerX'], inverse=False)
            
            output = don['best_model'](gamma_tiled, z)
            r = scale(output, don['scalerY'], inverse=True)
            return r
        
        r = model_forward(gamma)
        jacobian = torch.autograd.functional.jacobian(model_forward, gamma)
        
    return r, jacobian

if __name__ == "__main__":
    ## === Load trained operator network ===
    don = load_model()
    
    ## === Test forward pass and jacobian ===
    gamma = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    z = torch.linspace(0, 0.09, steps=100, dtype=torch.float32) 
    true_pick_shape, pick_gamma, z = pick_shape(idx=1203)

    pick_shape = kinematics_lambda(don, pick_gamma, z)
    plot_shape(torch.stack((pick_shape, true_pick_shape), dim=0), [z]*2)

    _, jacobian = kinematics_lambda_with_jacobian(don, gamma, z, full_jacobians=False)
    split_and_visualize_jacobian(jacobian, visualize=True)
