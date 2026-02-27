import numpy as np
import torch
import matplotlib.pyplot as plt

def cc_kinematics_pi(q, s, L):
    if q == 0:
        x = L * s
        y = torch.zeros_like(s)
        alpha = torch.zeros_like(s)
    else:
        x = L*torch.sin(s*q) / (q)
        y = L*(1 - torch.cos(s*q)) / (q)
        alpha = s*q
    shape = torch.stack([x,y], dim=1)
    return shape, alpha

def cc_kinematics_jacobian(q, s, L):
    if q == 0:
        dxdq = 0.0 * torch.ones_like(s)
        dydq = L * s**2 / 2
        dadq = s
    else:
        dxdq = L * (s * q * torch.cos(s * q) - torch.sin(s * q)) / (q ** 2)
        dydq = L * (s * q * torch.sin(s * q) + torch.cos(s * q) - 1) / (q ** 2)
        dadq = s 
    return torch.stack([dxdq, dydq], dim=1)


if __name__ == "__main__":
    L = 1 # length of the segment
    s = torch.linspace(0, 1, steps=100)
    q = torch.tensor(2, dtype=torch.float32)
    shape, alpha = cc_kinematics_pi(q, s, L)
    J = cc_kinematics_jacobian(q, s, L)
    shape0 = torch.stack([s, torch.zeros_like(s)], dim=1)

    ## === Some plots to verify the kinematics and jacobian ===
    plt.figure()
    plt.plot(shape[:,0].numpy(), shape[:,1].numpy(), label='test shape')  
    plt.plot(shape0[:,0].numpy(), shape0[:,1].numpy(), '--', label='initial shape')
    plt.legend()

    plt.figure()
    plt.plot(J[:,0].numpy(), label='dx/dq')
    plt.plot(J[:,1].numpy(), label='dy/dq')
    plt.legend()
    plt.show()
    a=0