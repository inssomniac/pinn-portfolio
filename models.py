import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import grad

def exact_solution(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return torch.sin(2 * np.pi * x) * torch.sin(3 * np.pi * y)

def source_term(xy):
    return -13.0 * np.pi**2 * exact_solution(xy)

def laplacian(u, xy):
    du_dxy = grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx = du_dxy[:, 0:1]
    du_dy = du_dxy[:, 1:2]
    d2u_dx2 = grad(du_dx, xy, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0][:, 0:1]
    d2u_dy2 = grad(du_dy, xy, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0][:, 1:2]
    return d2u_dx2 + d2u_dy2

def compute_loss(model, interior, boundary, w_f, w_b):
    u_int = model(interior)
    lap_u = laplacian(u_int, interior)
    loss_pde = torch.mean((lap_u - source_term(interior))**2)
    u_bc = model(boundary)
    loss_bc = torch.mean((u_bc - exact_solution(boundary))**2)
    return w_f * loss_pde + w_b * loss_bc, loss_pde, loss_bc


class PINN(nn.Module):
    """Стандартный MLP с активацией Tanh."""
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class PINNGELU(nn.Module):
    """MLP с активацией GELU."""
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.GELU()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class SIREN(nn.Module):
    """Использует sin как активацию с масштабирующим параметром omega_0."""
    def __init__(self, layers, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            if i == 0:
                nn.init.uniform_(layer.weight, -1 / layers[i], 1 / layers[i])
            else:
                bound = math.sqrt(6 / layers[i]) / omega_0
                nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.sin(self.omega_0 * x)
        return x


class FourierPINN(nn.Module):
    """MLP с Fourier Feature Mapping (Tancik et al., 2020)"""
    def __init__(self, layers, n_fourier=64, sigma=1.0):
        super().__init__()
        B = torch.randn(2, n_fourier) * sigma
        self.register_buffer('B', B)
        self.activation = nn.Tanh()
        mlp_sizes = [2 * n_fourier] + layers[1:]
        self.mlp = nn.ModuleList()
        for i in range(len(mlp_sizes) - 1):
            self.mlp.append(nn.Linear(mlp_sizes[i], mlp_sizes[i + 1]))

    def forward(self, x):
        proj = 2 * np.pi * (x @ self.B)
        x = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i < len(self.mlp) - 1:
                x = self.activation(x)
        return x


class HardBCPINN(nn.Module):
    """MLP с жёсткими граничными условиями Дирихле."""
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, xy):
        x_in = xy
        for i, layer in enumerate(self.layers):
            x_in = layer(x_in)
            if i < len(self.layers) - 1:
                x_in = self.activation(x_in)
        phi = xy[:, 0:1] * (1 - xy[:, 0:1]) * xy[:, 1:2] * (1 - xy[:, 1:2])
        return x_in * phi
