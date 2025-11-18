import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CvxModel(nn.Module):
    """
    Two-layer ICNN (input convex neural network) block.
    """
    def __init__(self, n_feature=2, n_hidden=4, n_output=1):
        super(CvxModel, self).__init__()
        # Non-negative weights to preserve convexity
        self.input_layer = nn.Linear(n_feature, n_hidden, bias=False)
        self.hidden_layer = nn.Linear(n_hidden, n_hidden, bias=False)
        self.output_layer = nn.Linear(n_hidden, n_output, bias=False)

        # Passthrough channels (with bias) ensure 0-convexity
        self.passthrough_layer = nn.Linear(n_feature, n_hidden)
        self.passthrough_output_layer = nn.Linear(n_feature, n_output)

        self.activation = F.softplus

    def forward(self, x):
        z1 = self.activation(self.input_layer(x))
        skip1 = self.passthrough_layer(x)

        z2 = self.activation(self.hidden_layer(z1) + skip1)
        skip2 = self.passthrough_output_layer(x)

        out = self.output_layer(z2) + skip2
        return out


class LyapunovNet(nn.Module):
    """
    V(x) = sum_i alpha_i * phi_i(x)^2 + softplus(ICNN(phi(x)))
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super(LyapunovNet, self).__init__()

        self.ICNN = CvxModel(input_dim, hidden_dim, output_dim)

        # Learnable positive weights α_i enforced via Softplus
        self.raw_alpha = nn.Parameter(torch.ones(input_dim))

        self.activation = F.softplus
        
    def psi(self, z):
        """Return the ICNN output psi(phi(x))."""
        return self.activation(self.ICNN(z))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: Koopman features phi(x)
        return: scalar Lyapunov value V(x)
        """
        return torch.sum(F.softplus(self.raw_alpha) * (z ** 2), dim=1, keepdim=True) + self.activation(self.ICNN(z))


def hessian_penalty_psi(lyap_net: LyapunovNet,
                        z: torch.Tensor,
                        eps: float = 1e-2,
                        num_samples: int = 1) -> torch.Tensor:
    """Approximate the Frobenius norm of the Hessian of psi(z)."""
    total_penalty = 0.0
    for _ in range(num_samples):
        noise = torch.randn_like(z)
        z_plus = (z.detach() + eps * noise).requires_grad_(True)
        z_minus = (z.detach() - eps * noise).requires_grad_(True)

        psi_plus = lyap_net.psi(z_plus).sum()
        psi_minus = lyap_net.psi(z_minus).sum()

        grad_plus = torch.autograd.grad(psi_plus, z_plus, create_graph=True)[0]
        grad_minus = torch.autograd.grad(psi_minus, z_minus, create_graph=True)[0]

        diff = grad_plus - grad_minus
        pen_k = (diff.pow(2).sum(dim=1)).mean()
        total_penalty = total_penalty + pen_k

    return total_penalty / num_samples


def lyapunov_loss(lyap_net: LyapunovNet,
                  V: torch.Tensor,
                  z: torch.Tensor,
                  grad: torch.Tensor,
                  alpha: float = 1.0,
                  hess_weight: float = 0.0,
                  hess_eps: float = 1e-2,
                  hess_samples: int = 1) -> torch.Tensor:
    """
    Base loss = max(0, V̇) + positive-definiteness prior.

    Parameters
    ----------
    lyap_net : LyapunovNet
        Network used to compute psi(z) for the Hessian penalty.
    V : torch.Tensor
        Lyapunov evaluation V(z).
    z : torch.Tensor
        Koopman feature vector with requires_grad=True.
    grad : torch.Tensor
        Estimated time derivative of the Koopman features.
    alpha : float
        Positive-definiteness margin.
    hess_weight : float
        Weight on the Hessian penalty term (0 disables it).
    hess_eps : float
        Finite-difference epsilon for the Hessian penalty.
    hess_samples : int
        Number of random directions sampled for the Hessian penalty.
    """
    grad_V = torch.autograd.grad(V.sum(), z, create_graph=True)[0]

    V_dot = torch.sum(grad_V * grad, dim=1, keepdim=True)

    loss_neg = torch.mean(torch.relu(V_dot+1e-3))

    loss_pos = torch.mean(torch.relu(alpha*torch.norm(z, dim=1, keepdim=True) ** 2 - V))

    loss = loss_neg + loss_pos

    if hess_weight > 0.0:
        hess_pen = hessian_penalty_psi(lyap_net, z, eps=hess_eps, num_samples=hess_samples)
        loss = loss + hess_weight * hess_pen

    return loss
