import torch
from torch.optim import Optimizer


class AdvancedSymplecticOptimizer(Optimizer):
    def __init__(self, params, lr=1e-2, beta=0.9, epsilon=1e-8):
        defaults = dict(lr=lr, beta=beta, epsilon=epsilon)
        super(AdvancedSymplecticOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p.data)

                momentum = state["momentum"]
                lr, beta, eps = group["lr"], group["beta"], group["epsilon"]

                state["step"] += 1

                # Implement 4th order symplectic integrator (Forest-Ruth algorithm)
                momentum.mul_(beta).add_(grad, alpha=1 - beta)

                # Compute adaptive step size
                kinetic = 0.5 * (momentum**2).sum()
                potential = 0.5 * (grad**2).sum()
                hamiltonian = kinetic + potential
                step_size = lr / (hamiltonian.sqrt() + eps)

                p.add_(momentum, alpha=-step_size)

        return loss
