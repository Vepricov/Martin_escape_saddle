import torch
from torch.optim import Optimizer

class DiagonalPreconditionedOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8, update_freq=1):
        """
        Optimizer that uses diagonal preconditioning with the Hessian's diagonal.
        
        Args:
            params: model parameters
            lr (float): learning rate
            eps (float): small constant for numerical stability
            update_freq (int): frequency of diagonal updates (in steps)
        """
        defaults = dict(lr=lr, eps=eps)
        super(DiagonalPreconditionedOptimizer, self).__init__(params, defaults)
        
        # Get total number of parameters
        self.num_params = sum(p.numel() for p in self.param_groups[0]['params'])
        self.update_freq = update_freq
        self.step_count = 0
        
        # Initialize diagonal elements storage
        self.diagonal = None
        
    def _get_basis_vector(self, idx, device):
        """Creates a basis vector e_i with 1 at position idx and 0 elsewhere."""
        e_i = torch.zeros(self.num_params, device=device)
        e_i[idx] = 1.0
        return e_i

    def step(self, closure):
        """Performs a single optimization step.
        
        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """
        # Get parameters
        params = []
        for group in self.param_groups:
            params.extend(group['params'])

        # First closure call to get initial loss and gradients
        loss = closure()  # This will do zero_grad and backward with retain_graph=True

        # Update diagonal elements periodically
        if self.diagonal is None or self.step_count % self.update_freq == 0:
            # Calculate Jacobian of loss w.r.t. model parameters
            J = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            J = torch.cat([e.flatten() for e in J])
            
            # Initialize diagonal elements storage
            diagonal = torch.zeros_like(J)
            
            # Compute diagonal elements using basis vectors
            for i in range(self.num_params):
                e_i = self._get_basis_vector(i, J.device)
                HVP = torch.autograd.grad(J, params, e_i, 
                                        retain_graph=(i < self.num_params-1))
                HVP = torch.cat([e.flatten() for e in HVP])
                diagonal[i] = HVP[i]
            
            # Store the computed diagonal
            self.diagonal = torch.max(torch.abs(diagonal), torch.ones_like(J) * self.param_groups[0]['eps'])

        # Update parameters using diagonal preconditioning
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Get the corresponding diagonal elements for this parameter
                numel = p.numel()
                d = self.diagonal[offset:offset + numel].reshape(p.shape)
                
                # Update with diagonal preconditioning: p_new = p_old - lr * D^{-1} * grad
                p.data.addcdiv_(p.grad, d, value=-group['lr'])
                
                offset += numel
        
        self.step_count += 1
        return loss