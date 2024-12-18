import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor
from typing import Any, Callable, Dict, Iterable, Optional, Union

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]
LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
OptFloat = Optional[float]

def _matrix_power(matrix: torch.Tensor, power: float, use_cpu=False) -> torch.Tensor:
    # use CPU for svd for speed up
    device = matrix.device
    if use_cpu: matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return u.to(device), s.pow_(power).diag().to(device), v.t().to(device)
    # return (u @ s.pow_(power).diag() @ v.t()).to(device)


class NEW_SOAP(Optimizer):
    r"""Implements NEW SOAP Algorithm .

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
        epsilon: epsilon added to each mat_gbar_j for numerical stability
            (default: 1e-4)
        update_freq: update frequency to compute inverse (default: 1)


    Note:
        Reference code: https://github.com/moskomule/shampoo.pytorch
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-1,
        momentum: float = 0.0,
        prec_momentum: float = None,
        weight_decay: float = 0.0,
        epsilon: float = 1e-4,
        update_freq: int = 1,
    ):

        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if momentum < 0.0:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if epsilon < 0.0:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if update_freq < 1:
            raise ValueError('Invalid momentum value: {}'.format(momentum))

        p_mom = prec_momentum if prec_momentum is not None else momentum
        defaults = dict(
            lr=lr,
            momentum=momentum,
            prec_momentum=p_mom,
            weight_decay=weight_decay,
            epsilon=epsilon,
            update_freq=update_freq,
        )
        super(NEW_SOAP, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]
                momentum = group['momentum']
                prec_momentum = group['prec_momentum']
                
                weight_decay = group['weight_decay']
                if len(state) == 0:
                    state['step'] = 0
                    if momentum > 0:
                        state['momentum_buffer'] = torch.zeros_like(grad)
                    for dim_id, dim in enumerate(grad.size()):
                        # precondition matrices
                        state[f'precond_{dim_id}'] = torch.zeros(
                            dim, dim, out=grad.new(dim, dim)
                        )
                        state[f'u_{dim_id}'] = torch.eye(
                            dim, dim, out=grad.new(dim, dim)
                        )
                        state[f's_{dim_id}'] = torch.zeros(
                            dim, out=grad.new(dim, dim)
                        )
                        state[f'v_{dim_id}'] = torch.eye(
                            dim, out=grad.new(dim, dim)
                        )
                        
                if momentum > 0:
                    grad.mul_(1 - momentum).add_(
                        state['momentum_buffer'], alpha=momentum
                    )
                    grad.mul_(1. / (1 - momentum**(state['step']+1)))
                    state['momentum_buffer'] = grad

                if weight_decay > 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # See Algorithm [TODO] for details
                for dim_id, dim in enumerate(grad.size()):
                    precond = state['precond_{}'.format(dim_id)]
                    #u, s, v = state[f'u_{dim_id}'], state[f's_{dim_id}'], state[f'v_{dim_id}'] 

                    # mat_{dim_id}(grad)
                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    grad = grad.view(dim, -1)

                    grad_t = grad.t()
                    eps_add = group['epsilon'] * torch.eye(dim,device=grad.device)
                    if prec_momentum > 0:
                        precond.mul_(prec_momentum).add_(
                            grad @ grad_t + eps_add, alpha=1-prec_momentum
                        )
                        precond.mul_(1. / (1 - prec_momentum**(state['step']+1)))
                    else:
                        # shampoo step
                        precond.add_(grad @ grad_t + eps_add)
                    if state['step'] % group['update_freq'] == 0:
                        u, s, v = _matrix_power(precond, -1 / order)
                        state[f'u_{dim_id}'] = u
                        state[f's_{dim_id}'] = s
                        state[f'v_{dim_id}'] = v
                    else:
                        # [TODO]
                        u = torch.eye(dim, out=grad.new(dim, dim))
                        s = torch.eye(dim, out=grad.new(dim, dim))
                        v = torch.eye(dim, out=grad.new(dim, dim))
                    inv_precond = u @ s @ v
                    
                    if dim_id == order - 1:
                        # finally
                        grad = grad_t @ inv_precond
                        # grad: (-1, last_dim)
                        grad = grad.view(original_size)
                    else:
                        # if not final
                        grad = inv_precond @ grad
                        # grad (dim, -1)
                        grad = grad.view(transposed_size)

                state['step'] += 1
                p.data.add_(grad, alpha=-group['lr'])

        return loss