import torch
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits


def _hungarian_match(noise: torch.Tensor, params: torch.Tensor, sins: torch.Tensor):
    cost = torch.cdist(noise, params)
    cost = cost.numpy()

    with threadpool_limits(limits=1):
        row_ind, col_ind = linear_sum_assignment(cost)

    noise = noise[row_ind]
    params = params[col_ind]
    sins = sins[col_ind]

    return noise, params, sins


def regular_collate_fn(batch):
    sins, params, sin_fn = zip(*batch)
    sins = torch.cat(sins, dim=0)
    params = torch.cat(params, dim=0)
    noise = torch.randn_like(params)
    sin_fn = sin_fn[0]
    return (sins, params, noise, sin_fn)


def ot_collate_fn(batch):
    sins, params, sin_fn = zip(*batch)
    sins = torch.cat(sins, dim=0)
    params = torch.cat(params, dim=0)
    noise = torch.randn_like(params)

    noise, params, sins = _hungarian_match(noise, params, sins)

    sin_fn = sin_fn[0]
    return (sins, params, noise, sin_fn)
