import torch
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits


def _hungarian_match(noise: torch.Tensor, params: torch.Tensor, *args):
    cost = torch.cdist(noise, params)
    cost = cost.numpy()

    with threadpool_limits(limits=1):
        row_ind, col_ind = linear_sum_assignment(cost)

    noise = noise[row_ind]
    params = params[col_ind]

    return_values = [noise, params]
    for arg in args:
        if arg is not None:
            return_values.append(arg[col_ind])
        else:
            return_values.append(None)

    return tuple(return_values)


def _collate_tuple(batch):
    sins, params, sin_fn = zip(*batch)
    sins = torch.cat(sins, dim=0)
    params = torch.cat(params, dim=0)
    noise = torch.randn_like(params)
    sin_fn = sin_fn[0]
    return (sins, params, noise, sin_fn)


def _collate_dict(batch):
    params = batch["params"]
    mel_spec = batch["mel_spec"]
    audio = batch["audio"]

    params = torch.cat(params, dim=0)
    mel_spec = torch.cat(mel_spec, dim=0)
    if audio is not None:
        audio = torch.cat(audio, dim=0)

    noise = torch.randn_like(params)

    return dict(
        params=params,
        noise=noise,
        mel_spec=mel_spec,
        audio=audio,
    )


def regular_collate_fn(batch):
    if isinstance(batch, tuple) or isinstance(batch, list):
        fn = _collate_tuple
    elif isinstance(batch, dict):
        fn = _collate_dict
    else:
        raise NotImplementedError(
            f"Expected tuple or dict for batch type, got {type(batch)}"
        )

    return fn(batch)


def _ot_collate_tuple(batch):
    sins, params, sin_fn = zip(*batch)
    sins = torch.cat(sins, dim=0)
    params = torch.cat(params, dim=0)
    noise = torch.randn_like(params)

    noise, params, sins = _hungarian_match(noise, params, sins)

    sin_fn = sin_fn[0]
    return (sins, params, noise, sin_fn)


def _ot_collate_dict(batch):
    params = batch["params"]
    mel_spec = batch["mel_spec"]
    audio = batch["audio"]

    params = torch.cat(params, dim=0)
    mel_spec = torch.cat(mel_spec, dim=0)
    if audio is not None:
        audio = torch.cat(audio, dim=0)

    noise = torch.randn_like(params)

    noise, params, mel_spec = _hungarian_match(noise, params, mel_spec, audio)

    return dict(
        params=params,
        noise=noise,
        mel_spec=mel_spec,
        audio=audio,
    )


def ot_collate_fn(batch):
    if isinstance(batch, tuple) or isinstance(batch, list):
        fn = _ot_collate_tuple
    elif isinstance(batch, dict):
        fn = _ot_collate_dict
    else:
        raise NotImplementedError(
            f"Expected tuple or dict for batch type, got {type(batch)}"
        )

    return fn(batch)
