from functools import partial

import numpy as np
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


def concatenate(list_of_arrays: Union[torch.Tensor, np.ndarray]):
    if isinstance(list_of_arrays[0], torch.Tensor):
        return concatenate(list_of_arrays)
    else:
        return np.concatenate(list_of_arrays, axis=0)


def _collate_tuple(batch):
    sins, params, sin_fn = zip(*batch)
    sins = concatenate(sins)
    params = concatenate(params)
    noise = torch.randn_like(params)
    sin_fn = sin_fn[0]
    return (sins, params, noise, sin_fn)


def _collate_dict(batch):
    # batch is a list of dicts and we want a dict of lists
    params = [d["params"] for d in batch]
    mel_spec = [d["mel_spec"] for d in batch]
    audio = [d["audio"] for d in batch]

    params = concatenate(params)
    mel_spec = concatenate(mel_spec)
    if audio[0] is not None:
        audio = concatenate(audio)

    noise = torch.randn_like(params)

    return dict(
        params=params,
        noise=noise,
        mel_spec=mel_spec,
        audio=audio,
    )


def regular_collate_fn(batch):
    item = batch[0]
    if isinstance(item, tuple) or isinstance(item, list):
        fn = _collate_tuple
    elif isinstance(item, dict):
        fn = _collate_dict
    else:
        raise NotImplementedError(
            f"Expected tuple or dict for batch type, got {type(item)}"
        )

    return fn(batch)


def _ot_collate_tuple(batch):
    sins, params, sin_fn = zip(*batch)
    sins = concatenate(sins)
    params = concatenate(params)
    noise = torch.randn_like(params)

    noise, params, sins = _hungarian_match(noise, params, sins)

    sin_fn = sin_fn[0]
    return (sins, params, noise, sin_fn)


def _ot_collate_dict(batch):
    params = [d["params"] for d in batch]
    mel_spec = [d["mel_spec"] for d in batch]
    audio = [d["audio"] for d in batch]

    params = concatenate(params)
    mel_spec = concatenate(mel_spec)
    if audio[0] is not None:
        audio = concatenate(audio)

    noise = torch.randn_like(params)

    noise, params, mel_spec = _hungarian_match(noise, params, mel_spec, audio)

    return dict(
        params=params,
        noise=noise,
        mel_spec=mel_spec,
        audio=audio,
    )


def ot_collate_fn(batch):
    item = batch[0]
    if isinstance(item, tuple) or isinstance(item, list):
        fn = _ot_collate_tuple
    elif isinstance(item, dict):
        fn = _ot_collate_dict
    else:
        raise NotImplementedError(
            f"Expected tuple or dict for batch type, got {type(item)}"
        )

    return fn(batch)
