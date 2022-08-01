"""
Functions taken from torchnlp.utils
"""

from typing import Union

import torch


def is_namedtuple(object_):
    return hasattr(object_, "_asdict") and isinstance(object_, tuple)


def collate_tensors(batch, stack_tensors=torch.stack):
    if all([torch.is_tensor(b) for b in batch]):
        return stack_tensors(batch)
    if all([isinstance(b, dict) for b in batch]) and all(
        [b.keys() == batch[0].keys() for b in batch]
    ):
        return {
            key: collate_tensors([d[key] for d in batch], stack_tensors)
            for key in batch[0]
        }
    elif all([is_namedtuple(b) for b in batch]):  # Handle ``namedtuple``
        return batch[0].__class__(
            **collate_tensors([b._asdict() for b in batch], stack_tensors)
        )
    elif all([isinstance(b, list) for b in batch]):
        # Handle list of lists such each list has some column to be batched, similar to:
        # [['a', 'b'], ['a', 'b']] → [['a', 'a'], ['b', 'b']]
        transposed = zip(*batch)
        return [collate_tensors(samples, stack_tensors) for samples in transposed]
    else:
        return batch


def lengths_to_mask(
    *lengths: Union[int, torch.Tensor],
    **kwargs,
):
    lengths = [l.squeeze().tolist() if torch.is_tensor(l) else l for l in lengths]
    # For cases where length is a scalar, this needs to convert it to a list.
    lengths = [l if isinstance(l, list) else [l] for l in lengths]
    assert all(len(l) == len(lengths[0]) for l in lengths)
    batch_size = len(lengths[0])
    other_dimensions = tuple([int(max(l)) for l in lengths])
    mask = torch.zeros(batch_size, *other_dimensions, **kwargs)
    for i, length in enumerate(zip(*tuple(lengths))):
        mask[i][[slice(int(l)) for l in length]].fill_(1)
    return mask.bool()


def mask_fill(
    fill_value: float,
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    padding_index: int,
) -> torch.tensor:
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)


def mean_pooling(token_emb: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
    sum_embeddings = torch.sum(token_emb * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
