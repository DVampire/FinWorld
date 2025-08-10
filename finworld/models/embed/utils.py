import torch
import string
from functools import reduce
from operator import mul
import numpy as np

def sin_cos_encode(x: torch.Tensor,
                   length: int = 10,
                   min_value: float = 0.0,
                   ) -> torch.Tensor:
    """
    Apply sine and cosine encoding to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
        dim (int): Dimension for sine and cosine encoding.

    Returns:
        torch.Tensor: Encoded tensor of shape (batch_size, seq_len, 3).
    """
    sin_encoding = torch.sin(x * (2 * np.pi * (x - min_value) / length))
    cos_encoding = torch.cos(x * (2 * np.pi * (x - min_value) / length))
    frac_encoding = (x - min_value) / length
    encoded = torch.stack([sin_encoding, cos_encoding, frac_encoding], dim=-1)
    return encoded

def get_patch_info(input_channel, data_size, patch_size):
    """
    Generate patch-related metadata for patchify/unpatchify.

    Args:
        input_channel (int): Number of input channels
        data_size (tuple): Shape of the input tensor (D1, D2, ..., Dn)
        patch_size (tuple): Patch size for each spatial dimension

    Returns:
        dict: {
            'N': batch size,
            'C': number of channels,
            'spatial_dims': original spatial dimensions (list),
            'patch_size': patch size for each dimension (list),
            'patches': number of patches per dimension (list),
        }
    """
    patches = [d // p for d, p in zip(data_size, patch_size)]
    num_patches = reduce(mul, patches)

    return {
        'input_channel': input_channel,
        'data_size': list(data_size),
        'patch_size': list(patch_size),
        'patches': patches,
        'num_patches': num_patches,
    }

def patchify(x, patch_info):
    """
    Convert input tensor to patch sequence.

    Args:
        x (Tensor): Input tensor of shape (N, C, D1, D2, ..., Dn)
        patch_info (dict): Metadata returned by get_patch_info

    Returns:
        Tensor: Patchified tensor of shape (N, L, patch_volume * C)
    """
    N = x.shape[0]
    input_channel = patch_info['input_channel']
    data_size = patch_info['data_size']
    patch_size = patch_info['patch_size']
    patches = patch_info['patches']
    dims = len(data_size)

    # Reshape to (N, C, n1, p1, n2, p2, ..., nn, pn)
    shape = [N, input_channel]
    for n, p in zip(patches, patch_size):
        shape.extend([n, p])
    x = x.reshape(*shape)

    # Build einsum string
    letters = string.ascii_lowercase
    idx = iter(letters)
    n_idx = [next(idx) for _ in range(dims)]
    p_idx = [next(idx) for _ in range(dims)]
    c_idx = next(idx)

    ein_input = f"n{c_idx}" + ''.join([a + b for a, b in zip(n_idx, p_idx)])
    ein_output = f"n" + ''.join(n_idx) + ''.join(p_idx) + c_idx
    x = torch.einsum(f"{ein_input}->{ein_output}", x)

    x = x.reshape(N, reduce(mul, patches), input_channel * reduce(mul, patch_size))
    return x

def unpatchify(x, patch_info):
    """
    Restore original tensor from patchified sequence.

    Args:
        x (Tensor): Patchified tensor of shape (N, L, patch_volume * C)
        patch_info (dict): Metadata returned by get_patch_info

    Returns:
        Tensor: Reconstructed tensor of shape (N, C, D1, ..., Dn)
    """
    N = x.shape[0]
    input_channel = patch_info['input_channel']
    data_size = patch_info['data_size']
    patch_size = patch_info['patch_size']
    patches = patch_info['patches']
    dims = len(data_size)

    patch_volume = reduce(mul, patch_size)
    x = x.reshape(N, *patches, input_channel, *patch_size)

    # Build einsum string (reverse of patchify)
    letters = string.ascii_lowercase
    idx = iter(letters)
    n_idx = [next(idx) for _ in range(dims)]
    p_idx = [next(idx) for _ in range(dims)]
    c_idx = next(idx)

    ein_input = f"n{''.join(n_idx)}{c_idx}{''.join(p_idx)}"
    ein_output = f"n{c_idx}" + ''.join([a + b for a, b in zip(n_idx, p_idx)])
    x = torch.einsum(f"{ein_input}->{ein_output}", x)

    x = x.reshape(N, input_channel, *data_size)
    return x