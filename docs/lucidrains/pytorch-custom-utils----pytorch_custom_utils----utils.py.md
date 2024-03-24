# `.\lucidrains\pytorch-custom-utils\pytorch_custom_utils\utils.py`

```py
# 导入所需的模块
from typing import Tuple
import torch.nn.functional as F

# 检查变量是否存在
def exists(v):
    return v is not None

# 填充和切片

# 在指定维度上填充张量
def pad_at_dim(t, pad: Tuple[int, int], *, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# 在指定维度上切片张量
def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

# 根据长度填充或切片张量
def pad_or_slice_to(t, length, *, dim, pad_value = 0):
    curr_length = t.shape[dim]

    if curr_length < length:
        t = pad_at_dim(t, (0, length - curr_length), dim = dim, value = pad_value)
    elif curr_length > length:
        t = slice_at_dim(t, slice(0, length), dim = dim)

    return t

# 与掩码相关

# 计算受掩码影响的张量的均值
def masked_mean(tensor, mask, dim = -1, eps = 1e-5):
    if not exists(mask):
        return tensor.mean(dim = dim)

    tensor.masked_fill_(~mask, 0.)

    total_el = mask.sum(dim = dim)
    num = tensor.sum(dim = dim)
    den = total_el.float().clamp(min = eps)
    mean = num / den
    mean.masked_fill_(total_el == 0, 0.)
    return mean

# 对多个掩码进行逻辑与操作
def maybe_and_mask(*masks):
    masks = [*filter(exists, masks)]
    if len(masks) == 0:
        return None

    mask, *rest_masks = masks
    for rest_mask in rest_masks:
        mask = mask & rest_mask

    return mask
```