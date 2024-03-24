# `.\lucidrains\CoLT5-attention\colt5_attention\topk.py`

```
import torch
from torch.cuda.amp import autocast

from collections import namedtuple
from colt5_attention.coor_descent import coor_descent

TopkReturn = namedtuple('TopkReturn', ['values', 'indices', 'coor_descent_values', 'gates'])

@autocast(enabled = False)
def topk(
    x,
    k,
    coor_descent_k_ratio = 9 / 8,
    n_iters = 20,
    eps = 1e-1,
    eps_init = None,
    eps_decay = 1.,
    mask = None,
    fused = False,
    non_differentiable = False
):
    """
    differentiable top-k on last dimension
    """

    if non_differentiable:
        # 如果不需要进行微分计算，则直接使用 torch.topk 函数获取前 k 个值和索引
        values, indices = torch.topk(x, k = k, dim = -1)
        return TopkReturn(values, indices, None, None)

    assert coor_descent_k_ratio >= 1.
    assert k > 0

    # whether to used fused kernel or not

    fn = coor_descent

    if fused and x.is_cuda:
        # 如果开启了 fused 选项并且在 GPU 上，则使用 triton_coor_descent 函数
        from colt5_attention.triton_coor_descent import triton_coor_descent
        fn = triton_coor_descent

    # do coordinate descent for gradients

    # 对梯度进行坐标下降优化
    coor_descent_out = fn(
        x,
        k = min(k * coor_descent_k_ratio, x.shape[-1]),   # 获取稍多一点以获得更好的学习效果，如 CoLT5 论文中所述（他们获取了 9/8 倍）
        mask = mask,
        n_iters = n_iters,
        eps = eps,
        eps_init = eps_init,
        eps_decay = eps_decay
    )

    # do straight through

    # 执行直通操作
    gates = coor_descent_out + (1 - coor_descent_out).detach()

    x = x * gates

    # hard topk

    # 使用 torch.topk 函数获取前 k 个值和索引
    values, indices = torch.topk(x, k, dim = -1)

    # return something that looks like a usual topk, but now differentiable

    # 返回类似于常规 topk 的结果，但现在是可微分的
    coor_descent_values = coor_descent_out.gather(-1, indices)
    gates = gates.gather(-1, indices)

    return TopkReturn(values, indices, coor_descent_values, gates)
```