# `.\lucidrains\pytorch-custom-utils\pytorch_custom_utils\get_adam_optimizer.py`

```
# 从 typing 模块导入 Tuple 类型
from typing import Tuple
# 从 torch.optim 模块导入 AdamW 和 Adam 优化器

# optimizer

# 将参数分为需要权重衰减和不需要权重衰减的两个列表
def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []

    for param in params:
        # 根据参数的维度判断是否需要权重衰减
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)

    return wd_params, no_wd_params

# 获取 Adam 优化器
def get_adam_optimizer(
    params,
    lr: float = 1e-4,
    wd: float = 1e-2,
    betas: Tuple[int, int] = (0.9, 0.99),
    eps: float = 1e-8,
    filter_by_requires_grad = False,
    omit_gammas_and_betas_from_wd = True,
    **kwargs
):
    # 判断是否需要权重衰减
    has_weight_decay = wd > 0.

    # 根据是否需要过滤 requires_grad 来筛选参数
    if filter_by_requires_grad:
        params = [t for t in params if t.requires_grad]

    # 设置优化器的参数
    opt_kwargs = dict(
        lr = lr,
        betas = betas,
        eps = eps
    )

    # 如果不需要权重衰减，则返回 Adam 优化器
    if not has_weight_decay:
        return Adam(params, **opt_kwargs)

    # 设置带有权重衰减的优化器参数
    opt_kwargs = {'weight_decay': wd, **opt_kwargs}

    # 如果不忽略 gammas 和 betas 的权重衰减，则返回 AdamW 优化器
    if not omit_gammas_and_betas_from_wd:
        return AdamW(params, **opt_kwargs)

    # 在 transformers 中有一种早期实践，其中从权重衰减中省略了 betas 和 gammas
    # 不确定是否真的需要
    wd_params, no_wd_params = separate_weight_decayable_params(params)

    # 将参数分为需要权重衰减和不需要权重衰减的两部分
    params = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]

    return AdamW(params, **opt_kwargs)
```