# `.\lucidrains\magvit2-pytorch\magvit2_pytorch\optimizer.py`

```
# 从 torch.optim 模块中导入 AdamW 和 Adam 优化器
from torch.optim import AdamW, Adam

# 将参数分为需要权重衰减和不需要权重衰减的两个列表
def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []

    # 遍历参数列表，根据参数的维度将参数分别添加到对应的列表中
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)

    return wd_params, no_wd_params

# 获取优化器
def get_optimizer(
    params,
    lr = 1e-4,
    wd = 1e-2,
    betas = (0.9, 0.99),
    eps = 1e-8,
    filter_by_requires_grad = False,
    group_wd_params = True,
    **kwargs
):
    # 如果需要根据 requires_grad 过滤参数
    if filter_by_requires_grad:
        params = [t for t in params if t.requires_grad]

    # 设置优化器的参数
    opt_kwargs = dict(lr = lr, betas = betas, eps = eps)

    # 如果权重衰减为 0，则返回 Adam 优化器
    if wd == 0:
        return Adam(params, **opt_kwargs)

    # 设置权重衰减参数
    opt_kwargs = {'weight_decay': wd, **opt_kwargs}

    # 如果不对权重衰减参数进行分组，则返回 AdamW 优化器
    if not group_wd_params:
        return AdamW(params, **opt_kwargs)

    # 将参数分为需要权重衰减和不需要权重衰减的两个列表
    wd_params, no_wd_params = separate_weight_decayable_params(params)

    # 组合参数列表，分别设置权重衰减
    params = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]

    return AdamW(params, **opt_kwargs)
```