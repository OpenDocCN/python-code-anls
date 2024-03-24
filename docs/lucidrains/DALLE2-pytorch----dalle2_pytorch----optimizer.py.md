# `.\lucidrains\DALLE2-pytorch\dalle2_pytorch\optimizer.py`

```py
# 从 torch.optim 模块中导入 AdamW 和 Adam 优化器
from torch.optim import AdamW, Adam

# 将参数分为需要权重衰减和不需要权重衰减的两个列表
def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        # 根据参数的维度判断是否需要权重衰减
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
    # 根据是否需要梯度过滤参数
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))

    # 如果权重衰减为0，则使用 Adam 优化器
    if wd == 0:
        return Adam(params, lr = lr, betas = betas, eps = eps)

    # 如果需要对参数进行分组权重衰减
    if group_wd_params:
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        # 将参数分为需要权重衰减和不需要权重衰减的两组
        params = [
            {'params': wd_params},
            {'params': no_wd_params, 'weight_decay': 0},
        ]

    # 使用 AdamW 优化器，设置学习率、权重衰减、动量参数和 epsilon
    return AdamW(params, lr = lr, weight_decay = wd, betas = betas, eps = eps)
```