# `.\lucidrains\q-transformer\q_transformer\optimizer.py`

```
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

# 获取 Adam 或 AdamW 优化器
def get_adam_optimizer(
    params,
    lr = 1e-4,
    wd = 1e-2,
    betas = (0.9, 0.99),
    eps = 1e-8,
    filter_by_requires_grad = False,
    group_wd_params = True
):
    # 判断是否需要权重衰减
    has_wd = wd > 0

    # 根据是否需要过滤梯度为 True 的参数来更新参数列表
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))

    # 如果需要对参数进行分组并进行权重衰减
    if group_wd_params and has_wd:
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        # 更新参数列表，分为需要权重衰减和不需要权重衰减的两组
        params = [
            {'params': wd_params},
            {'params': no_wd_params, 'weight_decay': 0},
        ]

    # 如果不需要权重衰减，则返回 Adam 优化器
    if not has_wd:
        return Adam(params, lr = lr, betas = betas, eps = eps)

    # 如果需要权重衰减，则返回 AdamW 优化器
    return AdamW(params, lr = lr, weight_decay = wd, betas = betas, eps = eps)
```