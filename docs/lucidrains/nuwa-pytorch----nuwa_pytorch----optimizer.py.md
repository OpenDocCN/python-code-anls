# `.\lucidrains\nuwa-pytorch\nuwa_pytorch\optimizer.py`

```py
# 导入 torch 库
import torch
# 从 torch.optim 中导入 AdamW 和 Adam 优化器

# 分离可进行权重衰减的参数
def separate_weight_decayable_params(params):
    # 找出参数中维度小于 2 的参数，即不需要进行权重衰减的参数
    no_wd_params = set([param for param in params if param.ndim < 2])
    # 计算需要进行权重衰减的参数
    wd_params = set(params) - no_wd_params
    return wd_params, no_wd_params

# 获取优化器
def get_optimizer(
    params,
    lr = 3e-4,
    wd = 1e-1,
    filter_by_requires_grad = False
):
    # 如果需要根据 requires_grad 过滤参数
    if filter_by_requires_grad:
        # 过滤出 requires_grad 为 True 的参数
        params = list(filter(lambda t: t.requires_grad, params))

    # 如果权重衰减参数为 0，则使用 Adam 优化器
    if wd == 0:
        return Adam(list(params), lr = lr)

    # 将参数转换为集合
    params = set(params)
    # 分离出需要进行权重衰减的参数和不需要进行权重衰减的参数
    wd_params, no_wd_params = separate_weight_decayable_params(params)

    # 构建参数组，其中包含需要进行权重衰减的参数和不需要进行权重衰减的参数
    param_groups = [
        {'params': list(wd_params)},
        {'params': list(no_wd_params), 'weight_decay': 0},
    ]

    # 使用 AdamW 优化器，设置学习率和权重衰减参数
    return AdamW(param_groups, lr = lr, weight_decay = wd)
```