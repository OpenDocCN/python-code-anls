# `.\lucidrains\tf-bind-transformer\tf_bind_transformer\optimizer.py`

```py
# 从 torch.optim 模块中导入 AdamW 优化器
from torch.optim import AdamW

# 将参数分为可进行权重衰减和不可进行权重衰减的参数
def separate_weight_decayable_params(params):
    # 找出参数中维度小于 2 的参数，即不可进行权重衰减的参数
    no_wd_params = set([param for param in params if param.ndim < 2])
    # 可进行权重衰减的参数为所有参数减去不可进行权重衰减的参数
    wd_params = set(params) - no_wd_params
    return wd_params, no_wd_params

# 根据参数和超参数创建 AdamW 优化器
def get_optimizer(params, lr = 3e-4, wd = 1e-1, filter_by_requires_grad = False):
    # 如果需要根据 requires_grad 过滤参数，则只保留 requires_grad 为 True 的参数
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))

    # 将参数转换为集合
    params = set(params)
    # 将参数分为可进行权重衰减和不可进行权重衰减的参数
    wd_params, no_wd_params = separate_weight_decayable_params(params)

    # 构建参数组，其中可进行权重衰减的参数使用默认权重衰减，不可进行权重衰减的参数不使用权重衰减
    param_groups = [
        {'params': list(wd_params)},
        {'params': list(no_wd_params), 'weight_decay': 0},
    ]

    # 返回使用 AdamW 优化器的参数组和超参数 lr 和 wd 的优化器
    return AdamW(param_groups, lr = lr, weight_decay = wd)
```