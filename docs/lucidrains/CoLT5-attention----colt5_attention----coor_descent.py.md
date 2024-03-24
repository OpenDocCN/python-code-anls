# `.\lucidrains\CoLT5-attention\colt5_attention\coor_descent.py`

```
# 导入 torch 库
import torch
# 导入 torch.nn.functional 模块，并重命名为 F
import torch.nn.functional as F
# 从 torch.cuda.amp 模块中导入 autocast 函数
from torch.cuda.amp import autocast
# 从 einops 库中导入 rearrange 函数
from einops import rearrange

# 定义函数，判断变量是否存在
def exists(val):
    return val is not None

# 定义函数，返回 val 或者默认值 d
def default(val, d):
    return val if exists(val) else d

# 定义函数，计算输入张量的对数，避免值过小
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 使用 autocast 装饰器，设置自动混合精度为关闭
@autocast(enabled = False)
# 定义坐标下降函数
def coor_descent(
    s,
    *,
    n_iters,
    k,
    eps = 1e-1,
    eps_init = None,
    eps_decay = 1.,
    mask = None
):
    """
    coordinate descent  - https://arxiv.org/abs/1502.04759, utilized in https://arxiv.org/abs/2303.09752
    ε-scaling           - https://arxiv.org/abs/1610.06519, utilized in https://arxiv.org/abs/2304.04947

    in a follow up paper applying coordinate descent routing to efficient fine tuning
    they were able to cut n_iters from 50 -> 20 by setting eps_init = 4 and eps_decay = 0.7
    eps was dependent on the task, and ranged from 0.02 to 1
    """

    # 断言迭代次数大于 0
    assert n_iters > 0

    # 定义 mask_value 为 s 数据类型的最小值
    mask_value = -torch.finfo(s.dtype).max

    # 如果 k 不是 torch.Tensor 类型，则将其转换为 torch.Tensor 类型
    if not isinstance(k, torch.Tensor):
        k = torch.Tensor([k]).to(s)
    else:
        k = rearrange(k, '... -> ... 1')

    # 计算 k 的对数
    logk = log(k)

    # 如果 mask 存在，则用 mask_value 填充 s
    if exists(mask):
        s = s.masked_fill(~mask, mask_value)

    # 初始化 a 和 b
    a = 0
    b = -s

    # 初始化当前的 epsilon 值
    current_eps = max(default(eps_init, eps), eps)

    # 迭代 n_iters 次
    for _ in range(n_iters):
        # 计算 sb
        sb = ((s + b) / current_eps)

        # 如果 mask 存在，则用 mask_value 填充 sb
        if exists(mask):
            sb = sb.masked_fill(~mask, mask_value)

        # 更新 a 和 b
        a = current_eps * (logk - sb.logsumexp(dim = -1, keepdim = True))
        b = -F.relu(s + a)

        # 更新当前的 epsilon 值
        current_eps = max(current_eps * eps_decay, eps)

    # 计算分数
    scores = ((s + a + b) / current_eps).exp()

    # 如果 mask 存在，则用 0 填充 scores
    if exists(mask):
        scores = scores.masked_fill(~mask, 0.)

    # 返回分数
    return scores
```