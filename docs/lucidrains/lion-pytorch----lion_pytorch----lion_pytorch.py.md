# `.\lucidrains\lion-pytorch\lion_pytorch\lion_pytorch.py`

```
# 导入必要的库
from typing import Tuple, Optional, Callable
import torch
from torch.optim.optimizer import Optimizer

# 定义一个函数，用于检查值是否存在
def exists(val):
    return val is not None

# 定义权重更新函数
def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # 根据学习率和权重衰减更新参数值
    p.data.mul_(1 - lr * wd)
    
    # 计算权重更新值
    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
    p.add_(update, alpha=-lr)
    
    # 更新动量的指数移动平均系数
    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

# 定义一个自定义优化器类 Lion，继承自 Optimizer 类
class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: bool = False
    ):
        # 断言学习率必须大于0，beta值必须在0到1之间
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        # 设置默认参数
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )

        # 调用父类的初始化方法
        super().__init__(params, defaults)

        # 设置更新函数为自定义的 update_fn
        self.update_fn = update_fn

        # 如果使用 Triton，则导入 Triton 的更新函数
        if use_triton:
            from lion_pytorch.triton import update_fn as triton_update_fn
            self.update_fn = triton_update_fn

    # 定义优化步骤函数
    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None
    ):

        loss = None
        # 如果存在闭包函数，则计算损失值
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        # 遍历参数组
        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):
                # 获取参数的梯度、学习率、权重衰减、beta1、beta2以及参数状态
                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]

                # 初始化参数状态 - 梯度值的指数移动平均
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                # 调用更新函数更新参数
                self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    wd,
                    beta1,
                    beta2
                )

        return loss
```