# `bitsandbytes\bitsandbytes\optim\lars.py`

```py
# 导入 torch 库
import torch
# 从 torch.optim 中导入 Optimizer 类
from torch.optim import Optimizer

# 从 bitsandbytes.optim.optimizer 中导入 Optimizer1State 类
from bitsandbytes.optim.optimizer import Optimizer1State

# 定义 LARS 类，继承自 Optimizer1State 类
class LARS(Optimizer1State):
    # 初始化方法
    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        max_unorm=0.02,
    ):
        # 如果 momentum 为 0，则抛出 NotImplementedError 异常
        if momentum == 0:
            raise NotImplementedError(
                "LARS without momentum is not supported!"
            )
        # 调用父类的初始化方法
        super().__init__(
            "lars",
            params,
            lr,
            (momentum, dampening),
            0.0,
            weight_decay,
            optim_bits,
            args,
            min_8bit_size,
            percentile_clipping,
            max_unorm=max_unorm,
            block_wise=False,
        )

# 定义 LARS8bit 类，继承自 Optimizer1State 类
class LARS8bit(Optimizer1State):
    # 初始化方法
    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        max_unorm=0.02,
    ):
        # 如果 momentum 为 0，则抛出 NotImplementedError 异常
        if momentum == 0:
            raise NotImplementedError(
                "LARS without momentum is not supported!"
            )
        # 调用父类的初始化方法
        super().__init__(
            "lars",
            params,
            lr,
            (momentum, dampening),
            0.0,
            weight_decay,
            8,
            args,
            min_8bit_size,
            percentile_clipping,
            max_unorm=max_unorm,
            block_wise=False,
        )

# 定义 LARS32bit 类，继承自 Optimizer1State 类
    # 初始化 LARS 优化器对象，设置各种参数
    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        max_unorm=0.02,
    ):
        # 如果动量为0，则抛出异常，因为不支持没有动量的 LARS 优化器
        if momentum == 0:
            raise NotImplementedError(
                "LARS without momentum is not supported!"
            )
        # 调用父类的初始化方法，设置优化器类型为 "lars"，传入参数
        super().__init__(
            "lars",
            params,
            lr,
            (momentum, dampening),
            0.0,
            weight_decay,
            32,
            args,
            min_8bit_size,
            percentile_clipping,
            max_unorm=max_unorm,
            block_wise=False,
        )
# 定义 PytorchLARS 类，继承自 Optimizer 类
class PytorchLARS(Optimizer):
    # 初始化函数，接受参数 params, lr, momentum, dampening, weight_decay, nesterov, max_unorm
    def __init__(
        self,
        params,
        lr=0.01,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        max_unorm=0.02,
    ):
        # 检查学习率是否小于0，如果是则抛出异常
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        # 检查动量是否小于0，如果是则抛出异常
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        # 检查权重衰减是否小于0，如果是则抛出异常
        if weight_decay < 0.0:
            raise ValueError(
                f"Invalid weight_decay value: {weight_decay}"
            )

        # 初始化默认参数字典
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            max_unorm=max_unorm,
        )
        # 如果使用 Nesterov 动量，并且动量小于等于0或者阻尼不为0，则抛出异常
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening"
            )
        # 调用父类的初始化函数
        super().__init__(params, defaults)

    # 设置状态函数，用于恢复对象状态
    def __setstate__(self, state):
        # 调用父类的设置状态函数
        super().__setstate__(state)
        # 遍历参数组，设置默认值 nesterov 为 False
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    # 使用 torch.no_grad() 修饰的函数
    @torch.no_grad()
    # 执行单次优化步骤
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # 初始化损失值
        loss = None
        # 如果有闭包函数，则重新计算模型并返回损失值
        if closure is not None:
            # 启用梯度计算
            with torch.enable_grad():
                loss = closure()

        # 遍历参数组
        for group in self.param_groups:
            # 初始化参数梯度列表、动量缓冲列表、权重衰减、动量、阻尼、Nesterov、最大更新范数、学习率
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            max_unorm = group["max_unorm"]
            lr = group["lr"]

            # 遍历参数
            for p in group["params"]:
                # 如果参数没有梯度，则跳过
                if p.grad is None:
                    continue

                # 获取参数状态
                state = self.state[p]
                d_p = p.grad
                # 如果有权重衰减，则加上权重衰减项
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # 如果有动量，则更新动量缓冲
                if momentum != 0:
                    buf = state.get("momentum_buffer", None)

                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    # 如果使用 Nesterov 动量，则计算更新值
                    if nesterov:
                        update = d_p + buf * momentum
                    else:
                        update = buf

                # 更新比例
                update_scale = 1.0
                # 如果有最大更新范数限制
                if max_unorm > 0.0:
                    assert p.dtype == torch.float32
                    pnorm = torch.norm(p.detach())
                    unorm = torch.norm(update)
                    # 如果更新范数超过最大更新范数限制，则调整更新比例
                    if unorm > max_unorm * pnorm:
                        update_scale = max_unorm * pnorm / unorm

                # 更新参数值
                p.add_(update, alpha=-lr * update_scale)

        # 返回损失值
        return loss
```