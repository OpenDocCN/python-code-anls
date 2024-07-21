# `.\pytorch\torch\distributed\optim\functional_rmsprop.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型声明
from typing import Dict, List, Optional

# 引入 PyTorch 库
import torch
# 引入 PyTorch 中的函数优化模块
import torch.optim._functional as F
# 引入 Tensor 类型
from torch import Tensor

# 定义空的导出列表
__all__: List[str] = []


# 定义一个兼容 TorchScript 的函数式 RMSprop 优化器
# 在这个优化器中，我们以一种函数式的方式使用它。
# 在更新参数时，我们允许分布式优化器将梯度传递给 `step` 函数。
# 这样做的好处是，我们能够分离梯度和参数，并允许多线程的训练器在更新参数时
# 不会出现数据积累到同一个 `.grad` 的情况。
# 注意：这个类应该只在分布式优化器的内部使用，不应该暴露给用户。
@torch.jit.script
class _FunctionalRMSprop:
    def __init__(
        self,
        params: List[Tensor],  # 参数列表，包含需要优化的张量
        lr: float = 1e-2,  # 学习率，默认为 0.01
        alpha: float = 0.99,  # RMSprop 中的衰减率，默认为 0.99
        eps: float = 1e-8,  # 用于数值稳定性的小常数，默认为 1e-8
        weight_decay: float = 0.0,  # 权重衰减（L2 正则化）的参数，默认为 0.0
        momentum: float = 0.0,  # 动量参数，默认为 0.0
        centered: bool = False,  # 是否使用中心化的 RMSprop，默认为 False
        foreach: bool = False,  # 是否对每个参数分别应用 RMSprop，默认为 False
        maximize: bool = False,  # 是否进行最大化优化，默认为 False
        _allow_empty_param_list: bool = False,  # 是否允许空的参数列表，默认为 False
    ):
        # 设置默认参数字典
        self.defaults = {
            "lr": lr,
            "alpha": alpha,
            "eps": eps,
            "weight_decay": weight_decay,
            "momentum": momentum,
        }
        # 是否使用中心化的 RMSprop
        self.centered = centered
        # 是否对每个参数分别应用 RMSprop
        self.foreach = foreach
        # 是否进行最大化优化
        self.maximize = maximize

        # 如果参数列表为空且不允许空参数列表，则抛出 ValueError 异常
        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # 设置参数组字典，只允许存在一个参数组，并不允许用户添加额外的参数组，因为这不是常见用例。
        self.param_group = {"params": params}

        # 初始化优化器状态，使用 TorchScript 的类型注解
        self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})
    # 定义一个方法，用于执行优化器的单步优化过程，接收梯度列表作为参数
    def step(self, gradients: List[Optional[Tensor]]):
        # 从参数组中获取所有参数
        params = self.param_group["params"]
        # 用于存储有梯度的参数
        params_with_grad = []
        # 用于存储梯度值
        grads = []
        # 用于存储平方平均值
        square_avgs = []
        # 用于存储梯度平均值
        grad_avgs = []
        # 用于存储动量缓冲
        momentum_buffer_list = []
        # 用于存储状态步数
        state_steps = []
        # 获取学习率，默认为设定值
        lr = self.defaults["lr"]
        # 获取 alpha 参数，默认为设定值
        alpha = self.defaults["alpha"]
        # 获取 eps 参数，默认为设定值
        eps = self.defaults["eps"]
        # 获取 momentum 参数，默认为设定值
        momentum = self.defaults["momentum"]
        # 获取 weight_decay 参数，默认为设定值
        weight_decay = self.defaults["weight_decay"]

        # 检查参数和梯度列表长度是否一致，若不一致则抛出异常
        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        # 检查是否有复数类型的参数
        has_complex = False
        for param, gradient in zip(params, gradients):
            if gradient is not None:
                has_complex |= torch.is_complex(param)
                # 将有梯度的参数加入列表
                params_with_grad.append(param)
                # 将梯度值加入列表
                grads.append(gradient)
                # 如果参数尚未在状态中初始化，则进行懒惰初始化
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    # 初始化步数状态为零张量
                    state["step"] = torch.tensor(0.0)
                    # 初始化平方平均值状态为与参数相同形状的零张量
                    state["square_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    # 如果设定了动量大于0，则初始化动量缓冲状态为与参数相同形状的零张量
                    if momentum > 0:
                        state["momentum_buffer"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )
                    # 如果使用中心化参数更新，则初始化梯度平均值状态为与参数相同形状的零张量
                    if self.centered:
                        state["grad_avg"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )

                # 获取当前参数的状态
                state = self.state[param]
                # 将当前参数的平方平均值状态加入列表
                square_avgs.append(state["square_avg"])
                # 如果动量大于0，则将当前参数的动量缓冲状态加入列表
                if momentum > 0:
                    momentum_buffer_list.append(state["momentum_buffer"])
                # 如果使用中心化参数更新，则将当前参数的梯度平均值状态加入列表
                if self.centered:
                    grad_avgs.append(state["grad_avg"])

                # 将当前参数的步数状态加入列表
                state_steps.append(state["step"])

        # 使用无梯度上下文管理器执行 RMSProp 算法
        with torch.no_grad():
            # 调用 RMSProp 算法优化参数
            F.rmsprop(
                params_with_grad,
                grads,
                square_avgs,
                grad_avgs,
                momentum_buffer_list,
                state_steps,
                lr=lr,
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
                momentum=momentum,
                centered=self.centered,
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
            )
```