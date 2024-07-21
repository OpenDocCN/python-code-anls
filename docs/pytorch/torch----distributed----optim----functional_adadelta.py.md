# `.\pytorch\torch\distributed\optim\functional_adadelta.py`

```py
# mypy: allow-untyped-defs
# 引入需要的类型声明
from typing import Dict, List, Optional

# 引入 PyTorch 库
import torch
# 引入 PyTorch 的函数式优化模块
import torch.optim._functional as F
# 引入 Tensor 类型
from torch import Tensor

# 定义一个空列表，用于存储公开的变量名
__all__: List[str] = []

# 定义一个兼容 TorchScript 的函数式 Adadelta 优化器
# 这里我们以函数式的方式使用这个优化器。
# 在更新参数时，不使用 `param.grad`，而是明确地允许分布式优化器将梯度传递给 `step` 函数。
# 这样可以将梯度和参数分离，允许多线程的训练器在更新参数时不会在同一个 `.grad` 上累积数据痕迹。
# 注意：这应该只由分布式优化器内部使用，不应该暴露给用户使用。
@torch.jit.script
class _FunctionalAdadelta:
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        foreach: bool = False,
        maximize: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        # 设置默认参数字典
        self.defaults = {
            "lr": lr,
            "rho": rho,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        # 是否遍历参数列表进行优化
        self.foreach = foreach
        # 是否最大化优化目标
        self.maximize = maximize

        # 如果参数列表为空且不允许空参数列表，则抛出 ValueError
        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # 设置参数组，只允许一个 param_group，并且不允许用户添加额外的 param group，因为这不是常见的使用情况。
        self.param_group = {"params": params}

        # 初始化状态字典，用于存储优化器状态信息
        self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})
    def step(self, gradients: List[Optional[Tensor]]):
        # 获取优化器中的参数组
        params = self.param_group["params"]
        # 存储具有梯度的参数列表
        params_with_grad = []
        # 存储梯度的列表
        grads = []
        # 存储平方平均值的列表
        square_avgs = []
        # 存储累积增量的列表
        acc_deltas = []
        # 存储状态步数的列表
        state_steps = []
        # 获取学习率
        lr = self.defaults["lr"]
        # 获取rho参数
        rho = self.defaults["rho"]
        # 获取eps参数
        eps = self.defaults["eps"]
        # 获取weight_decay参数
        weight_decay = self.defaults["weight_decay"]

        # 检查参数和梯度的数量是否一致
        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )
        
        # 检查是否存在复数梯度
        has_complex = False
        for param, gradient in zip(params, gradients):
            if gradient is not None:
                # 检查参数是否为复数
                has_complex |= torch.is_complex(param)
                # 将具有梯度的参数添加到列表中
                params_with_grad.append(param)
                # 将梯度添加到列表中
                grads.append(gradient)
                # 惰性状态初始化
                if param not in self.state:
                    # 如果参数尚未在状态中，则创建状态字典
                    self.state[param] = {}
                    state = self.state[param]
                    # 初始化步数状态
                    state["step"] = torch.tensor(0.0)
                    # 初始化平方平均值状态
                    state["square_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    # 初始化累积增量状态
                    state["acc_delta"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )

                # 获取参数的状态
                state = self.state[param]
                # 将参数的平方平均值状态添加到列表中
                square_avgs.append(state["square_avg"])
                # 将参数的累积增量状态添加到列表中
                acc_deltas.append(state["acc_delta"])
                # 将参数的步数状态添加到列表中
                state_steps.append(state["step"])

        # 使用torch.no_grad()上下文管理器，执行无梯度更新的操作
        with torch.no_grad():
            # 执行Adadelta优化算法
            F.adadelta(
                params_with_grad,
                grads,
                square_avgs,
                acc_deltas,
                state_steps,
                lr=lr,
                rho=rho,
                eps=eps,
                weight_decay=weight_decay,
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
            )
```