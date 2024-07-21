# `.\pytorch\torch\distributed\optim\functional_sgd.py`

```
# mypy: allow-untyped-defs
# 导入必要的类型声明
from typing import Dict, List, Optional

# 导入 PyTorch 库
import torch
# 导入 PyTorch 优化器的函数实现
import torch.optim._functional as F
# 导入 Tensor 类型
from torch import Tensor

# 定义一个 TorchScript 兼容的函数式 SGD 优化器
# 在这里我们以函数式的方式使用这些优化器。
# 不再使用 `param.grad` 来更新参数，
# 我们显式允许分布式优化器将梯度传递给 `step` 函数。
# 这样，我们可以将梯度和参数分开，并允许多线程的训练器在更新参数时不会累积到同一个 .grad 上。
# 注意：这仅应该由分布式优化器内部使用，不应该暴露给用户。
@torch.jit.script
class _FunctionalSGD:
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
        foreach: bool = False,
        fused: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        # 设置默认参数
        self.defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
        }
        # 是否使用 Nesterov 加速梯度法
        self.nesterov = nesterov
        # 是否最大化优化目标（而不是最小化）
        self.maximize = maximize
        # 是否针对每个参数进行优化
        self.foreach = foreach
        # 是否使用融合版本的优化器
        self.fused = fused
        # 保存优化器状态信息的字典
        self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})

        # 如果参数列表为空并且不允许空参数列表，则引发 ValueError
        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # 注意：我们只有一个 param_group，并且不允许用户添加额外的 param_group，因为这不是常见用例。
        self.param_group = {"params": params}
    def step_param(self, param: Tensor, grad: Optional[Tensor]):
        """Similar to self.step, but operates on a single parameter and
        its gradient.
        """
        # 从默认参数中获取权重衰减、动量和阻尼
        weight_decay = self.defaults["weight_decay"]
        momentum = self.defaults["momentum"]
        dampening = self.defaults["dampening"]
        lr = self.defaults["lr"]
        # 将当前参数封装成列表
        params = [param]
        # 初始化动量缓冲列表为空
        momentum_buffer_list: List[Optional[Tensor]] = []
        # 初始化梯度列表
        grads = []

        # 检查是否存在稀疏梯度
        has_sparse_grad = False
        if grad is not None:
            # 如果梯度不为None，将其添加到梯度列表中
            grads.append(grad)
            if grad.is_sparse:
                has_sparse_grad = True
            # 如果参数不在状态字典中，为其创建一个空的状态字典
            if param not in self.state:
                self.state[param] = {}
            state = self.state[param]
            # 如果状态字典中没有动量缓冲区，添加一个None到动量缓冲列表中
            if "momentum_buffer" not in state:
                momentum_buffer_list.append(None)
            else:
                # 否则，将动量缓冲区添加到动量缓冲列表中
                momentum_buffer_list.append(state["momentum_buffer"])

        # 使用torch.no_grad()上下文管理器，禁用梯度计算
        with torch.no_grad():
            # 调用F.sgd函数执行随机梯度下降算法
            F.sgd(
                params,
                grads,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=self.nesterov,
                maximize=self.maximize,
                has_sparse_grad=has_sparse_grad,
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
            )
        # 更新参数状态字典中的动量缓冲区
        state = self.state[param]
        momentum_buffer = momentum_buffer_list[0]
        if momentum_buffer is not None:
            state["momentum_buffer"] = momentum_buffer
    # 定义一个方法，用于更新优化器的参数
    def step(self, gradients: List[Optional[Tensor]]):
        # 获取当前参数组的参数列表
        params = self.param_group["params"]
        # 用于存储具有梯度的参数
        params_with_grad = []
        # 存储每个参数的梯度
        grads = []
        # 用于存储动量缓冲区的列表
        momentum_buffer_list: List[Optional[Tensor]] = []
        # 获取学习率
        lr = self.defaults["lr"]
        # 获取权重衰减因子
        weight_decay = self.defaults["weight_decay"]
        # 获取动量参数
        momentum = self.defaults["momentum"]
        # 获取阻尼参数
        dampening = self.defaults["dampening"]

        # 检查传入的梯度是否与参数列表长度相同
        if len(params) != len(gradients):
            # 抛出数值错误异常
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        # 标记是否存在稀疏梯度
        has_sparse_grad = False
        # 遍历参数和梯度列表
        for param, gradient in zip(params, gradients):
            # 如果梯度不为空
            if gradient is not None:
                # 将参数添加到有梯度的参数列表中
                params_with_grad.append(param)
                # 将梯度添加到梯度列表中
                grads.append(gradient)
                # 如果梯度是稀疏的
                if gradient.is_sparse:
                    has_sparse_grad = True

                # 如果参数不在优化器状态中
                if param not in self.state:
                    self.state[param] = {}

                # 获取参数的状态字典
                state = self.state[param]
                # 如果状态字典中没有动量缓冲区
                if "momentum_buffer" not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state["momentum_buffer"])

        # 使用无梯度上下文管理器，执行SGD更新参数
        with torch.no_grad():
            F.sgd(
                params_with_grad,
                grads,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=self.nesterov,
                maximize=self.maximize,
                has_sparse_grad=has_sparse_grad,
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
            )

        # 更新状态中的动量缓冲区
        for i, p in enumerate(params_with_grad):
            # 获取参数的状态字典
            state = self.state[p]
            # 获取动量缓冲区
            momentum_buffer = momentum_buffer_list[i]
            # 如果动量缓冲区不为空
            if momentum_buffer is not None:
                # 更新状态中的动量缓冲区
                state["momentum_buffer"] = momentum_buffer
```