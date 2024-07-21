# `.\pytorch\torch\distributed\optim\functional_adam.py`

```py
# mypy: allow-untyped-defs
# 引入类型相关的模块
from typing import Dict, List, Optional, Tuple

# 引入 PyTorch 库
import torch
# 引入 PyTorch 优化器函数
import torch.optim._functional as F
# 引入 Tensor 类型
from torch import Tensor

# 初始化空的导出模块名列表
__all__: List[str] = []


# 定义一个 TorchScript 兼容的函数式 Adam 优化器
# 在这里我们以函数式的方式使用这个优化器。
# 不使用 `param.grad` 来更新参数，
# 而是显式地允许分布式优化器将梯度传递给 `step` 函数。
# 这样做的好处是可以将梯度和参数分开，
# 允许多线程的训练器在更新参数时不会出现数据积累到同一个 .grad 上的情况。
# 注意：这应该只在分布式优化器的内部使用，不应该暴露给用户使用。
@torch.jit.script
class _FunctionalAdam:
    def __init__(
        self,
        params: List[Tensor],  # 参数列表
        lr: float = 1e-3,  # 学习率，默认为 0.001
        betas: Tuple[float, float] = (0.9, 0.999),  # beta 参数，默认为 (0.9, 0.999)
        eps: float = 1e-8,  # epsilon 值，默认为 1e-8
        weight_decay: float = 0.0,  # 权重衰减，默认为 0.0
        amsgrad: bool = False,  # 是否使用 AMSGrad，默认为 False
        maximize: bool = False,  # 是否最大化优化目标，默认为 False
        foreach: bool = False,  # 是否对每个参数单独计算，默认为 False
        fused: bool = False,  # 是否使用融合计算，默认为 False
        _allow_empty_param_list: bool = False,  # 是否允许空参数列表，默认为 False
    ):
        # 检查学习率是否非负
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        # 检查 epsilon 值是否非负
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        # 检查 beta 参数是否在有效范围内
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        # 检查权重衰减值是否非负
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # 设置默认参数字典
        self.defaults = {
            "lr": lr,
            "eps": eps,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
        }
        # 是否使用 AMSGrad
        self.amsgrad = amsgrad
        # 是否最大化优化目标
        self.maximize = maximize
        # 是否对每个参数单独计算
        self.foreach = foreach
        # 是否使用融合计算
        self.fused = fused
        # 状态字典，用于存储优化器状态
        self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})

        # 如果参数列表为空并且不允许空参数列表，则抛出异常
        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # 设置参数组，这里只允许有一个参数组，并且不允许用户添加额外的参数组
        self.param_group = {"params": params}
    def step_param(self, param: Tensor, grad: Optional[Tensor]):
        """
        Similar to step, but operates on a single parameter and optionally a
        gradient tensor.
        """
        # 初始化空列表用于存储有梯度的参数、梯度、指数移动平均等
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps: List[Tensor] = []

        # 检查参数是否为复数类型
        has_complex = torch.is_complex(param)

        # 如果梯度不为None，将参数和梯度添加到相应列表中
        if grad is not None:
            params_with_grad.append(param)
            grads.append(grad)

        # 如果参数不在状态字典中，初始化参数的状态信息
        if param not in self.state:
            self.state[param] = {}
            state = self.state[param]
            state["step"] = torch.tensor(0.0)
            # 初始化指数移动平均和平方指数移动平均为零张量
            state["exp_avg"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
            state["exp_avg_sq"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
            # 如果使用了AMSGrad，初始化最大平方指数移动平均为零张量
            if self.amsgrad:
                state["max_exp_avg_sq"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )

        # 获取参数的状态信息
        state = self.state[param]
        exp_avgs.append(state["exp_avg"])
        exp_avg_sqs.append(state["exp_avg_sq"])

        # 如果使用了AMSGrad，添加最大平方指数移动平均到列表中
        if self.amsgrad:
            max_exp_avg_sqs.append(state["max_exp_avg_sq"])

        # 添加步数信息到状态步数列表中
        state_steps.append(state["step"])

        # 使用torch.no_grad()上下文管理器确保在优化过程中不计算梯度
        with torch.no_grad():
            # 调用F.adam函数执行优化步骤
            F.adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=self.amsgrad,
                has_complex=has_complex,
                maximize=self.maximize,
                beta1=self.defaults["beta1"],
                beta2=self.defaults["beta2"],
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                eps=self.defaults["eps"],
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
            )
    # 定义一个方法 `step`，接收梯度列表作为参数
    def step(self, gradients: List[Optional[Tensor]]):
        # 获取优化器参数组中的参数列表
        params = self.param_group["params"]
        # 初始化一个空列表，用于存储具有梯度的参数
        params_with_grad = []
        # 初始化一个空列表，用于存储梯度
        grads = []
        # 初始化空列表，用于存储指数移动平均
        exp_avgs = []
        # 初始化空列表，用于存储指数移动平均的平方
        exp_avg_sqs = []
        # 初始化空列表，用于存储指数移动平均平方的最大值
        max_exp_avg_sqs = []
        # 初始化空列表，用于存储状态步数
        state_steps: List[Tensor] = []
        # 是否存在复数参数
        has_complex = False

        # 如果参数列表和梯度列表的长度不相等，则抛出值错误异常
        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        # 遍历参数组中的参数和对应的梯度
        for param, gradient in zip(self.param_group["params"], gradients):
            # 如果梯度不为 None
            if gradient is not None:
                # 检查参数是否为复数类型
                has_complex |= torch.is_complex(param)
                # 将具有梯度的参数添加到列表中
                params_with_grad.append(param)
                # 将梯度添加到列表中
                grads.append(gradient)
                # 如果参数的状态尚未初始化
                if param not in self.state:
                    # 初始化参数的状态字典
                    self.state[param] = {}
                    state = self.state[param]
                    # 初始化步数为 0 的张量
                    state["step"] = torch.tensor(0.0)
                    # 初始化指数移动平均值为全零张量
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    # 初始化指数移动平均的平方为全零张量
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    # 如果启用了 AMSGrad，则初始化最大指数移动平均的平方为全零张量
                    if self.amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )

                # 获取参数的状态字典
                state = self.state[param]
                # 将参数的指数移动平均值添加到列表中
                exp_avgs.append(state["exp_avg"])
                # 将参数的指数移动平均的平方添加到列表中
                exp_avg_sqs.append(state["exp_avg_sq"])

                # 如果启用了 AMSGrad，则将参数的最大指数移动平均的平方添加到列表中
                if self.amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                # 将参数的步数添加到列表中
                state_steps.append(state["step"])

        # 进入无梯度更新区域
        with torch.no_grad():
            # 调用 F.adam 函数执行 Adam 优化步骤
            F.adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=self.amsgrad,
                has_complex=has_complex,
                maximize=self.maximize,
                beta1=self.defaults["beta1"],
                beta2=self.defaults["beta2"],
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                eps=self.defaults["eps"],
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
            )
```