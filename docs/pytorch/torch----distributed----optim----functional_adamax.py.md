# `.\pytorch\torch\distributed\optim\functional_adamax.py`

```py
# mypy: allow-untyped-defs
# 导入必要的类型声明
from typing import Dict, List, Optional, Tuple

# 导入 PyTorch 库
import torch
# 导入 PyTorch 中的函数式优化模块
import torch.optim._functional as F
# 导入 PyTorch 中的张量类型
from torch import Tensor

# 定义一个 TorchScript 兼容的函数式 Adamax 优化器类
# 该类允许在分布式优化器中以函数式的方式使用
# 在更新参数时，不使用 `param.grad`，而是允许分布式优化器将梯度传递给 `step` 函数
# 这样可以将梯度和参数分离，允许多线程的训练器更新参数，而不会在累积到同一个 .grad 上产生数据轨迹
# 注意：这仅应由分布式优化器内部使用，不应暴露给用户使用
@torch.jit.script
class _FunctionalAdamax:
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        foreach: bool = False,
        maximize: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        # 检查学习率是否有效
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        # 检查 epsilon 是否有效
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        # 检查 beta 参数是否有效
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        # 检查权重衰减是否有效
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # 设置默认参数
        self.defaults = {
            "lr": lr,
            "eps": eps,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
        }
        # 是否对每个参数进行操作
        self.foreach = foreach
        # 是否最大化优化目标
        self.maximize = maximize
        # 状态字典，用于存储优化器状态信息
        self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})

        # 如果参数列表为空且不允许空参数列表，则抛出异常
        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # 设置参数组，仅允许一个参数组，并不允许用户添加额外的参数组，因为这不是常见的使用情况
        self.param_group = {"params": params}
    def step(self, gradients: List[Optional[Tensor]]):
        # 获取优化器参数组中的参数列表
        params = self.param_group["params"]
        # 用来存储有梯度的参数
        params_with_grad = []
        # 存储梯度的列表
        grads = []
        # 存储指数加权平均值的列表
        exp_avgs = []
        # 存储指数加权平均平方值的列表
        exp_infs = []
        # 存储步数状态的列表
        state_steps: List[Tensor] = []

        # 检查参数和梯度的长度是否一致
        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        # 检查是否有复数张量
        has_complex = False
        for param, gradient in zip(self.param_group["params"], gradients):
            if gradient is not None:
                # 判断参数是否为复数张量
                has_complex |= torch.is_complex(param)
                # 添加有梯度的参数和对应的梯度到列表中
                params_with_grad.append(param)
                grads.append(gradient)
                # 惰性状态初始化
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    state["step"] = torch.tensor(0.0)
                    # 指数移动平均值
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    # 指数移动平均平方值
                    state["exp_inf"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )

                state = self.state[param]

                # 将指数移动平均值和指数移动平均平方值添加到列表中
                exp_avgs.append(state["exp_avg"])
                exp_infs.append(state["exp_inf"])
                # 将步数状态添加到列表中
                state_steps.append(state["step"])

        # 使用无梯度的上下文执行Adamax优化步骤
        with torch.no_grad():
            F.adamax(
                params_with_grad,
                grads,
                exp_avgs,
                exp_infs,
                state_steps,
                eps=self.defaults["eps"],
                beta1=self.defaults["beta1"],
                beta2=self.defaults["beta2"],
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
            )
```