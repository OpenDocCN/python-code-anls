# `.\pytorch\torch\distributed\optim\functional_adagrad.py`

```py
# mypy: allow-untyped-defs
# 导入所需的类型注解
from typing import Dict, List, Optional

# 导入 PyTorch 库
import torch
# 导入 PyTorch 优化器的函数接口
import torch.optim._functional as F
# 导入 Tensor 类型
from torch import Tensor

# 定义一个 TorchScript 兼容的函数式 Adagrad 优化器
# 在这里我们以函数式的方式使用这个优化器。
# 不使用 `param.grad` 更新参数，而是明确让用户将梯度传递给 `step` 函数
# 这样可以分离梯度和参数，并允许多线程训练器更新参数，
# 而不会在累积到同一个 `.grad` 上留下数据痕迹。
# 注意: 这应该仅由分布式优化器内部使用，不应该暴露给用户。
@torch.jit.script
class _FunctionalAdagrad:
    def __init__(
        self,
        params: List[Tensor],  # 参数列表，包含需要优化的张量
        lr: float = 1e-2,  # 学习率，默认为 0.01
        lr_decay: float = 0.0,  # 学习率衰减，默认为 0.0
        weight_decay: float = 0.0,  # 权重衰减（L2正则化）默认为 0.0
        initial_accumulator_value: float = 0.0,  # 初始累加器值，默认为 0.0
        warmup_lr_multiplier: float = 1.0,  # 热身期学习率倍增因子，默认为 1.0
        warmup_num_iters: float = 0.0,  # 热身期迭代次数，默认为 0.0
        eps: float = 1e-10,  # 避免除零的小值，默认为 1e-10
        coalesce_grad: bool = True,  # 是否合并梯度，默认为 True
        foreach: bool = False,  # 是否对每个参数单独更新，默认为 False
        fused: bool = False,  # 是否使用融合的操作，默认为 False
        maximize: bool = False,  # 是否最大化优化，默认为 False
        _allow_empty_param_list: bool = False,  # 是否允许空参数列表，默认为 False
    ):
        # 设置默认参数字典
        self.defaults = {
            "lr": lr,
            "lr_decay": lr_decay,
            "eps": eps,
            "weight_decay": weight_decay,
            "initial_accumulator_value": initial_accumulator_value,
            "warmup_lr_multiplier": warmup_lr_multiplier,
            "warmup_num_iters": warmup_num_iters,
        }
        # 是否合并梯度
        self.coalesce_grad = coalesce_grad
        # 是否对每个参数单独更新
        self.foreach = foreach
        # 是否使用融合的操作
        self.fused = fused
        # 是否最大化优化
        self.maximize = maximize
        # 状态字典，用于存储每个参数的额外状态信息
        self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})

        # 如果参数列表为空并且不允许空参数列表，则抛出错误
        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # 注意: 我们只有一个参数组，并且不允许用户添加额外的参数组，因为这不是常见用例。
        self.param_group = {"params": params}

        # 循环遍历参数列表，初始化每个参数的状态字典
        for p in self.param_group["params"]:
            self.state[p] = {
                "sum": torch.full_like(p.data, initial_accumulator_value),  # 累加器，初始化为指定值
                "step": torch.tensor(0.0),  # 步数，初始化为零
            }
    # 定义一个方法，用于执行优化步骤，接受梯度列表作为输入参数
    def step(self, gradients: List[Optional[Tensor]]):
        # 从参数组中获取参数列表
        params = self.param_group["params"]
        # 用于存储有梯度的参数列表
        params_with_grad = []
        # 存储梯度的列表
        grads = []
        # 存储状态和步数的和的列表
        state_sums = []
        # 存储状态步数的列表
        state_steps: List[Tensor] = []

        # 检查参数和梯度的长度是否一致，如果不一致则抛出异常
        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        # 初始化是否有稀疏梯度和复杂参数的标志
        has_sparse_grad, has_complex = False, False
        # 遍历参数和梯度的组合
        for param, gradient in zip(self.param_group["params"], gradients):
            # 如果梯度不为None
            if gradient is not None:
                # 更新稀疏梯度和复杂参数的标志
                has_sparse_grad |= gradient.is_sparse
                has_complex |= torch.is_complex(param)
                # 将有梯度的参数加入列表中
                params_with_grad.append(param)
                # 将梯度加入梯度列表中
                grads.append(gradient)
                # 获取当前参数的状态
                state = self.state[param]
                # 将状态中的和加入状态和的列表
                state_sums.append(state["sum"])
                # 将状态中的步数加入状态步数的列表
                state_steps.append(state["step"])

        # 使用torch.no_grad()上下文管理器，确保在此过程中不计算梯度
        with torch.no_grad():
            # 调用F.adagrad进行Adagrad优化步骤
            F.adagrad(
                params,
                grads,
                state_sums,
                state_steps,
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                lr_decay=self.defaults["lr_decay"],
                eps=self.defaults["eps"],
                has_sparse_grad=has_sparse_grad,
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
            )
```