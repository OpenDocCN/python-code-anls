# `.\pytorch\torch\distributed\optim\functional_rprop.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型定义
from typing import Dict, List, Optional, Tuple

# 导入 PyTorch 相关模块
import torch
import torch.optim._functional as F
from torch import Tensor

# 声明一个空列表，用于存储模块导出的变量名
__all__: List[str] = []

# 定义一个 TorchScript 兼容的 Functional Rprop 优化器
# 在这里我们以函数式的方式使用优化器。
# 不像常规方法那样使用 `param.grad` 来更新参数，
# 我们明确允许分布式优化器将梯度传递给 `step` 函数。
# 这样一来，我们可以将梯度和参数分开，并允许多线程训练器在更新参数时无需数据轨迹累积到相同的 `.grad` 上。
# 注意：这应该只在分布式优化器的内部使用，而不是暴露给用户使用。
@torch.jit.script
class _FunctionalRprop:
    def __init__(
        self,
        params: List[Tensor],  # 参数列表，包含要优化的张量
        lr: float = 1e-2,  # 学习率，默认为 0.01
        etas: Tuple[float, float] = (0.5, 1.2),  # etas 参数，元组形式，默认为 (0.5, 1.2)
        step_sizes: Tuple[float, float] = (1e-6, 50),  # step_sizes 参数，元组形式，默认为 (1e-6, 50)
        foreach: bool = False,  # foreach 标志，默认为 False
        maximize: bool = False,  # maximize 标志，默认为 False
        _allow_empty_param_list: bool = False,  # 是否允许空参数列表，默认为 False
    ):
        # 设置默认参数字典，包含学习率 lr
        self.defaults = {
            "lr": lr,
        }
        # 设置 etas 参数
        self.etas = etas
        # 设置 step_sizes 参数
        self.step_sizes = step_sizes
        # 设置 foreach 标志
        self.foreach = foreach
        # 设置 maximize 标志
        self.maximize = maximize

        # 如果参数列表为空且不允许空参数列表，则抛出 ValueError 异常
        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        # 设置参数组 param_group，只包含一个 params 键，值为传入的参数列表
        self.param_group = {"params": params}

        # 初始化状态字典，用于存储张量与其状态的键值对
        self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})
    def step(self, gradients: List[Optional[Tensor]]):
        # 获取优化器参数组中的参数列表
        params = self.param_group["params"]
        # 用于存储有梯度的参数
        params_with_grad = []
        # 用于存储梯度
        grads = []
        # 用于存储上一步的参数值
        prevs = []
        # 用于存储步长大小
        step_sizes = []
        # 用于存储状态步数
        state_steps = []
        # 获取学习率
        lr = self.defaults["lr"]
        # 获取 etaminus 和 etaplus 参数
        etaminus, etaplus = self.etas
        # 获取 step_size_min 和 step_size_max 参数
        step_size_min, step_size_max = self.step_sizes

        # 检查参数和梯度列表长度是否相等
        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        # 检查是否存在复数参数
        has_complex = False
        for param, gradient in zip(params, gradients):
            if gradient is not None:
                has_complex |= torch.is_complex(param)
                params_with_grad.append(param)
                grads.append(gradient)
                # 懒惰状态初始化
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    state["step"] = torch.tensor(0.0)
                    state["prev"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["step_size"] = torch.full_like(gradient, lr)

                state = self.state[param]
                prevs.append(state["prev"])
                step_sizes.append(state["step_size"])
                state_steps.append(state["step"])

        # 使用 Rprop 算法更新参数
        with torch.no_grad():
            F.rprop(
                params_with_grad,
                grads,
                prevs,
                step_sizes,
                state_steps,
                step_size_min=step_size_min,
                step_size_max=step_size_max,
                etaminus=etaminus,
                etaplus=etaplus,
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
            )
```