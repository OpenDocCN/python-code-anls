# `.\pytorch\torch\nn\utils\weight_norm.py`

```
# mypy: allow-untyped-defs
r"""Weight Normalization from https://arxiv.org/abs/1602.07868."""
# 导入必要的类型和函数
from typing import Any, TypeVar
from typing_extensions import deprecated

# 导入 torch 相关模块和函数
from torch import _weight_norm, norm_except_dim
from torch.nn.modules import Module
from torch.nn.parameter import Parameter, UninitializedParameter

# 定义可以被导入的符号列表
__all__ = ["WeightNorm", "weight_norm", "remove_weight_norm"]

# 定义 WeightNorm 类
class WeightNorm:
    name: str  # 存储参数名
    dim: int   # 存储维度信息

    def __init__(self, name: str, dim: int) -> None:
        # 如果未指定维度，默认设置为 -1
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    # TODO Make return type more specific
    def compute_weight(self, module: Module) -> Any:
        # 计算参数的权重标准化值
        g = getattr(module, self.name + "_g")
        v = getattr(module, self.name + "_v")
        return _weight_norm(v, g, self.dim)

    @staticmethod
    @deprecated(
        "`torch.nn.utils.weight_norm` is deprecated "
        "in favor of `torch.nn.utils.parametrizations.weight_norm`.",
        category=FutureWarning,
    )
    def apply(module, name: str, dim: int) -> "WeightNorm":
        # 应用权重标准化到指定模块的参数
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError(
                    f"Cannot register two weight_norm hooks on the same parameter {name}"
                )

        if dim is None:
            dim = -1

        fn = WeightNorm(name, dim)

        weight = getattr(module, name)
        if isinstance(weight, UninitializedParameter):
            raise ValueError(
                "The module passed to `WeightNorm` can't have uninitialized parameters. "
                "Make sure to run the dummy forward before applying weight normalization"
            )
        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(
            name + "_g", Parameter(norm_except_dim(weight, 2, dim).data)
        )
        module.register_parameter(name + "_v", Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module: Module) -> None:
        # 从模块中移除权重标准化的参数
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + "_g"]
        del module._parameters[self.name + "_v"]
        setattr(module, self.name, Parameter(weight.data))

    def __call__(self, module: Module, inputs: Any) -> None:
        # 使 WeightNorm 对象可以像函数一样调用，应用权重标准化到模块
        setattr(module, self.name, self.compute_weight(module))


T_module = TypeVar("T_module", bound=Module)

def weight_norm(module: T_module, name: str = "weight", dim: int = 0) -> T_module:
    r"""Apply weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    """
    # 对给定模块中的参数应用权重标准化
    # 应用权重归一化到指定的模块和权重参数
    WeightNorm.apply(module, name, dim)
    # 返回已应用权重归一化钩子的原始模块
    return module
# 从模块中移除权重归一化重新参数化的方法
def remove_weight_norm(module: T_module, name: str = "weight") -> T_module:
    r"""Remove the weight normalization reparameterization from a module.

    Args:
        module (Module): containing module  # 接收包含权重参数的模块对象
        name (str, optional): name of weight parameter  # 权重参数的名称，默认为"weight"

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))  # 对 nn.Linear(20, 40) 应用权重归一化
        >>> remove_weight_norm(m)  # 移除权重归一化
    """
    # 遍历模块的前向钩子列表
    for k, hook in module._forward_pre_hooks.items():
        # 检查钩子是否为 WeightNorm 类型且名称匹配指定的名称
        if isinstance(hook, WeightNorm) and hook.name == name:
            # 调用钩子对象的 remove 方法，移除权重归一化
            hook.remove(module)
            # 从前向钩子字典中删除该钩子项
            del module._forward_pre_hooks[k]
            # 返回处理后的模块对象
            return module

    # 如果未找到匹配的权重归一化钩子，则引发 ValueError 异常
    raise ValueError(f"weight_norm of '{name}' not found in {module}")
```