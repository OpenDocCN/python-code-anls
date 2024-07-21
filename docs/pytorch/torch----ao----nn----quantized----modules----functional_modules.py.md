# `.\pytorch\torch\ao\nn\quantized\modules\functional_modules.py`

```py
# mypy: allow-untyped-defs
# 引入类型定义 List
from typing import List

# 引入 PyTorch 库
import torch
# 从 torch 中导入 Tensor 类型
from torch import Tensor
# 从 torch._ops 中导入 ops 模块
from torch._ops import ops

# 定义模块中公开的类列表
__all__ = ['FloatFunctional', 'FXFloatFunctional', 'QFunctional']

# FloatFunctional 类，继承自 torch.nn.Module
class FloatFunctional(torch.nn.Module):
    r"""State collector class for float operations.

    The instance of this class can be used instead of the ``torch.`` prefix for
    some operations. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    Examples::

        >>> f_add = FloatFunctional()
        >>> a = torch.tensor(3.0)
        >>> b = torch.tensor(4.0)
        >>> f_add.add(a, b)  # Equivalent to ``torch.add(a, b)``

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """
    
    # 初始化方法，设置 activation_post_process 为 Identity 模块
    def __init__(self):
        super().__init__()
        self.activation_post_process = torch.nn.Identity()

    # 前向传播方法，抛出运行时错误提示
    def forward(self, x):
        raise RuntimeError("FloatFunctional is not intended to use the " +
                           "'forward'. Please use the underlying operation")

    # add 方法，对应 torch.add(Tensor, Tensor)，并应用 activation_post_process
    r"""Operation equivalent to ``torch.add(Tensor, Tensor)``"""
    def add(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.add(x, y)
        r = self.activation_post_process(r)
        return r

    # add_scalar 方法，对应 torch.add(Tensor, float)，并应用 activation_post_process
    r"""Operation equivalent to ``torch.add(Tensor, float)``"""
    def add_scalar(self, x: Tensor, y: float) -> Tensor:
        r = torch.add(x, y)
        # 注意：此操作未被观察，因为观察不需要对量化操作进行观察。
        return r

    # mul 方法，对应 torch.mul(Tensor, Tensor)，并应用 activation_post_process
    r"""Operation equivalent to ``torch.mul(Tensor, Tensor)``"""
    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.mul(x, y)
        r = self.activation_post_process(r)
        return r

    # mul_scalar 方法，对应 torch.mul(Tensor, float)，并应用 activation_post_process
    r"""Operation equivalent to ``torch.mul(Tensor, float)``"""
    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        r = torch.mul(x, y)
        # 注意：此操作未被观察，因为观察不需要对量化操作进行观察。
        return r

    # cat 方法，对应 torch.cat 操作，并应用 activation_post_process
    r"""Operation equivalent to ``torch.cat``"""
    def cat(self, x: List[Tensor], dim: int = 0) -> Tensor:
        r = torch.cat(x, dim=dim)
        r = self.activation_post_process(r)
        return r

    # add_relu 方法，对应 relu(torch.add(x,y)) 操作，并应用 activation_post_process
    r"""Operation equivalent to ``relu(torch.add(x,y))``"""
    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.add(x, y)
        r = torch.nn.functional.relu(r)
        r = self.activation_post_process(r)
        return r

    # matmul 方法，对应 torch.matmul(Tensor, Tensor) 操作，并应用 activation_post_process
    r"""Operation equivalent to ``torch.matmul(Tensor, Tensor)``"""
    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.matmul(x, y)
        r = self.activation_post_process(r)
        return r

# FXFloatFunctional 类，继承自 torch.nn.Module，用于 FX 图模式量化之前替代 FloatFunctional 模块
class FXFloatFunctional(torch.nn.Module):
    r""" module to replace FloatFunctional module before FX graph mode quantization,
    since activation_post_process will be inserted in top level module directly

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """
    定义了一个名为FloatFunctional的类，用于封装一些与张量操作相关的功能。

    r"""Operation equivalent to ``torch.add(Tensor, Tensor)``"""
    定义了一个方法add，实现了两个张量相加的操作，并返回结果张量。

    r"""Operation equivalent to ``torch.add(Tensor, float)``"""
    定义了一个方法add_scalar，实现了张量与标量相加的操作，并返回结果张量。

    r"""Operation equivalent to ``torch.mul(Tensor, Tensor)``"""
    定义了一个方法mul，实现了两个张量相乘的操作，并返回结果张量。

    r"""Operation equivalent to ``torch.mul(Tensor, float)``"""
    定义了一个方法mul_scalar，实现了张量与标量相乘的操作，并返回结果张量。

    r"""Operation equivalent to ``torch.cat``"""
    定义了一个方法cat，实现了张量列表沿指定维度拼接的操作，并返回结果张量。

    r"""Operation equivalent to ``relu(torch.add(x,y))``"""
    定义了一个方法add_relu，实现了两个张量相加后应用ReLU激活函数的操作，并返回结果张量。

    r"""Operation equivalent to ``torch.matmul(Tensor, Tensor)``"""
    定义了一个方法matmul，实现了两个张量矩阵乘法的操作，并返回结果张量。
class QFunctional(torch.nn.Module):
    r"""Wrapper class for quantized operations.

    The instance of this class can be used instead of the
    ``torch.ops.quantized`` prefix. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    Examples::

        >>> q_add = QFunctional()
        >>> # xdoctest: +SKIP
        >>> a = torch.quantize_per_tensor(torch.tensor(3.0), 1.0, 0, torch.qint32)
        >>> b = torch.quantize_per_tensor(torch.tensor(4.0), 1.0, 0, torch.qint32)
        >>> q_add.add(a, b)  # Equivalent to ``torch.ops.quantized.add(a, b, 1.0, 0)``

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

    def __init__(self):
        super().__init__()
        self.scale = 1.0  # 初始化量化比例为 1.0
        self.zero_point = 0  # 初始化零点为 0
        self.activation_post_process = torch.nn.Identity()  # 默认使用恒等函数作为激活后处理器

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = torch.tensor(self.scale)  # 将量化比例保存到状态字典中
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)  # 将零点保存到状态字典中

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.scale = float(state_dict.pop(prefix + 'scale'))  # 从状态字典中加载并设置量化比例
        self.zero_point = int(state_dict.pop(prefix + 'zero_point'))  # 从状态字典中加载并设置零点
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                      missing_keys, unexpected_keys, error_msgs)

    def _get_name(self):
        return 'QFunctional'  # 返回类的名称字符串表示

    def extra_repr(self):
        return f'scale={self.scale}, zero_point={self.zero_point}'  # 返回对象的额外字符串表示，包括量化比例和零点

    def forward(self, x):
        raise RuntimeError("Functional is not intended to use the " +
                           "'forward'. Please use the underlying operation")  # 前向传播方法抛出运行时错误，提示不应直接使用前向传播方法

    r"""Operation equivalent to ``torch.ops.quantized.add``"""
    def add(self, x: Tensor, y: Tensor) -> Tensor:
        r = ops.quantized.add(x, y, scale=self.scale, zero_point=self.zero_point)  # 调用 quantized.add 函数，传入量化参数
        r = self.activation_post_process(r)  # 应用激活后处理器
        return r

    r"""Operation equivalent to ``torch.ops.quantized.add(Tensor, float)``"""
    def add_scalar(self, x: Tensor, y: float) -> Tensor:
        r = ops.quantized.add_scalar(x, y)  # 调用 quantized.add_scalar 函数，不涉及量化参数
        # Note: this operation is not observed because the observation is not
        # needed for the quantized op.
        return r

    r"""Operation equivalent to ``torch.ops.quantized.mul(Tensor, Tensor)``"""
    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        r = ops.quantized.mul(x, y, scale=self.scale, zero_point=self.zero_point)  # 调用 quantized.mul 函数，传入量化参数
        r = self.activation_post_process(r)  # 应用激活后处理器
        return r

    r"""Operation equivalent to ``torch.ops.quantized.mul(Tensor, float)``"""
    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        r = ops.quantized.mul_scalar(x, y)  # 调用 quantized.mul_scalar 函数，不涉及量化参数
        # Note: this operation is not observed because the observation is not
        # needed for the quantized op.
        return r
    # 使用 ops.quantized.mul_scalar 函数对张量 x 进行标量乘法运算 y
    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        r = ops.quantized.mul_scalar(x, y)
        # 注意：此操作未被记录，因为对量化操作不需要观察。
        return r

    # 等效于 torch.ops.quantized.cat 的操作
    def cat(self, x: List[Tensor], dim: int = 0) -> Tensor:
        r = ops.quantized.cat(x, scale=self.scale, zero_point=self.zero_point, dim=dim)
        # 对结果张量 r 进行激活后处理
        r = self.activation_post_process(r)
        return r

    # 等效于 torch.ops.quantized.add_relu 的操作
    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        r = ops.quantized.add_relu(x, y, scale=self.scale, zero_point=self.zero_point)
        # 对结果张量 r 进行激活后处理
        r = self.activation_post_process(r)
        return r

    # 等效于 torch.ops.quantized.matmul(Tensor, Tensor) 的操作
    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        r = ops.quantized.matmul(x, y, scale=self.scale, zero_point=self.zero_point)
        # 注意：此操作未被记录，因为对量化操作不需要观察。
        return r

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        assert type(mod) == FloatFunctional, \
            "QFunctional.from_float expects an instance of FloatFunctional"
        # 计算模块的量化参数 scale 和 zero_point
        scale, zero_point = mod.activation_post_process.calculate_qparams()  # type: ignore[operator]
        # 创建一个新的 QFunctional 对象
        new_mod = QFunctional()
        new_mod.scale = float(scale)
        new_mod.zero_point = int(zero_point)
        return new_mod
```