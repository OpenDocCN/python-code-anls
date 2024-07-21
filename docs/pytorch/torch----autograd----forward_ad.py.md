# `.\pytorch\torch\autograd\forward_ad.py`

```py
# mypy: allow-untyped-defs
# 导入标准库和第三方库
import os
from collections import namedtuple

# 导入类型提示
from typing import Any

# 导入PyTorch库
import torch
from .grad_mode import _DecoratorContextManager

# 暴露给外部的API列表
__all__ = [
    "UnpackedDualTensor",
    "enter_dual_level",
    "exit_dual_level",
    "make_dual",
    "unpack_dual",
    "dual_level",
]

# 全局变量，用于简化Python API的使用
_current_level = -1


def enter_dual_level():
    r"""进入一个新的前向梯度级别。

    这个级别可以用于创建和解包双重张量以计算前向梯度。

    此函数还更新了默认情况下其他API函数使用的当前级别。
    """
    global _current_level
    # 调用底层torch._C._enter_dual_level()来进入新的双重级别
    new_level = torch._C._enter_dual_level()
    if new_level != _current_level + 1:
        # 如果进入的新级别不是当前级别加1，则抛出运行时错误
        raise RuntimeError(
            "Entering a new forward AD level but the current level "
            "is not valid. Make sure you did not modified it directly."
        )
    _current_level = new_level
    return new_level


def exit_dual_level(*, level=None):
    r"""退出一个前向梯度级别。

    此函数删除与此级别关联的所有梯度。只允许删除最近进入的级别。

    此函数还更新了默认情况下其他API函数使用的当前级别。
    """
    global _current_level
    if level is None:
        level = _current_level
    if level != _current_level:
        # 如果要退出的级别不是当前级别，则抛出运行时错误
        raise RuntimeError(
            "Trying to exit a forward AD level that was not the last one "
            "that was created. This is not supported."
        )
    # 调用底层torch._C._exit_dual_level()来退出双重级别
    torch._C._exit_dual_level(level=level)
    _current_level = level - 1


def _maybe_load_decompositions():
    if os.environ.get("PYTORCH_JIT", "1") == "1" and __debug__:
        # 根据环境变量和调试模式，动态导入torch._decomp模块中的decompositions_for_jvp函数
        from torch._decomp import decompositions_for_jvp  # noqa: F401


def make_dual(tensor, tangent, *, level=None):
    r"""将张量值与其切向量关联，创建一个“双重张量”用于前向自动微分梯度计算。

    结果是一个新的张量，与 :attr:`tensor` 别名相同，带有 :attr:`tangent` 嵌入为一个属性，
    如果具有相同的存储布局则原样，否则复制。可以使用 :func:`unpack_dual` 恢复切向量属性。

    此函数支持反向微分。

    给定一个函数 `f`，其雅可比矩阵为 `J`，它允许计算 `J` 和给定向量 `v` 之间的雅可比向量积 (`jvp`)。

    示例::

        >>> # xdoctest: +SKIP("Undefined variables")
        >>> with dual_level():
        ...     inp = make_dual(x, v)
        ...     out = f(inp)
        ...     y, jvp = unpack_dual(out)

    请参阅 `前向自动微分教程 <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    了解如何使用此API的详细步骤。
    """
    # 见注释: [forward-mode AD decompositions mechanism]
    #
    # 从torch._decomp模块中导入decompositions_for_jvp函数以注册
    # 加载可能需要的分解到 JIT 注册表中
    #
    # FIXME: 我们要求 __debug__ 必须为 True，因为如果 Python 被运行时带有 -OO 或 -O 标志（即 __debug__ 为 False），
    # 我们会遇到以下错误：
    #
    # 返回值被注释为类型 Tuple[NoneType, NoneType]，但实际上是类型 Tuple[Tensor, Tensor]：
    #   文件 ".../torch/_decomp/__init__.py"，第 1585 行
    #     else:
    #         buffer = z
    #     return min - torch.log1p(z), buffer
    #     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- 这里
    _maybe_load_decompositions()

    # 如果 level 为 None，则使用当前的 _current_level
    if level is None:
        level = _current_level

    # 如果 level 小于 0，则抛出运行时错误
    if level < 0:
        raise RuntimeError(
            "Trying to create a dual Tensor for forward AD but no level "
            "exists, make sure to enter_dual_level() first."
        )

    # 检查 tensor 是否是浮点数或复数类型，如果不是则抛出值错误
    if not (tensor.is_floating_point() or tensor.is_complex()):
        raise ValueError(
            f"Expected primal to be floating point or complex, but got: {tensor.dtype}"
        )

    # 检查 tangent 是否是浮点数或复数类型，如果不是则抛出值错误
    if not (tangent.is_floating_point() or tangent.is_complex()):
        raise ValueError(
            f"Expected tangent to be floating point or complex, but got: {tangent.dtype}"
        )

    # 调用 torch._VF._make_dual 方法创建双重 Tensor，并返回结果
    return torch._VF._make_dual(tensor, tangent, level=level)
# 定义了一个命名元组 `_UnpackedDualTensor`，包含两个字段 "primal" 和 "tangent"
_UnpackedDualTensor = namedtuple("_UnpackedDualTensor", ["primal", "tangent"])

# 继承自 `_UnpackedDualTensor` 的类 `UnpackedDualTensor`
class UnpackedDualTensor(_UnpackedDualTensor):
    r"""Namedtuple returned by :func:`unpack_dual` containing the primal and tangent components of the dual tensor.

    See :func:`unpack_dual` for more details.

    """
    # 此类继承了命名元组 `_UnpackedDualTensor`，用于存储由 `unpack_dual` 函数返回的双重张量的原始和切向分量

    pass  # 类体为空

# 解包双重张量的函数 `unpack_dual`
def unpack_dual(tensor, *, level=None):
    r"""Unpack a "dual tensor" to get both its Tensor value and its forward AD gradient.

    The result is a namedtuple ``(primal, tangent)`` where ``primal`` is a view of
    :attr:`tensor`'s primal and ``tangent`` is :attr:`tensor`'s tangent as-is.
    Neither of these tensors can be dual tensor of level :attr:`level`.

    This function is backward differentiable.

    Example::

        >>> # xdoctest: +SKIP("Undefined variables")
        >>> with dual_level():
        ...     inp = make_dual(x, x_t)
        ...     out = f(inp)
        ...     y, jvp = unpack_dual(out)
        ...     jvp = unpack_dual(out).tangent

    Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.
    """
    # 如果未指定 level，默认使用当前级别 `_current_level`
    if level is None:
        level = _current_level

    # 如果 level 小于 0，返回一个只有 `primal` 字段的 `UnpackedDualTensor`
    if level < 0:
        return UnpackedDualTensor(tensor, None)

    # 调用 PyTorch 私有函数 `_VF._unpack_dual` 解包双重张量
    primal, dual = torch._VF._unpack_dual(tensor, level=level)

    # 返回一个 `UnpackedDualTensor` 对象，包含 `primal` 和 `dual`
    return UnpackedDualTensor(primal, dual)


# 上下文管理器 `dual_level`，用于控制前向自动微分的上下文
class dual_level(_DecoratorContextManager):
    r"""Context-manager for forward AD, where all forward AD computation must occur within the ``dual_level`` context.

    .. Note::

        The ``dual_level`` context appropriately enters and exit the dual level to
        controls the current forward AD level, which is used by default by the other
        functions in this API.

        We currently don't plan to support nested ``dual_level`` contexts, however, so
        only a single forward AD level is supported. To compute higher-order
        forward grads, one can use :func:`torch.func.jvp`.

    Example::

        >>> # xdoctest: +SKIP("Undefined variables")
        >>> x = torch.tensor([1])
        >>> x_t = torch.tensor([1])
        >>> with dual_level():
        ...     inp = make_dual(x, x_t)
        ...     # Do computations with inp
        ...     out = your_fn(inp)
        ...     _, grad = unpack_dual(out)
        >>> grad is None
        False
        >>> # After exiting the level, the grad is deleted
        >>> _, grad_after = unpack_dual(out)
        >>> grad is None
        True

    Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.
    """

    # 进入 `dual_level` 上下文时调用的方法
    def __enter__(self):
        return enter_dual_level()

    # 退出 `dual_level` 上下文时调用的方法
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        exit_dual_level()


# 私有辅助函数 `_is_fwd_grad_enabled`，用于检查前向梯度是否启用
_is_fwd_grad_enabled = torch._C._is_fwd_grad_enabled


# 私有辅助函数，用于启用或禁用前向梯度
# 这里代码被省略，但是注释要求不允许省略任何代码部分
# 如果您是用户并且希望使用这段代码，请提交一个问题来讨论使用案例。
class _set_fwd_grad_enabled(_DecoratorContextManager):
    # 初始化方法，设置前向梯度是否启用的装饰器上下文管理器
    def __init__(self, mode: bool) -> None:
        # 保存当前的前向梯度状态
        self.prev = _is_fwd_grad_enabled()
        # 调用 PyTorch C++ 扩展函数设置前向梯度是否启用
        torch._C._set_fwd_grad_enabled(mode)

    def __enter__(self) -> None:
        # 进入上下文时不执行任何操作
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        # 退出上下文时恢复之前保存的前向梯度状态
        torch._C._set_fwd_grad_enabled(self.prev)
```