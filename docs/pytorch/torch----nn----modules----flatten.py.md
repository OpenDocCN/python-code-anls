# `.\pytorch\torch\nn\modules\flatten.py`

```
# mypy: allow-untyped-defs
# 引入必要的类型声明
from typing import Tuple, Union

# 引入 Tensor 类型
from torch import Tensor
# 引入 _size 类型
from torch.types import _size

# 从自定义模块中导入 Module 类
from .module import Module

# 定义模块公开接口列表
__all__ = ["Flatten", "Unflatten"]


class Flatten(Module):
    r"""
    Flattens a contiguous range of dims into a tensor.

    For use with :class:`~nn.Sequential`, see :meth:`torch.flatten` for details.

    Shape:
        - Input: :math:`(*, S_{\text{start}},..., S_{i}, ..., S_{\text{end}}, *)`,
          where :math:`S_{i}` is the size at dimension :math:`i` and :math:`*` means any
          number of dimensions including none.
        - Output: :math:`(*, \prod_{i=\text{start}}^{\text{end}} S_{i}, *)`.

    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Examples::
        >>> input = torch.randn(32, 1, 5, 5)
        >>> # With default parameters
        >>> m = nn.Flatten()
        >>> output = m(input)
        >>> output.size()
        torch.Size([32, 25])
        >>> # With non-default parameters
        >>> m = nn.Flatten(0, 2)
        >>> output = m(input)
        >>> output.size()
        torch.Size([160, 5])
    """

    # 常量列表，指定在序列化和反序列化中保持不变的属性
    __constants__ = ["start_dim", "end_dim"]
    # 类的属性：开始维度和结束维度
    start_dim: int
    end_dim: int

    # 初始化方法
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        # 初始化开始维度和结束维度
        self.start_dim = start_dim
        self.end_dim = end_dim

    # 前向传播方法
    def forward(self, input: Tensor) -> Tensor:
        # 使用 Torch 中的 flatten 方法进行张量展平操作
        return input.flatten(self.start_dim, self.end_dim)

    # 额外的描述方法，用于返回模块的额外信息
    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"


class Unflatten(Module):
    r"""
    Unflattens a tensor dim expanding it to a desired shape. For use with :class:`~nn.Sequential`.

    * :attr:`dim` specifies the dimension of the input tensor to be unflattened, and it can
      be either `int` or `str` when `Tensor` or `NamedTensor` is used, respectively.

    * :attr:`unflattened_size` is the new shape of the unflattened dimension of the tensor and it can be
      a `tuple` of ints or a `list` of ints or `torch.Size` for `Tensor` input;  a `NamedShape`
      (tuple of `(name, size)` tuples) for `NamedTensor` input.

    Shape:
        - Input: :math:`(*, S_{\text{dim}}, *)`, where :math:`S_{\text{dim}}` is the size at
          dimension :attr:`dim` and :math:`*` means any number of dimensions including none.
        - Output: :math:`(*, U_1, ..., U_n, *)`, where :math:`U` = :attr:`unflattened_size` and
          :math:`\prod_{i=1}^n U_i = S_{\text{dim}}`.

    Args:
        dim (Union[int, str]): Dimension to be unflattened
        unflattened_size (Union[torch.Size, Tuple, List, NamedShape]): New shape of the unflattened dimension
    """

    # 初始化方法
    def __init__(self, dim: Union[int, str], unflattened_size: Union[Tuple, List, _size]) -> None:
        super().__init__()
        # 属性：指定的维度和未展平的尺寸
        self.dim = dim
        self.unflattened_size = unflattened_size
    NamedShape = Tuple[Tuple[str, int]]  # 声明一个类型别名 NamedShape，表示由元组组成的元组，每个元组包含一个字符串和一个整数

    __constants__ = ["dim", "unflattened_size"]
    dim: Union[int, str]  # dim 可以是整数或字符串类型
    unflattened_size: Union[_size, NamedShape]  # unflattened_size 可以是 _size 类型或 NamedShape 类型

    def __init__(
        self, dim: Union[int, str], unflattened_size: Union[_size, NamedShape]
    ) -> None:
        super().__init__()  # 调用父类的构造方法

        if isinstance(dim, int):
            self._require_tuple_int(unflattened_size)  # 如果 dim 是整数，确保 unflattened_size 是元组类型的整数
        elif isinstance(dim, str):
            self._require_tuple_tuple(unflattened_size)  # 如果 dim 是字符串，确保 unflattened_size 是字符串元组类型
        else:
            raise TypeError("invalid argument type for dim parameter")  # 如果 dim 不是整数或字符串，抛出类型错误异常

        self.dim = dim  # 将 dim 赋值给对象属性
        self.unflattened_size = unflattened_size  # 将 unflattened_size 赋值给对象属性

    def _require_tuple_tuple(self, input):
        if isinstance(input, tuple):  # 如果 input 是元组
            for idx, elem in enumerate(input):  # 遍历元组中的每个元素
                if not isinstance(elem, tuple):  # 如果元素不是元组类型
                    raise TypeError(
                        "unflattened_size must be tuple of tuples, "  # 抛出类型错误异常，要求 unflattened_size 必须是元组的元组
                        + f"but found element of type {type(elem).__name__} at pos {idx}"
                    )
            return
        raise TypeError(
            "unflattened_size must be a tuple of tuples, "  # 如果 input 不是元组类型，抛出类型错误异常，要求 unflattened_size 必须是元组的元组
            + f"but found type {type(input).__name__}"
        )

    def _require_tuple_int(self, input):
        if isinstance(input, (tuple, list)):  # 如果 input 是元组或列表
            for idx, elem in enumerate(input):  # 遍历元组或列表中的每个元素
                if not isinstance(elem, int):  # 如果元素不是整数类型
                    raise TypeError(
                        "unflattened_size must be tuple of ints, "  # 抛出类型错误异常，要求 unflattened_size 必须是整数的元组
                        + f"but found element of type {type(elem).__name__} at pos {idx}"
                    )
            return
        raise TypeError(
            f"unflattened_size must be a tuple of ints, but found type {type(input).__name__}"  # 如果 input 不是元组或列表类型，抛出类型错误异常，要求 unflattened_size 必须是整数的元组
        )

    def forward(self, input: Tensor) -> Tensor:
        return input.unflatten(self.dim, self.unflattened_size)  # 调用 Tensor 对象的 unflatten 方法，对输入进行重新组织

    def extra_repr(self) -> str:
        return f"dim={self.dim}, unflattened_size={self.unflattened_size}"  # 返回对象的字符串表示形式，包括 dim 和 unflattened_size 属性
```