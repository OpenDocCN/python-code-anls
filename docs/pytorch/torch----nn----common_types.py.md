# `.\pytorch\torch\nn\common_types.py`

```
# 导入必要的类型声明
from typing import Optional, Tuple, TypeVar, Union

# 导入 PyTorch 的 Tensor 类型
from torch import Tensor

# 创建一些有用的类型别名

# 用于可以作为元组提供的参数模板，或者可以作为标量的参数，PyTorch 会内部将其广播到元组。
# 有几个变体：未知大小的元组，以及用于 1d、2d 或 3d 操作的固定大小元组。
T = TypeVar("T")
_scalar_or_tuple_any_t = Union[T, Tuple[T, ...]]
_scalar_or_tuple_1_t = Union[T, Tuple[T]]
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_scalar_or_tuple_3_t = Union[T, Tuple[T, T, T]]
_scalar_or_tuple_4_t = Union[T, Tuple[T, T, T, T]]
_scalar_or_tuple_5_t = Union[T, Tuple[T, T, T, T, T]]
_scalar_or_tuple_6_t = Union[T, Tuple[T, T, T, T, T, T]]

# 用于表示尺寸参数的参数（例如，内核大小、填充）
_size_any_t = _scalar_or_tuple_any_t[int]
_size_1_t = _scalar_or_tuple_1_t[int]
_size_2_t = _scalar_or_tuple_2_t[int]
_size_3_t = _scalar_or_tuple_3_t[int]
_size_4_t = _scalar_or_tuple_4_t[int]
_size_5_t = _scalar_or_tuple_5_t[int]
_size_6_t = _scalar_or_tuple_6_t[int]

# 用于表示可选尺寸参数的参数（例如，自适应池化参数）
_size_any_opt_t = _scalar_or_tuple_any_t[Optional[int]]
_size_2_opt_t = _scalar_or_tuple_2_t[Optional[int]]
_size_3_opt_t = _scalar_or_tuple_3_t[Optional[int]]

# 用于表示调整输入每个维度比率的参数（例如，上采样参数）
_ratio_2_t = _scalar_or_tuple_2_t[float]
_ratio_3_t = _scalar_or_tuple_3_t[float]
_ratio_any_t = _scalar_or_tuple_any_t[float]

# 用于表示 Tensor 列表的返回值类型
_tensor_list_t = _scalar_or_tuple_any_t[Tensor]

# 用于可能返回索引的最大池化操作的返回值类型。
# 使用建议的 'Literal' 功能进行 Python 类型化，最终可能可以消除这些定义。
_maybe_indices_t = _scalar_or_tuple_2_t[Tensor]
```