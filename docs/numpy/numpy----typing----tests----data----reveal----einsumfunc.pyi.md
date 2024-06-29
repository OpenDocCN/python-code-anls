# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\einsumfunc.pyi`

```
# 导入 sys 模块，用于版本信息检查
import sys
# 导入 Any 类型，用于类型提示
from typing import Any

# 导入 numpy 库
import numpy as np
# 导入 numpy.typing 库，用于类型提示
import numpy.typing as npt

# 如果 Python 版本大于等于 3.11，导入 typing 模块的 assert_type 函数
if sys.version_info >= (3, 11):
    from typing import assert_type
# 否则，从 typing_extensions 模块导入 assert_type 函数
else:
    from typing_extensions import assert_type

# 定义不同类型的列表变量
AR_LIKE_b: list[bool]
AR_LIKE_u: list[np.uint32]
AR_LIKE_i: list[int]
AR_LIKE_f: list[float]
AR_LIKE_c: list[complex]
AR_LIKE_U: list[str]
AR_o: npt.NDArray[np.object_]

# 定义输出类型为 np.float64 的数组变量
OUT_f: npt.NDArray[np.float64]

# 使用 assert_type 函数验证 np.einsum 函数的返回类型
assert_type(np.einsum("i,i->i", AR_LIKE_b, AR_LIKE_b), Any)
assert_type(np.einsum("i,i->i", AR_o, AR_o), Any)
assert_type(np.einsum("i,i->i", AR_LIKE_u, AR_LIKE_u), Any)
assert_type(np.einsum("i,i->i", AR_LIKE_i, AR_LIKE_i), Any)
assert_type(np.einsum("i,i->i", AR_LIKE_f, AR_LIKE_f), Any)
assert_type(np.einsum("i,i->i", AR_LIKE_c, AR_LIKE_c), Any)
assert_type(np.einsum("i,i->i", AR_LIKE_b, AR_LIKE_i), Any)
assert_type(np.einsum("i,i,i,i->i", AR_LIKE_b, AR_LIKE_u, AR_LIKE_i, AR_LIKE_c), Any)

# 使用 assert_type 函数验证 np.einsum 函数的返回类型，并指定输出数组类型为 np.float64
assert_type(np.einsum("i,i->i", AR_LIKE_c, AR_LIKE_c, out=OUT_f), npt.NDArray[np.float64])
# 使用 assert_type 函数验证 np.einsum 函数的返回类型，并指定输入 dtype 和 casting 类型
assert_type(np.einsum("i,i->i", AR_LIKE_U, AR_LIKE_U, dtype=bool, casting="unsafe", out=OUT_f), npt.NDArray[np.float64])
# 使用 assert_type 函数验证 np.einsum 函数的返回类型，并指定输入 dtype 类型
assert_type(np.einsum("i,i->i", AR_LIKE_f, AR_LIKE_f, dtype="c16"), Any)
# 使用 assert_type 函数验证 np.einsum 函数的返回类型，并指定输入 dtype 和 casting 类型
assert_type(np.einsum("i,i->i", AR_LIKE_U, AR_LIKE_U, dtype=bool, casting="unsafe"), Any)

# 使用 assert_type 函数验证 np.einsum_path 函数的返回类型
assert_type(np.einsum_path("i,i->i", AR_LIKE_b, AR_LIKE_b), tuple[list[Any], str])
assert_type(np.einsum_path("i,i->i", AR_LIKE_u, AR_LIKE_u), tuple[list[Any], str])
assert_type(np.einsum_path("i,i->i", AR_LIKE_i, AR_LIKE_i), tuple[list[Any], str])
assert_type(np.einsum_path("i,i->i", AR_LIKE_f, AR_LIKE_f), tuple[list[Any], str])
assert_type(np.einsum_path("i,i->i", AR_LIKE_c, AR_LIKE_c), tuple[list[Any], str])
assert_type(np.einsum_path("i,i->i", AR_LIKE_b, AR_LIKE_i), tuple[list[Any], str])
assert_type(np.einsum_path("i,i,i,i->i", AR_LIKE_b, AR_LIKE_u, AR_LIKE_i, AR_LIKE_c), tuple[list[Any], str])

# 使用 assert_type 函数验证 np.einsum 函数的返回类型
assert_type(np.einsum([[1, 1], [1, 1]], AR_LIKE_i, AR_LIKE_i), Any)
# 使用 assert_type 函数验证 np.einsum_path 函数的返回类型
assert_type(np.einsum_path([[1, 1], [1, 1]], AR_LIKE_i, AR_LIKE_i), tuple[list[Any], str])
```