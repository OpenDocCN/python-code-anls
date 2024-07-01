# `.\numpy\numpy\typing\tests\data\reveal\index_tricks.pyi`

```py
import sys
from typing import Any, Literal  # 导入必要的模块和类型提示

import numpy as np  # 导入 NumPy 库
import numpy.typing as npt  # 导入 NumPy 的类型提示

if sys.version_info >= (3, 11):
    from typing import assert_type  # 根据 Python 版本导入不同版本的 assert_type
else:
    from typing_extensions import assert_type

AR_LIKE_b: list[bool]  # 定义 AR_LIKE_b 变量类型为布尔类型的列表
AR_LIKE_i: list[int]  # 定义 AR_LIKE_i 变量类型为整数类型的列表
AR_LIKE_f: list[float]  # 定义 AR_LIKE_f 变量类型为浮点数类型的列表
AR_LIKE_U: list[str]  # 定义 AR_LIKE_U 变量类型为字符串类型的列表

AR_i8: npt.NDArray[np.int64]  # 定义 AR_i8 变量类型为 NumPy 中的 np.int64 数组类型

assert_type(np.ndenumerate(AR_i8), np.ndenumerate[np.int64])  # 断言 AR_i8 的 ndenumerate 类型
assert_type(np.ndenumerate(AR_LIKE_f), np.ndenumerate[np.float64])  # 断言 AR_LIKE_f 的 ndenumerate 类型
assert_type(np.ndenumerate(AR_LIKE_U), np.ndenumerate[np.str_])  # 断言 AR_LIKE_U 的 ndenumerate 类型

assert_type(np.ndenumerate(AR_i8).iter, np.flatiter[npt.NDArray[np.int64]])  # 断言 AR_i8 的 ndenumerate 的 iter 属性类型
assert_type(np.ndenumerate(AR_LIKE_f).iter, np.flatiter[npt.NDArray[np.float64]])  # 断言 AR_LIKE_f 的 ndenumerate 的 iter 属性类型
assert_type(np.ndenumerate(AR_LIKE_U).iter, np.flatiter[npt.NDArray[np.str_]])  # 断言 AR_LIKE_U 的 ndenumerate 的 iter 属性类型

assert_type(next(np.ndenumerate(AR_i8)), tuple[tuple[int, ...], np.int64])  # 断言调用 np.ndenumerate(AR_i8) 后返回的类型
assert_type(next(np.ndenumerate(AR_LIKE_f)), tuple[tuple[int, ...], np.float64])  # 断言调用 np.ndenumerate(AR_LIKE_f) 后返回的类型
assert_type(next(np.ndenumerate(AR_LIKE_U)), tuple[tuple[int, ...], np.str_])  # 断言调用 np.ndenumerate(AR_LIKE_U) 后返回的类型

assert_type(iter(np.ndenumerate(AR_i8)), np.ndenumerate[np.int64])  # 断言调用 iter(np.ndenumerate(AR_i8)) 后返回的类型
assert_type(iter(np.ndenumerate(AR_LIKE_f)), np.ndenumerate[np.float64])  # 断言调用 iter(np.ndenumerate(AR_LIKE_f)) 后返回的类型
assert_type(iter(np.ndenumerate(AR_LIKE_U)), np.ndenumerate[np.str_])  # 断言调用 iter(np.ndenumerate(AR_LIKE_U)) 后返回的类型

assert_type(np.ndindex(1, 2, 3), np.ndindex)  # 断言 np.ndindex(1, 2, 3) 的类型
assert_type(np.ndindex((1, 2, 3)), np.ndindex)  # 断言 np.ndindex((1, 2, 3)) 的类型
assert_type(iter(np.ndindex(1, 2, 3)), np.ndindex)  # 断言调用 iter(np.ndindex(1, 2, 3)) 后返回的类型
assert_type(next(np.ndindex(1, 2, 3)), tuple[int, ...])  # 断言调用 next(np.ndindex(1, 2, 3)) 后返回的类型

assert_type(np.unravel_index([22, 41, 37], (7, 6)), tuple[npt.NDArray[np.intp], ...])  # 断言调用 np.unravel_index([22, 41, 37], (7, 6)) 的类型
assert_type(np.unravel_index([31, 41, 13], (7, 6), order="F"), tuple[npt.NDArray[np.intp], ...])  # 断言调用 np.unravel_index([31, 41, 13], (7, 6), order="F") 的类型
assert_type(np.unravel_index(1621, (6, 7, 8, 9)), tuple[np.intp, ...])  # 断言调用 np.unravel_index(1621, (6, 7, 8, 9)) 的类型

assert_type(np.ravel_multi_index([[1]], (7, 6)), npt.NDArray[np.intp])  # 断言调用 np.ravel_multi_index([[1]], (7, 6)) 的类型
assert_type(np.ravel_multi_index(AR_LIKE_i, (7, 6)), np.intp)  # 断言调用 np.ravel_multi_index(AR_LIKE_i, (7, 6)) 的类型
assert_type(np.ravel_multi_index(AR_LIKE_i, (7, 6), order="F"), np.intp)  # 断言调用 np.ravel_multi_index(AR_LIKE_i, (7, 6), order="F") 的类型
assert_type(np.ravel_multi_index(AR_LIKE_i, (4, 6), mode="clip"), np.intp)  # 断言调用 np.ravel_multi_index(AR_LIKE_i, (4, 6), mode="clip") 的类型
assert_type(np.ravel_multi_index(AR_LIKE_i, (4, 4), mode=("clip", "wrap")), np.intp)  # 断言调用 np.ravel_multi_index(AR_LIKE_i, (4, 4), mode=("clip", "wrap")) 的类型
assert_type(np.ravel_multi_index((3, 1, 4, 1), (6, 7, 8, 9)), np.intp)  # 断言调用 np.ravel_multi_index((3, 1, 4, 1), (6, 7, 8, 9)) 的类型

assert_type(np.mgrid[1:1:2], npt.NDArray[Any])  # 断言调用 np.mgrid[1:1:2] 的类型
assert_type(np.mgrid[1:1:2, None:10], npt.NDArray[Any])  # 断言调用 np.mgrid[1:1:2, None:10] 的类型

assert_type(np.ogrid[1:1:2], tuple[npt.NDArray[Any], ...])  # 断言调用 np.ogrid[1:1:2] 的类型
assert_type(np.ogrid[1:1:2, None:10], tuple[npt.NDArray[Any], ...])  # 断言调用 np.ogrid[1:1:2, None:10] 的类型

assert_type(np.index_exp[0:1], tuple[slice])  # 断言 np.index_exp[0:1] 的类型
assert_type(np.index_exp[0:1, None:3], tuple[slice, slice])  # 断言 np.index_exp[0:1, None:3] 的类型
assert_type(np.index_exp[0, 0:1, ..., [0, 1, 3]], tuple[Literal[0], slice, ellipsis, list[int]])  # 断言 np.index_exp[0, 0:1, ..., [0, 1, 3]] 的类型

assert_type(np.s_[0:1], slice)  # 断言 np.s_[0:1] 的类型
assert_type(np.s_[0:1, None:3], tuple[slice, slice])  # 断言 np.s_[0:1, None:3] 的类型
assert_type(np.s_[0, 0:1, ..., [0, 1, 3]], tuple[Literal[0], slice, ellipsis, list[int]])  # 断言 np.s_[0, 0:1, ..., [0, 1, 3]] 的类型

assert_type(np.ix_(AR_LIKE_b), tuple[npt.NDArray[np.bool], ...])  # 断言调用 np.ix_(AR_LIKE_b) 的类型
assert_type(np.ix_(AR_LIKE_i, AR_LIKE_f), tuple[npt.NDArray[np.float64], ...])  # 断言调用 np.ix_(AR_LIKE_i, AR_LIKE_f) 的类型
assert_type(np.ix_(AR_i8), tuple[npt.NDArray[np.int64], ...])  # 断言调用 np.ix_(AR_i8) 的类型

assert_type(np.fill_diagonal(AR_i8, 5), None)  # 断言调用 np.fill_diagonal(AR_i8,
# 断言语句：验证 np.diag_indices(4) 的返回类型为 tuple[npt.NDArray[np.int_], ...]
assert_type(np.diag_indices(4), tuple[npt.NDArray[np.int_], ...])

# 断言语句：验证 np.diag_indices(2, 3) 的返回类型为 tuple[npt.NDArray[np.int_], ...]
assert_type(np.diag_indices(2, 3), tuple[npt.NDArray[np.int_], ...])

# 断言语句：验证 np.diag_indices_from(AR_i8) 的返回类型为 tuple[npt.NDArray[np.int_], ...]
assert_type(np.diag_indices_from(AR_i8), tuple[npt.NDArray[np.int_], ...])
```