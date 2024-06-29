# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\false_positives.pyi`

```
`
# 导入 numpy 库并简化为 np，提供高效的数组操作功能
import numpy as np
# 导入 numpy.typing 模块中的 npt，用于在类型提示中使用 numpy 数组类型
import numpy.typing as npt

# 定义一个 numpy 数组类型的别名 AR_f8，元素类型为 np.float64
AR_f8: npt.NDArray[np.float64]

# NOTE: Mypy bug presumable due to the special-casing of heterogeneous tuples;
# xref numpy/numpy#20901
#
# NOTE: 由于 Mypy 的一个 bug 可能是由于对异构元组的特殊处理；
# xref numpy/numpy#20901
#
# 预期的输出应该与使用列表而不是元组时没有区别
# np.concatenate 用于连接数组，第一个参数是一个包含单个整数 1 的列表，第二个参数是 AR_f8 数组
np.concatenate(([1], AR_f8))  # E: Argument 1 to "concatenate" has incompatible type
```