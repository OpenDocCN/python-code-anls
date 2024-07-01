# `.\numpy\numpy\typing\tests\data\reveal\false_positives.pyi`

```py
# 导入 sys 模块，用于访问系统相关信息
import sys
# 导入 Any 类型用于类型提示
from typing import Any

# 导入 numpy 库并将其命名为 np
import numpy as np
# 导入 numpy.typing 中的 npt 模块，用于类型提示
import numpy.typing as npt

# 如果 Python 版本大于等于 3.11，则从 typing 模块中导入 assert_type 函数
if sys.version_info >= (3, 11):
    from typing import assert_type
# 否则，从 typing_extensions 模块中导入 assert_type 函数
else:
    from typing_extensions import assert_type

# 定义 AR_Any 变量，类型为 npt.NDArray[Any]
AR_Any: npt.NDArray[Any]

# 由于 Mypy 的一个 bug，它忽略了 `Any` 参数化类型的重载歧义；
# 参考 numpy/numpy#20099 和 python/mypy#11347
#
# 期望的输出应该类似于 `npt.NDArray[Any]`
assert_type(AR_Any + 2, npt.NDArray[np.signedinteger[Any]])
```