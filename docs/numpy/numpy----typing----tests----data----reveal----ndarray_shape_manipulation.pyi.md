# `.\numpy\numpy\typing\tests\data\reveal\ndarray_shape_manipulation.pyi`

```py
# 导入系统模块 sys
# 从 typing 模块中导入 Any 类型
import sys
from typing import Any

# 导入 numpy 库，并将其命名为 np
import numpy as np
# 导入 numpy.typing 模块，引入 npt 别名，用于类型提示
import numpy.typing as npt

# 如果 Python 版本大于等于 3.11，则从 typing 模块导入 assert_type 函数
if sys.version_info >= (3, 11):
    from typing import assert_type
# 否则，从 typing_extensions 模块导入 assert_type 函数
else:
    from typing_extensions import assert_type

# 声明变量 nd，指定其类型为 npt.NDArray[np.int64]
nd: npt.NDArray[np.int64]

# reshape 操作的类型断言
assert_type(nd.reshape(), npt.NDArray[np.int64])
assert_type(nd.reshape(4), npt.NDArray[np.int64])
assert_type(nd.reshape(2, 2), npt.NDArray[np.int64])
assert_type(nd.reshape((2, 2)), npt.NDArray[np.int64])

# 指定 order="C" 参数的 reshape 操作的类型断言
assert_type(nd.reshape((2, 2), order="C"), npt.NDArray[np.int64])
assert_type(nd.reshape(4, order="C"), npt.NDArray[np.int64])

# resize 操作不返回值，无需类型断言

# transpose 操作的类型断言
assert_type(nd.transpose(), npt.NDArray[np.int64])
assert_type(nd.transpose(1, 0), npt.NDArray[np.int64])
assert_type(nd.transpose((1, 0)), npt.NDArray[np.int64])

# swapaxes 操作的类型断言
assert_type(nd.swapaxes(0, 1), npt.NDArray[np.int64])

# flatten 操作的类型断言
assert_type(nd.flatten(), npt.NDArray[np.int64])
assert_type(nd.flatten("C"), npt.NDArray[np.int64])

# ravel 操作的类型断言
assert_type(nd.ravel(), npt.NDArray[np.int64])
assert_type(nd.ravel("C"), npt.NDArray[np.int64])

# squeeze 操作的类型断言
assert_type(nd.squeeze(), npt.NDArray[np.int64])
assert_type(nd.squeeze(0), npt.NDArray[np.int64])
assert_type(nd.squeeze((0, 2)), npt.NDArray[np.int64])
```