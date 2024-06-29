# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\constants.pyi`

```
# 导入 sys 模块，用于访问系统相关的功能
import sys

# 导入 numpy 库，并将其命名为 np，以便使用其提供的数学函数和数据结构
import numpy as np

# 根据 Python 解释器版本号，选择合适的 assert_type 函数进行导入
if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

# 使用 assert_type 函数检查 np.e 是否为 float 类型
assert_type(np.e, float)
# 使用 assert_type 函数检查 np.euler_gamma 是否为 float 类型
assert_type(np.euler_gamma, float)
# 使用 assert_type 函数检查 np.inf 是否为 float 类型
assert_type(np.inf, float)
# 使用 assert_type 函数检查 np.nan 是否为 float 类型
assert_type(np.nan, float)
# 使用 assert_type 函数检查 np.pi 是否为 float 类型
assert_type(np.pi, float)

# 使用 assert_type 函数检查 np.little_endian 是否为 bool 类型
assert_type(np.little_endian, bool)
# 使用 assert_type 函数检查 np.True_ 是否为 np.bool 类型
assert_type(np.True_, np.bool)
# 使用 assert_type 函数检查 np.False_ 是否为 np.bool 类型
assert_type(np.False_, np.bool)
```