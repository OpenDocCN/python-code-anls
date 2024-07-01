# `.\numpy\numpy\typing\tests\data\fail\lib_utils.pyi`

```py
# 导入 numpy 库中的 array_utils 模块，用于处理数组相关的实用工具
import numpy.lib.array_utils as array_utils
# 调用 array_utils 模块中的 byte_bounds 函数，传入参数 1，返回其字节边界
array_utils.byte_bounds(1)  # E: incompatible type
```