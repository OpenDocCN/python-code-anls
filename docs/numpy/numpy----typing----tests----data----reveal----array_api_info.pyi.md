# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\array_api_info.pyi`

```py
# 导入系统相关的模块
import sys
# 导入 List 类型的泛型支持
from typing import List

# 导入 numpy 库，并重命名为 np
import numpy as np

# 根据 Python 版本导入对应的 assert_type 函数
if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

# 获取 numpy 库中的 __array_namespace_info__ 函数返回的对象
array_namespace_info = np.__array_namespace_info__()

# 使用 assert_type 函数确认 array_namespace_info.__module__ 是字符串类型
assert_type(array_namespace_info.__module__, str)
# 使用 assert_type 函数确认 array_namespace_info.capabilities() 返回的类型
assert_type(array_namespace_info.capabilities(), np._array_api_info.Capabilities)
# 使用 assert_type 函数确认 array_namespace_info.default_device() 返回的默认设备是字符串类型
assert_type(array_namespace_info.default_device(), str)
# 使用 assert_type 函数确认 array_namespace_info.default_dtypes() 返回的默认数据类型集合类型
assert_type(array_namespace_info.default_dtypes(), np._array_api_info.DefaultDataTypes)
# 使用 assert_type 函数确认 array_namespace_info.dtypes() 返回的数据类型集合类型
assert_type(array_namespace_info.dtypes(), np._array_api_info.DataTypes)
# 使用 assert_type 函数确认 array_namespace_info.devices() 返回的设备列表类型
assert_type(array_namespace_info.devices(), List[str])
```