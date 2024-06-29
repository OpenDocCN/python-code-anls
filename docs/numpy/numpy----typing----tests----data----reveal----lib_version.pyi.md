# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\lib_version.pyi`

```py
# 导入 sys 模块，用于获取 Python 解释器的相关信息
import sys

# 从 numpy.lib 模块中导入 NumpyVersion 类
from numpy.lib import NumpyVersion

# 如果 Python 版本大于等于 3.11，则从 typing 模块中导入 assert_type 函数
if sys.version_info >= (3, 11):
    from typing import assert_type
# 否则，从 typing_extensions 模块中导入 assert_type 函数
else:
    from typing_extensions import assert_type

# 创建 NumpyVersion 对象，指定版本号为 "1.8.0"
version = NumpyVersion("1.8.0")

# 断言 version.vstring 属性的类型为 str
assert_type(version.vstring, str)
# 断言 version.version 属性的类型为 str
assert_type(version.version, str)
# 断言 version.major 属性的类型为 int
assert_type(version.major, int)
# 断言 version.minor 属性的类型为 int
assert_type(version.minor, int)
# 断言 version.bugfix 属性的类型为 int
assert_type(version.bugfix, int)
# 断言 version.pre_release 属性的类型为 str
assert_type(version.pre_release, str)
# 断言 version.is_devversion 属性的类型为 bool
assert_type(version.is_devversion, bool)

# 断言 version 与自身相等的结果的类型为 bool
assert_type(version == version, bool)
# 断言 version 与自身不相等的结果的类型为 bool
assert_type(version != version, bool)
# 断言 version 小于 "1.8.0" 的结果的类型为 bool
assert_type(version < "1.8.0", bool)
# 断言 version 小于等于自身的结果的类型为 bool
assert_type(version <= version, bool)
# 断言 version 大于自身的结果的类型为 bool
assert_type(version > version, bool)
# 断言 version 大于等于 "1.8.0" 的结果的类型为 bool
assert_type(version >= "1.8.0", bool)
```