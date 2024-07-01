# `.\numpy\numpy\typing\tests\data\reveal\warnings_and_errors.pyi`

```py
# 导入 sys 模块，用于访问系统相关的功能
import sys

# 导入 numpy.exceptions 模块中的各种异常类，方便后续使用
import numpy.exceptions as ex

# 如果 Python 版本大于等于 3.11，则从 typing 模块导入 assert_type 函数
if sys.version_info >= (3, 11):
    from typing import assert_type
# 否则，从 typing_extensions 模块导入 assert_type 函数
else:
    from typing_extensions import assert_type

# 使用 assert_type 函数检查并确保 ex.ModuleDeprecationWarning 是 ex.ModuleDeprecationWarning 类型
assert_type(ex.ModuleDeprecationWarning(), ex.ModuleDeprecationWarning)
# 使用 assert_type 函数检查并确保 ex.VisibleDeprecationWarning 是 ex.VisibleDeprecationWarning 类型
assert_type(ex.VisibleDeprecationWarning(), ex.VisibleDeprecationWarning)
# 使用 assert_type 函数检查并确保 ex.ComplexWarning 是 ex.ComplexWarning 类型
assert_type(ex.ComplexWarning(), ex.ComplexWarning)
# 使用 assert_type 函数检查并确保 ex.RankWarning 是 ex.RankWarning 类型
assert_type(ex.RankWarning(), ex.RankWarning)
# 使用 assert_type 函数检查并确保 ex.TooHardError 是 ex.TooHardError 类型
assert_type(ex.TooHardError(), ex.TooHardError)
# 使用 assert_type 函数检查并确保 ex.AxisError 实例化时传入的参数符合 ex.AxisError 类型
assert_type(ex.AxisError("test"), ex.AxisError)
# 使用 assert_type 函数检查并确保 ex.AxisError 实例化时传入的参数符合 ex.AxisError 类型
assert_type(ex.AxisError(5, 1), ex.AxisError)
```