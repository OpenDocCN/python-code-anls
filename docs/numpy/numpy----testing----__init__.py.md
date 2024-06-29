# `.\numpy\numpy\testing\__init__.py`

```
"""
Common test support for all numpy test scripts.

This single module should provide all the common functionality for numpy tests
in a single location, so that test scripts can just import it and work right
away.
"""

# 从 unittest 模块中导入 TestCase 类，用于编写单元测试
from unittest import TestCase

# 导入当前包中的 _private 子模块
from . import _private

# 从 _private.utils 中导入所有公开的符号
from ._private.utils import *

# 从 _private.utils 中导入 _assert_valid_refcount 和 _gen_alignment_data 符号
from ._private.utils import (_assert_valid_refcount, _gen_alignment_data)

# 从 _private 子模块中导入 extbuild 符号
from ._private import extbuild

# 从当前包中导入 overrides 模块
from . import overrides

# 将 _private.utils 中所有公开的符号添加到 __all__ 列表中，并增加 'TestCase' 和 'overrides' 符号
__all__ = (
    _private.utils.__all__ + ['TestCase', 'overrides']
)

# 从 numpy._pytesttester 模块中导入 PytestTester 类，并使用当前模块名字创建 test 对象
from numpy._pytesttester import PytestTester
test = PytestTester(__name__)

# 删除导入的 PytestTester 类，以避免在当前模块中保留未使用的符号
del PytestTester
```