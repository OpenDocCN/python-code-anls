# `.\numpy\numpy\f2py\tests\test_block_docstring.py`

```py
# 导入系统模块 sys
import sys
# 导入 pytest 测试框架
import pytest
# 从当前包中导入 util 模块
from . import util

# 导入 numpy 的测试工具 IS_PYPY
from numpy.testing import IS_PYPY

# 使用 pytest 的标记 @pytest.mark.slow 标记测试类为慢速测试
@pytest.mark.slow
# 定义一个测试类 TestBlockDocString，继承自 util.F2PyTest
class TestBlockDocString(util.F2PyTest):
    # 定义测试类的属性 sources，包含一个文件路径列表
    sources = [util.getpath("tests", "src", "block_docstring", "foo.f")]

    # 使用 pytest 的标记 @pytest.mark.skipif 根据条件跳过测试
    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Fails with MinGW64 Gfortran (Issue #9673)")
    # 使用 pytest 的标记 @pytest.mark.xfail 标记测试预期会失败
    @pytest.mark.xfail(IS_PYPY,
                       reason="PyPy cannot modify tp_doc after PyType_Ready")
    # 定义测试方法 test_block_docstring
    def test_block_docstring(self):
        # 期望的文档字符串内容
        expected = "bar : 'i'-array(2,3)\n"
        # 断言模块 self.module.block 的文档字符串与期望值相等
        assert self.module.block.__doc__ == expected
```