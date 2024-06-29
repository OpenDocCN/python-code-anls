# `.\numpy\numpy\f2py\tests\test_mixed.py`

```
import os
import textwrap
import pytest

from numpy.testing import IS_PYPY
from . import util  # 导入本地的util模块

class TestMixed(util.F2PyTest):  # 定义一个测试类TestMixed，继承自util.F2PyTest
    sources = [
        util.getpath("tests", "src", "mixed", "foo.f"),  # 获取特定路径下的文件路径
        util.getpath("tests", "src", "mixed", "foo_fixed.f90"),  # 获取特定路径下的文件路径
        util.getpath("tests", "src", "mixed", "foo_free.f90"),  # 获取特定路径下的文件路径
    ]

    @pytest.mark.slow  # 标记该测试方法为slow，运行时可能较慢
    def test_all(self):
        assert self.module.bar11() == 11  # 断言调用self.module.bar11()返回值为11
        assert self.module.foo_fixed.bar12() == 12  # 断言调用self.module.foo_fixed.bar12()返回值为12
        assert self.module.foo_free.bar13() == 13  # 断言调用self.module.foo_free.bar13()返回值为13

    @pytest.mark.xfail(IS_PYPY, reason="PyPy cannot modify tp_doc after PyType_Ready")
    # 标记该测试方法为xfail，在PyPy环境下预期失败，并附带原因说明
    def test_docstring(self):
        expected = textwrap.dedent("""\
        a = bar11()

        Wrapper for ``bar11``.

        Returns
        -------
        a : int
        """)
        assert self.module.bar11.__doc__ == expected  # 断言self.module.bar11的文档字符串与expected相等
```