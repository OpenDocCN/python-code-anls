# `.\numpy\numpy\f2py\tests\test_kind.py`

```py
# 导入必要的模块和库
import sys  # 导入系统模块
import os  # 导入操作系统模块
import pytest  # 导入 pytest 测试框架
import platform  # 导入平台信息模块

# 导入 util 模块中的特定函数
from numpy.f2py.crackfortran import (
    _selected_int_kind_func as selected_int_kind,  # 导入整数类型处理函数
    _selected_real_kind_func as selected_real_kind,  # 导入实数类型处理函数
)
from . import util  # 导入当前包中的 util 模块

# 定义一个测试类 TestKind，继承自 util.F2PyTest 类
class TestKind(util.F2PyTest):
    sources = [util.getpath("tests", "src", "kind", "foo.f90")]  # 指定测试源文件路径

    # 装饰器标记，如果系统最大整数值小于 2^31+1，则跳过该测试
    @pytest.mark.skipif(sys.maxsize < 2 ** 31 + 1, reason="Fails for 32 bit machines")
    def test_int(self):
        """Test `int` kind_func for integers up to 10**40."""
        selectedintkind = self.module.selectedintkind  # 获取模块中的整数处理函数

        # 遍历整数范围 [0, 39]，验证整数处理函数的正确性
        for i in range(40):
            assert selectedintkind(i) == selected_int_kind(
                i
            ), f"selectedintkind({i}): expected {selected_int_kind(i)!r} but got {selectedintkind(i)!r}"

    # 测试实数类型处理函数
    def test_real(self):
        """
        Test (processor-dependent) `real` kind_func for real numbers
        of up to 31 digits precision (extended/quadruple).
        """
        selectedrealkind = self.module.selectedrealkind  # 获取模块中的实数处理函数

        # 遍历实数范围 [0, 31]，验证实数处理函数的正确性
        for i in range(32):
            assert selectedrealkind(i) == selected_real_kind(
                i
            ), f"selectedrealkind({i}): expected {selected_real_kind(i)!r} but got {selectedrealkind(i)!r}"

    # 装饰器标记，如果当前机器是 PowerPC，则标记测试为预期失败
    @pytest.mark.xfail(platform.machine().lower().startswith("ppc"), reason="Some PowerPC may not support full IEEE 754 precision")
    def test_quad_precision(self):
        """
        Test kind_func for quadruple precision [`real(16)`] of 32+ digits .
        """
        selectedrealkind = self.module.selectedrealkind  # 获取模块中的实数处理函数

        # 遍历实数范围 [32, 39]，验证实数处理函数在四重精度下的正确性
        for i in range(32, 40):
            assert selectedrealkind(i) == selected_real_kind(
                i
            ), f"selectedrealkind({i}): expected {selected_real_kind(i)!r} but got {selectedrealkind(i)!r}"
```