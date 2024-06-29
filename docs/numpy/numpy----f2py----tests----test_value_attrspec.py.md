# `.\numpy\numpy\f2py\tests\test_value_attrspec.py`

```
# 导入标准库 os 和 pytest 模块
import os
import pytest

# 从当前包中导入 util 模块
from . import util

# 定义测试类 TestValueAttr，继承自 util.F2PyTest
class TestValueAttr(util.F2PyTest):
    # 指定源文件路径列表，该路径为 util 模块中的 getpath 函数返回的路径
    sources = [util.getpath("tests", "src", "value_attrspec", "gh21665.f90")]

    # 标记为慢速测试（slow），用以区分测试的特性或执行时间
    @pytest.mark.slow
    # 定义测试方法 test_gh21665，用于测试 GitHub 问题编号为 21665 的情况
    def test_gh21665(self):
        # 输入值为 2
        inp = 2
        # 调用被测试模块（self.module）中的 fortfuncs 对象的 square 方法，并传入输入值 inp
        out = self.module.fortfuncs.square(inp)
        # 预期输出值为 4
        exp_out = 4
        # 断言实际输出值与预期输出值相等
        assert out == exp_out
```