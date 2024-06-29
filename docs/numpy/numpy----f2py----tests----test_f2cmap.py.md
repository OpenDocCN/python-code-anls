# `.\numpy\numpy\f2py\tests\test_f2cmap.py`

```py
# 导入名为 util 的当前目录下的模块
from . import util
# 导入 NumPy 库，并将其命名为 np
import numpy as np

# 创建一个名为 TestF2Cmap 的类，继承自 util.F2PyTest 类
class TestF2Cmap(util.F2PyTest):
    # 定义一个名为 sources 的类变量，包含两个路径字符串的列表
    sources = [
        # 获取指定路径下的文件路径字符串，构成一个 Fortran 源文件的路径
        util.getpath("tests", "src", "f2cmap", "isoFortranEnvMap.f90"),
        # 获取指定路径下的文件路径字符串，构成一个 .f2py_f2cmap 文件的路径
        util.getpath("tests", "src", "f2cmap", ".f2py_f2cmap")
    ]

    # 定义一个名为 test_gh15095 的测试方法
    # 测试用例名称为 gh-15095
    def test_gh15095(self):
        # 创建一个包含三个元素的全为 1 的 NumPy 数组
        inp = np.ones(3)
        # 调用 self.module 的 func1 方法，输入为 inp 数组，返回值赋给 out
        out = self.module.func1(inp)
        # 预期输出为整数 3
        exp_out = 3
        # 断言 out 的值等于预期输出 exp_out
        assert out == exp_out
```