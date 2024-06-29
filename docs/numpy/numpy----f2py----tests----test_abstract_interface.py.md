# `.\numpy\numpy\f2py\tests\test_abstract_interface.py`

```
# 从 pathlib 模块导入 Path 类，用于处理文件路径
from pathlib import Path
# 导入 pytest 模块，用于编写和运行测试用例
import pytest
# 导入 textwrap 模块，用于处理和操作文本内容的格式
import textwrap
# 从当前包中导入 util 模块，用于测试中的辅助功能
from . import util
# 从 numpy.f2py 模块导入 crackfortran 函数，用于处理 Fortran 代码
from numpy.f2py import crackfortran
# 从 numpy.testing 模块导入 IS_WASM 常量，用于条件判断
from numpy.testing import IS_WASM

# 使用 pytest 的装饰器标记该测试类，当 IS_WASM 为 True 时跳过测试，给出理由
@pytest.mark.skipif(IS_WASM, reason="Cannot start subprocess")
# 标记该测试类为慢速测试
@pytest.mark.slow
class TestAbstractInterface(util.F2PyTest):
    # 定义测试用例的源文件路径列表
    sources = [util.getpath("tests", "src", "abstract_interface", "foo.f90")]

    # 定义需要跳过的测试方法列表
    skip = ["add1", "add2"]

    # 测试抽象接口的方法
    def test_abstract_interface(self):
        # 断言调用 self.module.ops_module.foo(3, 5) 返回结果为 (8, 13)
        assert self.module.ops_module.foo(3, 5) == (8, 13)

    # 测试解析抽象接口的方法
    def test_parse_abstract_interface(self):
        # 测试用例名称为 gh18403
        fpath = util.getpath("tests", "src", "abstract_interface", "gh18403_mod.f90")
        # 调用 crackfortran.crackfortran 函数处理指定路径的 Fortran 文件，返回模块对象列表
        mod = crackfortran.crackfortran([str(fpath)])
        # 断言 mod 列表长度为 1
        assert len(mod) == 1
        # 断言 mod[0]["body"] 列表的长度为 1
        assert len(mod[0]["body"]) == 1
        # 断言 mod[0]["body"][0]["block"] 的值为 "abstract interface"
        assert mod[0]["body"][0]["block"] == "abstract interface"
```