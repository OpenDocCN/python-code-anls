# `.\numpy\numpy\f2py\tests\test_modules.py`

```py
# 导入 pytest 模块，用于编写和运行测试
import pytest
# 导入 textwrap 模块，用于处理文本的缩进和格式化
import textwrap

# 从当前包中导入 util 模块
from . import util
# 导入 numpy.testing 模块中的 IS_PYPY 常量
from numpy.testing import IS_PYPY

# 使用 pytest.mark.slow 标记的测试类，继承自 util.F2PyTest 类
@pytest.mark.slow
class TestModuleDocString(util.F2PyTest):
    # 源文件列表，包含一个 Fortran 源文件路径
    sources = [util.getpath("tests", "src", "modules", "module_data_docstring.f90")]

    # 标记为 xfail，当在 PyPy 环境下运行时，测试预期会失败
    @pytest.mark.xfail(IS_PYPY, reason="PyPy cannot modify tp_doc after PyType_Ready")
    def test_module_docstring(self):
        # 断言模块的文档字符串等于指定的文本（经过 textwrap.dedent 处理）
        assert self.module.mod.__doc__ == textwrap.dedent(
            """\
                     i : 'i'-scalar
                     x : 'i'-array(4)
                     a : 'f'-array(2,3)
                     b : 'f'-array(-1,-1), not allocated\x00
                     foo()\n
                     Wrapper for ``foo``.\n\n"""
        )

# 使用 pytest.mark.slow 标记的测试类，继承自 util.F2PyTest 类
@pytest.mark.slow
class TestModuleAndSubroutine(util.F2PyTest):
    # 模块名称为 "example"
    module_name = "example"
    # 源文件列表，包含两个 Fortran 源文件路径
    sources = [
        util.getpath("tests", "src", "modules", "gh25337", "data.f90"),
        util.getpath("tests", "src", "modules", "gh25337", "use_data.f90"),
    ]

    def test_gh25337(self):
        # 设置数据模块的偏移量为 3
        self.module.data.set_shift(3)
        # 断言模块中包含 "data" 对象
        assert "data" in dir(self.module)

# 使用 pytest.mark.slow 标记的测试类，继承自 util.F2PyTest 类
@pytest.mark.slow
class TestUsedModule(util.F2PyTest):
    # 模块名称为 "fmath"
    module_name = "fmath"
    # 源文件列表，包含一个 Fortran 源文件路径
    sources = [
        util.getpath("tests", "src", "modules", "use_modules.f90"),
    ]

    def test_gh25867(self):
        # 获取已编译模块的列表（排除双下划线开头的特殊成员）
        compiled_mods = [x for x in dir(self.module) if "__" not in x]
        # 断言模块中包含 "useops" 模块
        assert "useops" in compiled_mods
        # 调用并断言模块中的 useops.sum_and_double 方法返回值为 20
        assert self.module.useops.sum_and_double(3, 7) == 20
        # 断言模块中包含 "mathops" 模块
        assert "mathops" in compiled_mods
        # 调用并断言模块中的 mathops.add 方法返回值为 10
        assert self.module.mathops.add(3, 7) == 10
```