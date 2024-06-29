# `.\numpy\numpy\f2py\tests\test_return_logical.py`

```py
import pytest  # 导入 pytest 库

from numpy import array  # 从 numpy 库导入 array 函数
from . import util  # 导入当前目录下的 util 模块


class TestReturnLogical(util.F2PyTest):
    def check_function(self, t):
        assert t(True) == 1  # 断言 t(True) 的返回值为 1
        assert t(False) == 0  # 断言 t(False) 的返回值为 0
        assert t(0) == 0  # 断言 t(0) 的返回值为 0
        assert t(None) == 0  # 断言 t(None) 的返回值为 0
        assert t(0.0) == 0  # 断言 t(0.0) 的返回值为 0
        assert t(0j) == 0  # 断言 t(0j) 的返回值为 0
        assert t(1j) == 1  # 断言 t(1j) 的返回值为 1
        assert t(234) == 1  # 断言 t(234) 的返回值为 1
        assert t(234.6) == 1  # 断言 t(234.6) 的返回值为 1
        assert t(234.6 + 3j) == 1  # 断言 t(234.6 + 3j) 的返回值为 1
        assert t("234") == 1  # 断言 t("234") 的返回值为 1
        assert t("aaa") == 1  # 断言 t("aaa") 的返回值为 1
        assert t("") == 0  # 断言 t("") 的返回值为 0
        assert t([]) == 0  # 断言 t([]) 的返回值为 0
        assert t(()) == 0  # 断言 t(()) 的返回值为 0
        assert t({}) == 0  # 断言 t({}) 的返回值为 0
        assert t(t) == 1  # 断言 t(t) 的返回值为 1
        assert t(-234) == 1  # 断言 t(-234) 的返回值为 1
        assert t(10**100) == 1  # 断言 t(10**100) 的返回值为 1
        assert t([234]) == 1  # 断言 t([234]) 的返回值为 1
        assert t((234,)) == 1  # 断言 t((234,)) 的返回值为 1
        assert t(array(234)) == 1  # 断言 t(array(234)) 的返回值为 1
        assert t(array([234])) == 1  # 断言 t(array([234])) 的返回值为 1
        assert t(array([[234]])) == 1  # 断言 t(array([[234]])) 的返回值为 1
        assert t(array([127], "b")) == 1  # 断言 t(array([127], "b")) 的返回值为 1
        assert t(array([234], "h")) == 1  # 断言 t(array([234], "h")) 的返回值为 1
        assert t(array([234], "i")) == 1  # 断言 t(array([234], "i")) 的返回值为 1
        assert t(array([234], "l")) == 1  # 断言 t(array([234], "l")) 的返回值为 1
        assert t(array([234], "f")) == 1  # 断言 t(array([234], "f")) 的返回值为 1
        assert t(array([234], "d")) == 1  # 断言 t(array([234], "d")) 的返回值为 1
        assert t(array([234 + 3j], "F")) == 1  # 断言 t(array([234 + 3j], "F")) 的返回值为 1
        assert t(array([234], "D")) == 1  # 断言 t(array([234], "D")) 的返回值为 1
        assert t(array(0)) == 0  # 断言 t(array(0)) 的返回值为 0
        assert t(array([0])) == 0  # 断言 t(array([0])) 的返回值为 0
        assert t(array([[0]])) == 0  # 断言 t(array([[0]])) 的返回值为 0
        assert t(array([0j])) == 0  # 断言 t(array([0j])) 的返回值为 0
        assert t(array([1])) == 1  # 断言 t(array([1])) 的返回值为 1
        pytest.raises(ValueError, t, array([0, 0]))  # 使用 pytest 断言 t(array([0, 0])) 抛出 ValueError


class TestFReturnLogical(TestReturnLogical):
    sources = [
        util.getpath("tests", "src", "return_logical", "foo77.f"),  # 设置 sources 列表，包含 foo77.f 文件路径
        util.getpath("tests", "src", "return_logical", "foo90.f90"),  # 设置 sources 列表，包含 foo90.f90 文件路径
    ]

    @pytest.mark.slow  # 标记为慢速测试
    @pytest.mark.parametrize("name", "t0,t1,t2,t4,s0,s1,s2,s4".split(","))  # 参数化测试用例名列表
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name))  # 调用 check_function 方法测试指定的函数名

    @pytest.mark.slow  # 标记为慢速测试
    @pytest.mark.parametrize("name",
                             "t0,t1,t2,t4,t8,s0,s1,s2,s4,s8".split(","))  # 参数化测试用例名列表
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_logical, name))  # 调用 check_function 方法测试指定的函数名
```