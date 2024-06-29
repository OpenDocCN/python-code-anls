# `.\numpy\numpy\f2py\tests\test_return_integer.py`

```
# 导入 pytest 模块，用于编写和运行测试
import pytest

# 导入 numpy 中的 array 对象
from numpy import array

# 从当前包中导入 util 模块
from . import util

# 为测试类标记为慢速测试
@pytest.mark.slow
class TestReturnInteger(util.F2PyTest):
    # 定义用于检查函数行为的方法
    def check_function(self, t, tname):
        # 断言函数 t 正确处理整数输入，返回相同整数
        assert t(123) == 123
        assert t(123.6) == 123
        assert t("123") == 123
        assert t(-123) == -123
        assert t([123]) == 123
        assert t((123, )) == 123
        assert t(array(123)) == 123
        assert t(array(123, "b")) == 123
        assert t(array(123, "h")) == 123
        assert t(array(123, "i")) == 123
        assert t(array(123, "l")) == 123
        assert t(array(123, "B")) == 123
        assert t(array(123, "f")) == 123
        assert t(array(123, "d")) == 123

        # 测试抛出 ValueError 异常情况
        pytest.raises(ValueError, t, "abc")

        # 测试抛出 IndexError 异常情况
        pytest.raises(IndexError, t, [])
        pytest.raises(IndexError, t, ())

        # 测试抛出任意异常情况
        pytest.raises(Exception, t, t)
        pytest.raises(Exception, t, {})

        # 对于特定的 tname 值，测试是否能够抛出 OverflowError 异常
        if tname in ["t8", "s8"]:
            pytest.raises(OverflowError, t, 100000000000000000000000)
            pytest.raises(OverflowError, t, 10000000011111111111111.23)


# 继承自 TestReturnInteger 类，测试与 Fortran 相关的整数返回函数
class TestFReturnInteger(TestReturnInteger):
    # 指定测试的源文件路径列表
    sources = [
        util.getpath("tests", "src", "return_integer", "foo77.f"),
        util.getpath("tests", "src", "return_integer", "foo90.f90"),
    ]

    # 参数化测试方法，依次对 t0 到 s8 进行测试
    @pytest.mark.parametrize("name",
                             "t0,t1,t2,t4,t8,s0,s1,s2,s4,s8".split(","))
    def test_all_f77(self, name):
        # 调用 check_function 方法测试每个函数名对应的函数
        self.check_function(getattr(self.module, name), name)

    # 参数化测试方法，依次对 t0 到 s8 进行测试（针对 Fortran 90）
    @pytest.mark.parametrize("name",
                             "t0,t1,t2,t4,t8,s0,s1,s2,s4,s8".split(","))
    def test_all_f90(self, name):
        # 调用 check_function 方法测试每个函数名对应的函数（针对 Fortran 90）
        self.check_function(getattr(self.module.f90_return_integer, name),
                            name)
```