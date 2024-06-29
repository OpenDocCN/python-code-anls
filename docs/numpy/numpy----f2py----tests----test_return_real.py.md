# `.\numpy\numpy\f2py\tests\test_return_real.py`

```
# 导入必要的模块
import platform  # 导入平台模块，用于获取操作系统信息
import pytest  # 导入 pytest 测试框架
import numpy as np  # 导入 NumPy 库，并命名为 np

from numpy import array  # 从 NumPy 中导入 array 函数
from . import util  # 从当前包中导入 util 模块

@pytest.mark.slow
class TestReturnReal(util.F2PyTest):
    # 定义 TestReturnReal 类，继承自 util.F2PyTest 类，标记为 pytest 的慢速测试用例

    def check_function(self, t, tname):
        # 定义检查函数 check_function，接受 t 函数和 tname 参数

        if tname in ["t0", "t4", "s0", "s4"]:
            err = 1e-5  # 如果 tname 在指定列表中，设置误差阈值为 1e-5
        else:
            err = 0.0  # 否则，误差阈值为 0.0

        # 一系列断言，验证 t 函数对于不同类型输入的返回值是否在误差范围内
        assert abs(t(234) - 234.0) <= err
        assert abs(t(234.6) - 234.6) <= err
        assert abs(t("234") - 234) <= err
        assert abs(t("234.6") - 234.6) <= err
        assert abs(t(-234) + 234) <= err
        assert abs(t([234]) - 234) <= err
        assert abs(t((234, )) - 234.0) <= err
        assert abs(t(array(234)) - 234.0) <= err
        assert abs(t(array(234).astype("b")) + 22) <= err
        assert abs(t(array(234, "h")) - 234.0) <= err
        assert abs(t(array(234, "i")) - 234.0) <= err
        assert abs(t(array(234, "l")) - 234.0) <= err
        assert abs(t(array(234, "B")) - 234.0) <= err
        assert abs(t(array(234, "f")) - 234.0) <= err
        assert abs(t(array(234, "d")) - 234.0) <= err

        if tname in ["t0", "t4", "s0", "s4"]:
            assert t(1e200) == t(1e300)  # 对于特定 tname，验证 t 函数在处理大数时的行为（应为 inf）

        # 使用 pytest.raises 断言捕获异常，验证 t 函数在特定输入情况下会触发 ValueError 或 IndexError 异常
        pytest.raises(ValueError, t, "abc")
        pytest.raises(IndexError, t, [])
        pytest.raises(IndexError, t, ())

        # 使用 pytest.raises 断言捕获异常，验证 t 函数在其他异常情况下会触发 Exception 异常
        pytest.raises(Exception, t, t)
        pytest.raises(Exception, t, {})

        try:
            r = t(10**400)
            assert repr(r) in ["inf", "Infinity"]  # 验证 t 函数返回值的字符串表示是否为 "inf" 或 "Infinity"
        except OverflowError:
            pass  # 捕获 OverflowError 异常，不进行任何操作

@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Prone to error when run with numpy/f2py/tests on mac os, "
    "but not when run in isolation",
)
@pytest.mark.skipif(
    np.dtype(np.intp).itemsize < 8,
    reason="32-bit builds are buggy"
)
class TestCReturnReal(TestReturnReal):
    # 定义 TestCReturnReal 类，继承自 TestReturnReal 类，标记为 pytest 的跳过条件测试用例

    suffix = ".pyf"
    module_name = "c_ext_return_real"
    code = """
python module c_ext_return_real
usercode \'\'\'
float t4(float value) { return value; }
void s4(float *t4, float value) { *t4 = value; }
double t8(double value) { return value; }
void s8(double *t8, double value) { *t8 = value; }
\'\'\'
interface
  function t4(value)
    real*4 intent(c) :: t4,value
  end
  function t8(value)
    real*8 intent(c) :: t8,value
  end
  subroutine s4(t4,value)
    intent(c) s4
    real*4 intent(out) :: t4
    real*4 intent(c) :: value
  end
  subroutine s8(t8,value)
    intent(c) s8
    real*8 intent(out) :: t8
    real*8 intent(c) :: value
  end
end interface
end python module c_ext_return_real
    """

    @pytest.mark.parametrize("name", "t4,t8,s4,s8".split(","))
    def test_all(self, name):
        # 使用 pytest.mark.parametrize 注入参数化测试，依次测试 t4, t8, s4, s8 函数
        self.check_function(getattr(self.module, name), name)  # 调用 check_function 验证各函数行为

class TestFReturnReal(TestReturnReal):
    # 定义 TestFReturnReal 类，继承自 TestReturnReal 类

    sources = [
        util.getpath("tests", "src", "return_real", "foo77.f"),
        util.getpath("tests", "src", "return_real", "foo90.f90"),
    ]

    @pytest.mark.parametrize("name", "t0,t4,t8,td,s0,s4,s8,sd".split(","))
    def test_all(self, name):
        # 使用 pytest.mark.parametrize 注入参数化测试，依次测试 t0, t4, t8, td, s0, s4, s8, sd 函数
    # 定义一个测试方法，用于测试给定名称的 Fortran 77 函数是否符合预期
    def test_all_f77(self, name):
        # 调用 self.module 中的特定名称的函数，并使用 self.check_function 进行检查
        self.check_function(getattr(self.module, name), name)
    
    # 使用 pytest 的参数化装饰器，指定多个参数来执行测试，参数名称为 "t0", "t4", "t8", "td", "s0", "s4", "s8", "sd"
    @pytest.mark.parametrize("name", "t0,t4,t8,td,s0,s4,s8,sd".split(","))
    def test_all_f90(self, name):
        # 调用 self.module.f90_return_real 中特定名称的函数，并使用 self.check_function 进行检查
        self.check_function(getattr(self.module.f90_return_real, name), name)
```