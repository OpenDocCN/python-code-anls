# `.\numpy\numpy\f2py\tests\test_return_complex.py`

```
import pytest  # 导入 pytest 测试框架

from numpy import array  # 从 numpy 中导入 array 类
from . import util  # 导入当前目录下的 util 模块

@pytest.mark.slow  # 标记该测试类为慢速测试
class TestReturnComplex(util.F2PyTest):  # 定义测试类 TestReturnComplex，继承自 util.F2PyTest

    def check_function(self, t, tname):  # 定义方法 check_function，用于测试函数 t
        if tname in ["t0", "t8", "s0", "s8"]:  # 如果 tname 在列表中
            err = 1e-5  # 设置误差容限为 1e-5
        else:  # 否则
            err = 0.0  # 设置误差容限为 0.0
        assert abs(t(234j) - 234.0j) <= err  # 断言 t(234j) 的返回值与期望值 234.0j 的差的绝对值小于等于 err
        assert abs(t(234.6) - 234.6) <= err  # 断言 t(234.6) 的返回值与期望值 234.6 的差的绝对值小于等于 err
        assert abs(t(234) - 234.0) <= err  # 断言 t(234) 的返回值与期望值 234.0 的差的绝对值小于等于 err
        assert abs(t(234.6 + 3j) - (234.6 + 3j)) <= err  # 断言 t(234.6 + 3j) 的返回值与期望值 (234.6 + 3j) 的差的绝对值小于等于 err
        # assert abs(t('234')-234.)<=err  # 注释掉的断言
        # assert abs(t('234.6')-234.6)<=err  # 注释掉的断言
        assert abs(t(-234) + 234.0) <= err  # 断言 t(-234) 的返回值与期望值 234.0 的和的绝对值小于等于 err
        assert abs(t([234]) - 234.0) <= err  # 断言 t([234]) 的返回值与期望值 234.0 的差的绝对值小于等于 err
        assert abs(t((234, )) - 234.0) <= err  # 断言 t((234,)) 的返回值与期望值 234.0 的差的绝对值小于等于 err
        assert abs(t(array(234)) - 234.0) <= err  # 断言 t(array(234)) 的返回值与期望值 234.0 的差的绝对值小于等于 err
        assert abs(t(array(23 + 4j, "F")) - (23 + 4j)) <= err  # 断言 t(array(23+4j, 'F')) 的返回值与期望值 (23+4j) 的差的绝对值小于等于 err
        assert abs(t(array([234])) - 234.0) <= err  # 断言 t(array([234])) 的返回值与期望值 234.0 的差的绝对值小于等于 err
        assert abs(t(array([[234]])) - 234.0) <= err  # 断言 t(array([[234]])) 的返回值与期望值 234.0 的差的绝对值小于等于 err
        assert abs(t(array([234]).astype("b")) + 22.0) <= err  # 断言 t(array([234], 'b')) 的返回值与期望值 22.0 的和的绝对值小于等于 err
        assert abs(t(array([234], "h")) - 234.0) <= err  # 断言 t(array([234], 'h')) 的返回值与期望值 234.0 的差的绝对值小于等于 err
        assert abs(t(array([234], "i")) - 234.0) <= err  # 断言 t(array([234], 'i')) 的返回值与期望值 234.0 的差的绝对值小于等于 err
        assert abs(t(array([234], "l")) - 234.0) <= err  # 断言 t(array([234], 'l')) 的返回值与期望值 234.0 的差的绝对值小于等于 err
        assert abs(t(array([234], "q")) - 234.0) <= err  # 断言 t(array([234], 'q')) 的返回值与期望值 234.0 的差的绝对值小于等于 err
        assert abs(t(array([234], "f")) - 234.0) <= err  # 断言 t(array([234], 'f')) 的返回值与期望值 234.0 的差的绝对值小于等于 err
        assert abs(t(array([234], "d")) - 234.0) <= err  # 断言 t(array([234], 'd')) 的返回值与期望值 234.0 的差的绝对值小于等于 err
        assert abs(t(array([234 + 3j], "F")) - (234 + 3j)) <= err  # 断言 t(array([234+3j], 'F')) 的返回值与期望值 (234+3j) 的差的绝对值小于等于 err
        assert abs(t(array([234], "D")) - 234.0) <= err  # 断言 t(array([234], 'D')) 的返回值与期望值 234.0 的差的绝对值小于等于 err

        # pytest.raises(TypeError, t, array([234], 'S1'))  # 注释掉的断言，测试 t(array([234], 'S1')) 是否抛出 TypeError 异常
        pytest.raises(TypeError, t, "abc")  # 断言 t("abc") 抛出 TypeError 异常

        pytest.raises(IndexError, t, [])  # 断言 t([]) 抛出 IndexError 异常
        pytest.raises(IndexError, t, ())  # 断言 t(()) 抛出 IndexError 异常

        pytest.raises(TypeError, t, t)  # 断言 t(t) 抛出 TypeError 异常
        pytest.raises(TypeError, t, {})  # 断言 t({}) 抛出 TypeError 异常

        try:
            r = t(10**400)  # 尝试计算 t(10**400)
            assert repr(r) in ["(inf+0j)", "(Infinity+0j)"]  # 断言 r 的字符串表示在列表中
        except OverflowError:  # 如果出现 OverflowError 异常
            pass  # 忽略异常

class TestFReturnComplex(TestReturnComplex):  # 定义测试类 TestFReturnComplex，继承自 TestReturnComplex
    sources = [  # 定义源文件列表
        util.getpath("tests", "src", "return_complex", "foo77.f"),  # 获取 foo77.f 的路径
        util.getpath("tests", "src", "return_complex", "foo90.f90"),  # 获取 foo90.f90 的路径
    ]

    @pytest.mark.parametrize("name", "t0,t8,t16,td,s0,s8,s16,sd".split(","))  # 参数化测试方法名
    def test_all_f77(self, name):  # 定义测试方法 test_all_f77
        self.check_function(getattr(self.module, name), name)  # 调用 check_function 测试函数名为 name 的函数

    @pytest.mark.parametrize("name", "t0,t8,t16,td,s0,s8,s16,sd".split(","))  # 参数化测试方法名
    def test_all_f90(self, name):  # 定义测试方法 test_all_f90
        self.check_function(getattr(self.module.f90_return_complex, name),  # 调用 check_function 测试 module.f90_return_complex 下的 name 函数
                            name)  # 传入函数名 name
```