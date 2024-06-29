# `.\numpy\numpy\_core\tests\test_scalarinherit.py`

```py
""" Test printing of scalar types.

"""
# 引入 pytest 库，用于编写和运行测试
import pytest

# 引入 numpy 库及其子模块
import numpy as np
# 从 numpy.testing 模块中引入测试工具函数
from numpy.testing import assert_, assert_raises


# 定义类 A，作为基类
class A:
    pass

# 定义类 B，继承自 A 和 np.float64
class B(A, np.float64):
    pass

# 定义类 C，继承自 B
class C(B):
    pass

# 定义类 D，继承自 C 和 B
class D(C, B):
    pass

# 定义类 B0，继承自 np.float64 和 A
class B0(np.float64, A):
    pass

# 定义类 C0，继承自 B0
class C0(B0):
    pass

# 定义类 HasNew，重载 __new__ 方法
class HasNew:
    def __new__(cls, *args, **kwargs):
        return cls, args, kwargs

# 定义类 B1，继承自 np.float64 和 HasNew
class B1(np.float64, HasNew):
    pass


# 定义测试类 TestInherit
class TestInherit:
    # 定义测试方法 test_init
    def test_init(self):
        # 创建 B 类的实例 x
        x = B(1.0)
        # 断言 x 转换为字符串后结果为 '1.0'
        assert_(str(x) == '1.0')
        # 创建 C 类的实例 y
        y = C(2.0)
        # 断言 y 转换为字符串后结果为 '2.0'
        assert_(str(y) == '2.0')
        # 创建 D 类的实例 z
        z = D(3.0)
        # 断言 z 转换为字符串后结果为 '3.0'
        assert_(str(z) == '3.0')

    # 定义测试方法 test_init2
    def test_init2(self):
        # 创建 B0 类的实例 x
        x = B0(1.0)
        # 断言 x 转换为字符串后结果为 '1.0'
        assert_(str(x) == '1.0')
        # 创建 C0 类的实例 y
        y = C0(2.0)
        # 断言 y 转换为字符串后结果为 '2.0'
        assert_(str(y) == '2.0')

    # 定义测试方法 test_gh_15395
    def test_gh_15395(self):
        # 创建 B1 类的实例 x
        x = B1(1.0)
        # 断言 x 转换为字符串后结果为 '1.0'
        assert_(str(x) == '1.0')

        # 使用 pytest 检查以下代码是否引发 TypeError 异常
        with pytest.raises(TypeError):
            B1(1.0, 2.0)


# 定义测试类 TestCharacter
class TestCharacter:
    # 定义测试方法 test_char_radd
    def test_char_radd(self):
        # 测试 np.bytes_ 和 np.str_ 的加法操作
        # np.bytes_ 对象 np_s 和 np.str_ 对象 np_u
        np_s = np.bytes_('abc')
        np_u = np.str_('abc')
        # 字节串 b'def' 和 字符串 'def'
        s = b'def'
        u = 'def'
        # 断言 np_s 与 np_u 的右加操作结果为 NotImplemented
        assert_(np_s.__radd__(np_s) is NotImplemented)
        assert_(np_s.__radd__(np_u) is NotImplemented)
        assert_(np_s.__radd__(s) is NotImplemented)
        assert_(np_s.__radd__(u) is NotImplemented)
        assert_(np_u.__radd__(np_s) is NotImplemented)
        assert_(np_u.__radd__(np_u) is NotImplemented)
        assert_(np_u.__radd__(s) is NotImplemented)
        assert_(np_u.__radd__(u) is NotImplemented)
        # 断言 s + np_s 的结果为 b'defabc'
        assert_(s + np_s == b'defabc')
        # 断言 u + np_u 的结果为 'defabc'

        # 定义类 MyStr，继承自 str 和 np.generic
        class MyStr(str, np.generic):
            # 空类，用于测试特殊情况
            pass

        # 使用 assert_raises 检查以下代码是否引发 TypeError 异常
        with assert_raises(TypeError):
            ret = s + MyStr('abc')

        # 定义类 MyBytes，继承自 bytes 和 np.generic
        class MyBytes(bytes, np.generic):
            # 空类，用于测试特殊情况
            pass

        # 测试 MyBytes 类的加法操作
        ret = s + MyBytes(b'abc')
        # 断言 ret 的类型与 s 的类型相同
        assert(type(ret) is type(s))
        # 断言 ret 的值等于 b"defabc"

    # 定义测试方法 test_char_repeat
    def test_char_repeat(self):
        # np.bytes_ 对象 np_s 和 np.str_ 对象 np_u
        np_s = np.bytes_('abc')
        np_u = np.str_('abc')
        # 预期的重复结果
        res_s = b'abc' * 5
        res_u = 'abc' * 5
        # 断言 np_s 的重复 5 次结果等于 res_s
        assert_(np_s * 5 == res_s)
        # 断言 np_u 的重复 5 次结果等于 res_u
        assert_(np_u * 5 == res_u)
```