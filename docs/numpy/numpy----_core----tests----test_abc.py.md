# `.\numpy\numpy\_core\tests\test_abc.py`

```py
# 导入需要的模块
from numpy.testing import assert_

import numbers

import numpy as np
from numpy._core.numerictypes import sctypes
# 测试用例类
class TestABC:
    # 测试抽象类
    def test_abstract(self):
        # 判断np.number是否是numbers.Number的子类
        assert_(issubclass(np.number, numbers.Number))

        # 判断np.inexact是否是numbers.Complex的子类
        assert_(issubclass(np.inexact, numbers.Complex))
        # 判断np.complexfloating是否是numbers.Complex的子类
        assert_(issubclass(np.complexfloating, numbers.Complex))
        # 判断np.floating是否是numbers.Real的子类
        assert_(issubclass(np.floating, numbers.Real))

        # 判断np.integer是否是numbers.Integral的子类
        assert_(issubclass(np.integer, numbers.Integral))
        # 判断np.signedinteger是否是numbers.Integral的子类
        assert_(issubclass(np.signedinteger, numbers.Integral))
        # 判断np.unsignedinteger是否是numbers.Integral的子类
        assert_(issubclass(np.unsignedinteger, numbers.Integral))

    # 测试浮点数类型
    def test_floats(self):
        # 遍历浮点数类型
        for t in sctypes['float']:
            # 判断是否是numbers.Real的实例
            assert_(isinstance(t(), numbers.Real),
                    f"{t.__name__} is not instance of Real")
            # 判断是否是numbers.Real的子类
            assert_(issubclass(t, numbers.Real),
                    f"{t.__name__} is not subclass of Real")
            # 判断不是numbers.Rational的实例
            assert_(not isinstance(t(), numbers.Rational),
                    f"{t.__name__} is instance of Rational")
            # 判断不是numbers.Rational的子类
            assert_(not issubclass(t, numbers.Rational),
                    f"{t.__name__} is subclass of Rational")

    # 测试复数类型
    def test_complex(self):
        # 遍历复数类型
        for t in sctypes['complex']:
            # 判断是否是numbers.Complex的实例
            assert_(isinstance(t(), numbers.Complex),
                    f"{t.__name__} is not instance of Complex")
            # 判断是否是numbers.Complex的子类
            assert_(issubclass(t, numbers.Complex),
                    f"{t.__name__} is not subclass of Complex")
            # 判断不是numbers.Real的实例
            assert_(not isinstance(t(), numbers.Real),
                    f"{t.__name__} is instance of Real")
            # 判断不是numbers.Real的子类
            assert_(not issubclass(t, numbers.Real),
                    f"{t.__name__} is subclass of Real")

    # 测试整数类型
    def test_int(self):
        # 遍历整数类型
        for t in sctypes['int']:
            # 判断是否是numbers.Integral的实例
            assert_(isinstance(t(), numbers.Integral),
                    f"{t.__name__} is not instance of Integral")
            # 判断是否是numbers.Integral的子类
            assert_(issubclass(t, numbers.Integral),
                    f"{t.__name__} is not subclass of Integral")

    # 测试无符号整数类型
    def test_uint(self):
        # 遍历无符号整数类型
        for t in sctypes['uint']:
            # 判断是否是numbers.Integral的实例
            assert_(isinstance(t(), numbers.Integral),
                    f"{t.__name__} is not instance of Integral")
            # 判断是否是numbers.Integral的子类
            assert_(issubclass(t, numbers.Integral),
                    f"{t.__name__} is not subclass of Integral")
```