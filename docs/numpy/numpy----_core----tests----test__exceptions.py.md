# `.\numpy\numpy\_core\tests\test__exceptions.py`

```
"""
Tests of the ._exceptions module. Primarily for exercising the __str__ methods.
"""

import pickle  # 导入pickle模块，用于序列化和反序列化对象

import pytest  # 导入pytest模块，用于编写和运行测试
import numpy as np  # 导入NumPy库，并使用np作为别名
from numpy.exceptions import AxisError  # 从NumPy异常中导入AxisError异常类

_ArrayMemoryError = np._core._exceptions._ArrayMemoryError  # 定义_ArrayMemoryError为NumPy中的_ArrayMemoryError异常类
_UFuncNoLoopError = np._core._exceptions._UFuncNoLoopError  # 定义_UFuncNoLoopError为NumPy中的_UFuncNoLoopError异常类

class TestArrayMemoryError:
    def test_pickling(self):
        """ Test that _ArrayMemoryError can be pickled """
        error = _ArrayMemoryError((1023,), np.dtype(np.uint8))  # 创建_ArrayMemoryError异常的实例
        res = pickle.loads(pickle.dumps(error))  # 序列化和反序列化error对象
        assert res._total_size == error._total_size  # 断言反序列化后的对象的_total_size属性与原对象相等

    def test_str(self):
        e = _ArrayMemoryError((1023,), np.dtype(np.uint8))  # 创建_ArrayMemoryError异常的实例
        str(e)  # 测试调用异常对象的__str__方法不会导致崩溃

    # testing these properties is easier than testing the full string repr
    def test__size_to_string(self):
        """ Test e._size_to_string """
        f = _ArrayMemoryError._size_to_string  # 获取_ArrayMemoryError类的_size_to_string静态方法的引用
        Ki = 1024  # 定义Ki为1024
        assert f(0) == '0 bytes'  # 测试_size_to_string方法处理0字节的情况
        assert f(1) == '1 bytes'  # 测试_size_to_string方法处理1字节的情况
        assert f(1023) == '1023 bytes'  # 测试_size_to_string方法处理1023字节的情况
        assert f(Ki) == '1.00 KiB'  # 测试_size_to_string方法处理1024字节的情况
        assert f(Ki+1) == '1.00 KiB'  # 测试_size_to_string方法处理1025字节的情况
        assert f(10*Ki) == '10.0 KiB'  # 测试_size_to_string方法处理10240字节的情况
        assert f(int(999.4*Ki)) == '999. KiB'  # 测试_size_to_string方法处理999.4 KiB的情况
        assert f(int(1023.4*Ki)) == '1023. KiB'  # 测试_size_to_string方法处理1023.4 KiB的情况
        assert f(int(1023.5*Ki)) == '1.00 MiB'  # 测试_size_to_string方法处理1023.5 KiB的情况
        assert f(Ki*Ki) == '1.00 MiB'  # 测试_size_to_string方法处理1048576字节的情况
        assert f(int(Ki*Ki*Ki*0.9999)) == '1.00 GiB'  # 测试_size_to_string方法处理1073741824字节的情况
        assert f(Ki*Ki*Ki*Ki*Ki*Ki) == '1.00 EiB'  # 测试_size_to_string方法处理1152921504606846976字节的情况
        # larger than sys.maxsize, adding larger prefixes isn't going to help
        # anyway.
        assert f(Ki*Ki*Ki*Ki*Ki*Ki*123456) == '123456. EiB'  # 测试_size_to_string方法处理大于sys.maxsize的情况

    def test__total_size(self):
        """ Test e._total_size """
        e = _ArrayMemoryError((1,), np.dtype(np.uint8))  # 创建_ArrayMemoryError异常的实例
        assert e._total_size == 1  # 断言_total_size属性的值为1

        e = _ArrayMemoryError((2, 4), np.dtype((np.uint64, 16)))  # 创建_ArrayMemoryError异常的实例
        assert e._total_size == 1024  # 断言_total_size属性的值为1024


class TestUFuncNoLoopError:
    def test_pickling(self):
        """ Test that _UFuncNoLoopError can be pickled """
        assert isinstance(pickle.dumps(_UFuncNoLoopError), bytes)  # 断言_UFuncNoLoopError对象可以被序列化为字节流


@pytest.mark.parametrize("args", [
    (2, 1, None),
    (2, 1, "test_prefix"),
    ("test message",),
])
class TestAxisError:
    def test_attr(self, args):
        """Validate attribute types."""
        exc = AxisError(*args)  # 根据参数args创建AxisError异常的实例
        if len(args) == 1:
            assert exc.axis is None  # 如果args只有一个元素，断言axis属性为None
            assert exc.ndim is None  # 如果args只有一个元素，断言ndim属性为None
        else:
            axis, ndim, *_ = args
            assert exc.axis == axis  # 断言axis属性与args中的第一个元素相等
            assert exc.ndim == ndim  # 断言ndim属性与args中的第二个元素相等

    def test_pickling(self, args):
        """Test that `AxisError` can be pickled."""
        exc = AxisError(*args)  # 根据参数args创建AxisError异常的实例
        exc2 = pickle.loads(pickle.dumps(exc))  # 序列化和反序列化exc对象

        assert type(exc) is type(exc2)  # 断言exc和exc2的类型相同
        for name in ("axis", "ndim", "args"):
            attr1 = getattr(exc, name)
            attr2 = getattr(exc2, name)
            assert attr1 == attr2, name  # 断言exc和exc2的各个属性值相等
```