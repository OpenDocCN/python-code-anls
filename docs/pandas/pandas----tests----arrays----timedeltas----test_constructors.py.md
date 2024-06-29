# `D:\src\scipysrc\pandas\pandas\tests\arrays\timedeltas\test_constructors.py`

```
# 导入所需的库
import numpy as np
import pytest

# 从 pandas 库中导入 TimedeltaArray 类
from pandas.core.arrays import TimedeltaArray

# 测试 TimedeltaArray 构造函数的各种情况
class TestTimedeltaArrayConstructor:

    # 测试当传入的数据类型无法转换为 timedelta64[ns] 类型时是否会引发 TypeError
    def test_other_type_raises(self):
        # 错误消息字符串，指明预期的异常信息
        msg = r"dtype bool cannot be converted to timedelta64\[ns\]"
        # 使用 pytest 检查是否会引发指定类型的异常，并匹配消息字符串
        with pytest.raises(TypeError, match=msg):
            TimedeltaArray._from_sequence(np.array([1, 2, 3], dtype="bool"))

    # 测试当传入的 dtype 不符合 timedelta64 要求时是否会引发 ValueError
    def test_incorrect_dtype_raises(self):
        # 第一个测试用例：传入的 dtype 是 'category'，但应该是 np.timedelta64 类型
        msg = "dtype 'category' is invalid, should be np.timedelta64 dtype"
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence(
                np.array([1, 2, 3], dtype="i8"), dtype="category"
            )

        # 第二个测试用例：传入的 dtype 是 'int64'，但应该是 np.timedelta64 类型
        msg = "dtype 'int64' is invalid, should be np.timedelta64 dtype"
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence(
                np.array([1, 2, 3], dtype="i8"), dtype=np.dtype("int64")
            )

        # 第三个测试用例：传入的 dtype 是 'datetime64[ns]'，但应该是 np.timedelta64 类型
        msg = r"dtype 'datetime64\[ns\]' is invalid, should be np.timedelta64 dtype"
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence(
                np.array([1, 2, 3], dtype="i8"), dtype=np.dtype("M8[ns]")
            )

        # 第四个测试用例：传入的 dtype 是 'datetime64[us, UTC]'，但应该是 np.timedelta64 类型
        msg = (
            r"dtype 'datetime64\[us, UTC\]' is invalid, should be np.timedelta64 dtype"
        )
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence(
                np.array([1, 2, 3], dtype="i8"), dtype="M8[us, UTC]"
            )

        # 第五个测试用例：传入的 dtype 是 'm8[Y]'，但支持的 timedelta64 分辨率只有 's', 'ms', 'us', 'ns'
        msg = "Supported timedelta64 resolutions are 's', 'ms', 'us', 'ns'"
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence(
                np.array([1, 2, 3], dtype="i8"), dtype=np.dtype("m8[Y]")
            )

    # 测试 TimedeltaArray._from_sequence 方法中的 copy 参数
    def test_copy(self):
        # 创建测试数据，dtype 是 'm8[ns]'，即 timedelta64[ns] 类型
        data = np.array([1, 2, 3], dtype="m8[ns]")
        # 测试 copy=False 的情况
        arr = TimedeltaArray._from_sequence(data, copy=False)
        # 断言 arr._ndarray 与原始 data 是同一个对象
        assert arr._ndarray is data

        # 测试 copy=True 的情况
        arr = TimedeltaArray._from_sequence(data, copy=True)
        # 断言 arr._ndarray 不是原始 data 对象
        assert arr._ndarray is not data
        # 断言 arr._ndarray 的基础数据不是原始 data 对象
        assert arr._ndarray.base is not data

    # 测试 TimedeltaArray._from_sequence 方法中的 dtype 参数
    def test_from_sequence_dtype(self):
        # 测试传入 dtype='object' 时是否会引发 ValueError
        msg = "dtype 'object' is invalid, should be np.timedelta64 dtype"
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence([], dtype=object)
```