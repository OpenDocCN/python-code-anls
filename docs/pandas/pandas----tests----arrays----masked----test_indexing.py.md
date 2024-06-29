# `D:\src\scipysrc\pandas\pandas\tests\arrays\masked\test_indexing.py`

```
# 导入必要的模块
import re  # 导入正则表达式模块
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试

import pandas as pd  # 导入pandas库，用于数据处理

# 定义测试类 TestSetitemValidation
class TestSetitemValidation:
    
    # 定义内部方法，用于检查设置无效值时的异常情况
    def _check_setitem_invalid(self, arr, invalid):
        # 构造异常消息，指出数据类型和无效值
        msg = f"Invalid value '{invalid!s}' for dtype {arr.dtype}"
        msg = re.escape(msg)  # 转义消息中的特殊字符以便匹配

        # 使用 pytest 检查赋值操作是否引发预期的 TypeError 异常，消息匹配给定的正则表达式
        with pytest.raises(TypeError, match=msg):
            arr[0] = invalid  # 检查单个元素赋值

        with pytest.raises(TypeError, match=msg):
            arr[:] = invalid  # 检查切片赋值

        with pytest.raises(TypeError, match=msg):
            arr[[0]] = invalid  # 检查索引列表赋值

        # 下面的代码是待修复的部分，暂时注释掉不执行
        # with pytest.raises(TypeError):
        #    arr[[0]] = [invalid]

        # with pytest.raises(TypeError):
        #    arr[[0]] = np.array([invalid], dtype=object)

        # 创建 Series 对象，检查设置单个元素时是否引发预期的 TypeError 异常
        ser = pd.Series(arr)
        with pytest.raises(TypeError, match=msg):
            ser[0] = invalid
            # TODO: so, so many other variants of this...

    # 定义无效标量值列表
    _invalid_scalars = [
        1 + 2j,  # 复数
        "True",  # 字符串 "True"
        "1",  # 字符串 "1"
        "1.0",  # 字符串 "1.0"
        pd.NaT,  # pandas 中的 NaT (Not a Time)
        np.datetime64("NaT"),  # NumPy 中的 datetime64 的 NaT
        np.timedelta64("NaT"),  # NumPy 中的 timedelta64 的 NaT
    ]

    # 参数化测试，针对不同的无效标量值执行测试
    @pytest.mark.parametrize(
        "invalid", _invalid_scalars + [1, 1.0, np.int64(1), np.float64(1)]
    )
    def test_setitem_validation_scalar_bool(self, invalid):
        # 创建布尔类型的 pandas array
        arr = pd.array([True, False, None], dtype="boolean")
        self._check_setitem_invalid(arr, invalid)

    # 参数化测试，针对不同的无效标量值执行测试
    @pytest.mark.parametrize("invalid", _invalid_scalars + [True, 1.5, np.float64(1.5)])
    def test_setitem_validation_scalar_int(self, invalid, any_int_ea_dtype):
        # 创建整数类型的 pandas array
        arr = pd.array([1, 2, None], dtype=any_int_ea_dtype)
        self._check_setitem_invalid(arr, invalid)

    # 参数化测试，针对不同的无效标量值执行测试
    @pytest.mark.parametrize("invalid", _invalid_scalars + [True])
    def test_setitem_validation_scalar_float(self, invalid, float_ea_dtype):
        # 创建浮点数类型的 pandas array
        arr = pd.array([1, 2, None], dtype=float_ea_dtype)
        self._check_setitem_invalid(arr, invalid)
```