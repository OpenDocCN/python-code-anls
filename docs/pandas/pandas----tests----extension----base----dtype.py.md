# `D:\src\scipysrc\pandas\pandas\tests\extension\base\dtype.py`

```
    # 导入必要的库
    import numpy as np
    import pytest

    # 导入 pandas 库及其测试工具模块
    import pandas as pd
    import pandas._testing as tm
    from pandas.api.types import (
        infer_dtype,
        is_object_dtype,
        is_string_dtype,
    )

class BaseDtypeTests:
    """ExtensionDtype 类的基类"""

    # 测试 dtype 的名称是否为字符串类型
    def test_name(self, dtype):
        assert isinstance(dtype.name, str)

    # 测试 dtype 的种类是否符合预期
    def test_kind(self, dtype):
        valid = set("biufcmMOSUV")
        assert dtype.kind in valid

    # 测试从名称判断 dtype 是否匹配
    def test_is_dtype_from_name(self, dtype):
        result = type(dtype).is_dtype(dtype.name)
        assert result is True

    # 测试 dtype 是否能正确识别数据类型
    def test_is_dtype_unboxes_dtype(self, data, dtype):
        assert dtype.is_dtype(data) is True

    # 测试从 dtype 自身判断 dtype 是否匹配
    def test_is_dtype_from_self(self, dtype):
        result = type(dtype).is_dtype(dtype)
        assert result is True

    # 测试不同类型的输入是否能正确判断为非字符串类型
    def test_is_dtype_other_input(self, dtype):
        assert dtype.is_dtype([1, 2, 3]) is False

    # 测试 dtype 是否不是字符串类型
    def test_is_not_string_type(self, dtype):
        assert not is_string_dtype(dtype)

    # 测试 dtype 是否不是对象类型
    def test_is_not_object_type(self, dtype):
        assert not is_object_dtype(dtype)

    # 测试 dtype 是否等于其名称，但不等于名称加后缀
    def test_eq_with_str(self, dtype):
        assert dtype == dtype.name
        assert dtype != dtype.name + "-suffix"

    # 测试 dtype 是否不等于 numpy 的对象类型
    def test_eq_with_numpy_object(self, dtype):
        assert dtype != np.dtype("object")

    # 测试 dtype 是否等于自身，但不等于其它对象
    def test_eq_with_self(self, dtype):
        assert dtype == dtype
        assert dtype != object()

    # 测试构造数组类型是否与数据类型相匹配
    def test_array_type(self, data, dtype):
        assert dtype.construct_array_type() is type(data)

    # 测试检查数据类型是否与预期相符
    def test_check_dtype(self, data):
        dtype = data.dtype

        # 使用 .dtypes 检查等价性
        df = pd.DataFrame(
            {
                "A": pd.Series(data, dtype=dtype),
                "B": data,
                "C": pd.Series(["foo"] * len(data), dtype=object),
                "D": 1,
            }
        )
        result = df.dtypes == str(dtype)
        assert np.dtype("int64") != "Int64"

        expected = pd.Series([True, True, False, False], index=list("ABCD"))

        # 断言 Series 相等
        tm.assert_series_equal(result, expected)

        expected = pd.Series([True, True, False, False], index=list("ABCD"))
        result = df.dtypes.apply(str) == str(dtype)

        # 断言 Series 相等
        tm.assert_series_equal(result, expected)

    # 测试 dtype 是否可哈希
    def test_hashable(self, dtype):
        hash(dtype)  # 不应该报错

    # 测试 dtype 的字符串表示是否与其名称相符
    def test_str(self, dtype):
        assert str(dtype) == dtype.name

    # 测试 dtype 是否等于其名称，但不等于另一种类型
    def test_eq(self, dtype):
        assert dtype == dtype.name
        assert dtype != "anonther_type"

    # 测试从字符串构造自身名称的 dtype 是否正确
    def test_construct_from_string_own_name(self, dtype):
        result = dtype.construct_from_string(dtype.name)
        assert type(result) is type(dtype)

        # 作为类方法检查是否可行
        result = type(dtype).construct_from_string(dtype.name)
        assert type(result) is type(dtype)
    # 测试用例：测试当传入另一种类型时是否会引发异常
    def test_construct_from_string_another_type_raises(self, dtype):
        # 构建异常信息，表明无法从 'another_type' 构建成 dtype 的对象
        msg = f"Cannot construct a '{type(dtype).__name__}' from 'another_type'"
        # 使用 pytest 检查是否引发了 TypeError 异常，并匹配特定的异常消息
        with pytest.raises(TypeError, match=msg):
            type(dtype).construct_from_string("another_type")

    # 测试用例：测试当传入错误类型时是否会引发异常
    def test_construct_from_string_wrong_type_raises(self, dtype):
        # 使用 pytest 检查是否引发了 TypeError 异常，并匹配特定的异常消息
        with pytest.raises(
            TypeError,
            match="'construct_from_string' expects a string, got <class 'int'>",
        ):
            type(dtype).construct_from_string(0)

    # 测试用例：测试获取通用数据类型的函数
    def test_get_common_dtype(self, dtype):
        # 实际情况中，我们通常不会使用长度为1的列表调用此函数
        # （我们直接使用该 dtype 作为通用数据类型），但作为良好的实践，仍然测试此情况的正常工作
        # 这也是我们通常能够测试的唯一情况
        # 断言获取通用数据类型函数返回的结果与传入的 dtype 相同
        assert dtype._get_common_dtype([dtype]) == dtype

    # 测试用例：测试推断数据类型的函数
    @pytest.mark.parametrize("skipna", [True, False])
    def test_infer_dtype(self, data, data_missing, skipna):
        # 只测试此函数能否正常工作而不引发错误
        # 调用推断数据类型函数，传入数据和 skipna 参数
        res = infer_dtype(data, skipna=skipna)
        # 断言返回结果是字符串类型
        assert isinstance(res, str)
        # 再次调用推断数据类型函数，传入缺失数据和 skipna 参数
        res = infer_dtype(data_missing, skipna=skipna)
        # 断言返回结果是字符串类型
        assert isinstance(res, str)
```