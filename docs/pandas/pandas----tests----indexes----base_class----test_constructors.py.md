# `D:\src\scipysrc\pandas\pandas\tests\indexes\base_class\test_constructors.py`

```
# 导入需要的库
import numpy as np
import pytest

# 导入 pandas 库及其子模块
import pandas as pd
from pandas import (
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm

# 定义 Index 构造器的测试类
class TestIndexConstructor:
    # 测试 Index 构造器的情况，特别是不返回子类的情况

    @pytest.mark.parametrize("value", [1, np.int64(1)])
    def test_constructor_corner(self, value):
        # 边界情况测试
        msg = (
            r"Index\(\.\.\.\) must be called with a collection of some "
            f"kind, {value} was passed"
        )
        # 使用 pytest 检查是否会抛出 TypeError 异常，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            Index(value)

    @pytest.mark.parametrize("index_vals", [[("A", 1), "B"], ["B", ("A", 1)]])
    def test_construction_list_mixed_tuples(self, index_vals):
        # 参见 gh-10697：如果我们从混合的元组列表构造，确保我们与排序顺序无关。
        # 构造 Index 对象
        index = Index(index_vals)
        # 断言 index 是 Index 类的实例，但不是 MultiIndex 类的实例
        assert isinstance(index, Index)
        assert not isinstance(index, MultiIndex)

    def test_constructor_cast(self):
        # 测试类型转换
        msg = "could not convert string to float"
        # 使用 pytest 检查是否会抛出 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match=msg):
            Index(["a", "b", "c"], dtype=float)

    @pytest.mark.parametrize("tuple_list", [[()], [(), ()]])
    def test_construct_empty_tuples(self, tuple_list):
        # GH #45608
        # 构造 Index 对象
        result = Index(tuple_list)
        # 使用 MultiIndex 的 from_tuples 方法构造期望的结果
        expected = MultiIndex.from_tuples(tuple_list)
        # 使用 pandas 测试模块 tm 来断言两个 Index 对象相等
        tm.assert_index_equal(result, expected)

    def test_index_string_inference(self):
        # GH#54430
        # 如果没有安装 pyarrow，则跳过测试
        pytest.importorskip("pyarrow")
        # 定义预期的 Index 对象
        dtype = "string[pyarrow_numpy]"
        expected = Index(["a", "b"], dtype=dtype)
        # 使用 pd.option_context 来设置未来推断字符串的选项，构造 Index 对象
        with pd.option_context("future.infer_string", True):
            ser = Index(["a", "b"])
        # 使用 pandas 测试模块 tm 来断言两个 Index 对象相等
        tm.assert_index_equal(ser, expected)

        expected = Index(["a", 1], dtype="object")
        # 使用 pd.option_context 来设置未来推断字符串的选项，构造 Index 对象
        with pd.option_context("future.infer_string", True):
            ser = Index(["a", 1])
        # 使用 pandas 测试模块 tm 来断言两个 Index 对象相等
        tm.assert_index_equal(ser, expected)

    @pytest.mark.parametrize("klass", [Series, Index])
    def test_inference_on_pandas_objects(self, klass):
        # GH#56012
        # 构造 pandas 对象
        obj = klass([pd.Timestamp("2019-12-31")], dtype=object)
        # 使用构造器构造 Index 对象
        result = Index(obj)
        # 断言结果的 dtype 是 np.object_
        assert result.dtype == np.object_

    def test_constructor_not_read_only(self):
        # GH#57130
        # 构造 Series 对象
        ser = Series([1, 2], dtype=object)
        # 使用 Series 对象构造 Index 对象
        idx = Index(ser)
        # 断言索引对象的值是可写的
        assert idx._values.flags.writeable
```