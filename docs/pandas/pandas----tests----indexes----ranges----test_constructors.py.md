# `D:\src\scipysrc\pandas\pandas\tests\indexes\ranges\test_constructors.py`

```
from datetime import datetime  # 导入 datetime 模块中的 datetime 类

import numpy as np  # 导入 numpy 库，并使用 np 作为别名
import pytest  # 导入 pytest 测试框架

from pandas import (  # 从 pandas 库中导入以下模块
    Index,  # 导入 Index 类
    RangeIndex,  # 导入 RangeIndex 类
    Series,  # 导入 Series 类
)
import pandas._testing as tm  # 导入 pandas._testing 模块，并使用 tm 作为别名


class TestRangeIndexConstructors:  # 定义 TestRangeIndexConstructors 类

    @pytest.mark.parametrize("name", [None, "foo"])  # 使用 pytest 参数化装饰器，测试 name 参数为 None 和 "foo"
    @pytest.mark.parametrize(
        "args, kwargs, start, stop, step",
        [  # 使用 pytest 参数化装饰器，测试不同参数组合
            ((5,), {}, 0, 5, 1),
            ((1, 5), {}, 1, 5, 1),
            ((1, 5, 2), {}, 1, 5, 2),
            ((0,), {}, 0, 0, 1),
            ((0, 0), {}, 0, 0, 1),
            ((), {"start": 0}, 0, 0, 1),
            ((), {"stop": 0}, 0, 0, 1),
        ],
    )
    def test_constructor(self, args, kwargs, start, stop, step, name):
        # 测试 RangeIndex 构造函数
        result = RangeIndex(*args, name=name, **kwargs)  # 使用给定参数创建 RangeIndex 对象
        expected = Index(np.arange(start, stop, step, dtype=np.int64), name=name)  # 创建预期的 Index 对象
        assert isinstance(result, RangeIndex)  # 断言 result 是 RangeIndex 类型的对象
        assert result.name is name  # 断言 result 的 name 属性与给定的 name 相等
        assert result._range == range(start, stop, step)  # 断言 result 的 _range 属性与给定的 range 相等
        tm.assert_index_equal(result, expected, exact="equiv")  # 使用 pandas._testing 模块中的 assert_index_equal 函数比较 result 和 expected

    def test_constructor_invalid_args(self):
        # 测试 RangeIndex 构造函数的无效参数情况
        msg = "RangeIndex\\(\\.\\.\\.\\) must be called with integers"
        with pytest.raises(TypeError, match=msg):  # 使用 pytest 的 raises 断言捕获 TypeError 异常，验证异常消息符合预期
            RangeIndex()

        with pytest.raises(TypeError, match=msg):
            RangeIndex(name="Foo")

        # we don't allow on a bare Index
        msg = (
            r"Index\(\.\.\.\) must be called with a collection of some "
            r"kind, 0 was passed"
        )
        with pytest.raises(TypeError, match=msg):
            Index(0)

    @pytest.mark.parametrize(
        "args",
        [  # 使用 pytest 参数化装饰器，测试不同的参数类型
            Index(["a", "b"]),
            Series(["a", "b"]),
            np.array(["a", "b"]),
            [],
            np.arange(0, 10),
            np.array([1]),
            [1],
        ],
    )
    def test_constructor_additional_invalid_args(self, args):
        # 测试 RangeIndex 构造函数的额外无效参数情况
        msg = f"Value needs to be a scalar value, was type {type(args).__name__}"
        with pytest.raises(TypeError, match=msg):  # 使用 pytest 的 raises 断言捕获 TypeError 异常，验证异常消息符合预期
            RangeIndex(args)

    @pytest.mark.parametrize("args", ["foo", datetime(2000, 1, 1, 0, 0)])
    def test_constructor_invalid_args_wrong_type(self, args):
        # 测试 RangeIndex 构造函数的参数类型错误情况
        msg = f"Wrong type {type(args)} for value {args}"
        with pytest.raises(TypeError, match=msg):  # 使用 pytest 的 raises 断言捕获 TypeError 异常，验证异常消息符合预期
            RangeIndex(args)

    def test_constructor_same(self):
        # 测试 RangeIndex 构造函数相同参数情况下的行为
        # pass thru w and w/o copy
        index = RangeIndex(1, 5, 2)
        result = RangeIndex(index, copy=False)
        assert result.identical(index)  # 断言 result 与 index 对象相同

        result = RangeIndex(index, copy=True)
        tm.assert_index_equal(result, index, exact=True)  # 使用 pandas._testing 模块中的 assert_index_equal 函数比较 result 和 index

        result = RangeIndex(index)
        tm.assert_index_equal(result, index, exact=True)  # 使用 pandas._testing 模块中的 assert_index_equal 函数比较 result 和 index

        with pytest.raises(
            ValueError,
            match="Incorrect `dtype` passed: expected signed integer, received float64",
        ):
            RangeIndex(index, dtype="float64")
    # 测试 RangeIndex 类的构造函数，使用 range 对象初始化索引
    def test_constructor_range_object(self):
        # 创建 RangeIndex 对象并进行断言比较
        result = RangeIndex(range(1, 5, 2))
        expected = RangeIndex(1, 5, 2)
        tm.assert_index_equal(result, expected, exact=True)

    # 测试 RangeIndex 类的 from_range 方法
    def test_constructor_range(self):
        # 测试正常情况下的 from_range 方法使用
        result = RangeIndex.from_range(range(1, 5, 2))
        expected = RangeIndex(1, 5, 2)
        tm.assert_index_equal(result, expected, exact=True)

        # 测试只有起始和结束的情况
        result = RangeIndex.from_range(range(5, 6))
        expected = RangeIndex(5, 6, 1)
        tm.assert_index_equal(result, expected, exact=True)

        # 测试起始大于结束的情况，应返回空的 RangeIndex 对象
        # 此处提供了一个无效的范围
        result = RangeIndex.from_range(range(5, 1))
        expected = RangeIndex(0, 0, 1)
        tm.assert_index_equal(result, expected, exact=True)

        # 测试仅给定结束值的情况
        result = RangeIndex.from_range(range(5))
        expected = RangeIndex(0, 5, 1)
        tm.assert_index_equal(result, expected, exact=True)

        # 使用 Index 类创建 RangeIndex 对象并进行比较
        result = Index(range(1, 5, 2))
        expected = RangeIndex(1, 5, 2)
        tm.assert_index_equal(result, expected, exact=True)

        # 测试 from_range 方法不应接受额外的 copy 参数
        msg = (
            r"(RangeIndex.)?from_range\(\) got an unexpected keyword argument( 'copy')?"
        )
        # 使用 pytest 的 raises 断言检查是否抛出预期的 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            RangeIndex.from_range(range(10), copy=True)

    # 测试 RangeIndex 类的构造函数中的命名相关功能
    def test_constructor_name(self):
        # GH#12288，验证命名功能是否正常工作
        orig = RangeIndex(10)
        orig.name = "original"

        # 创建原始 RangeIndex 的副本并命名
        copy = RangeIndex(orig)
        copy.name = "copy"

        # 断言原始对象和副本的名称是否正确
        assert orig.name == "original"
        assert copy.name == "copy"

        # 将副本转换为 Index 类并检查名称是否正确传递
        new = Index(copy)
        assert new.name == "copy"

        # 修改新的 Index 对象的名称，并验证各个对象的名称是否如预期更改
        new.name = "new"
        assert orig.name == "original"
        assert copy.name == "copy"
        assert new.name == "new"

    # 测试 RangeIndex 类的构造函数中的边界情况处理
    def test_constructor_corner(self):
        # 创建一个包含对象类型的数组和 RangeIndex 对象
        arr = np.array([1, 2, 3, 4], dtype=object)
        index = RangeIndex(1, 5)
        # 断言 RangeIndex 对象的值类型为 np.int64
        assert index.values.dtype == np.int64
        expected = Index(arr).astype("int64")

        # 使用 pandas 测试工具断言 index 对象和 expected 相等
        tm.assert_index_equal(index, expected, exact="equiv")

        # 测试不允许的非整数类型参数，应该引发 TypeError 异常
        with pytest.raises(TypeError, match=r"Wrong type \<class 'str'\>"):
            RangeIndex("1", "10", "1")
        with pytest.raises(TypeError, match=r"Wrong type \<class 'float'\>"):
            RangeIndex(1.1, 10.2, 1.3)

        # 测试不允许的传递类型参数，应该引发 ValueError 异常
        with pytest.raises(
            ValueError,
            match="Incorrect `dtype` passed: expected signed integer, received float64",
        ):
            RangeIndex(1, 5, dtype="float64")
```