# `D:\src\scipysrc\pandas\pandas\tests\base\test_constructors.py`

```
# 从 datetime 模块中导入 datetime 类
from datetime import datetime
# 导入 sys 模块
import sys

# 导入 numpy 库并重命名为 np
import numpy as np
# 导入 pytest 库
import pytest

# 从 pandas.compat 中导入 PYPY 变量
from pandas.compat import PYPY

# 导入 pandas 库并重命名为 pd
import pandas as pd
# 从 pandas 中导入 DataFrame、Index、Series 类
from pandas import (
    DataFrame,
    Index,
    Series,
)
# 导入 pandas._testing 模块并重命名为 tm
import pandas._testing as tm
# 从 pandas.core.accessor 中导入 PandasDelegate 类
from pandas.core.accessor import PandasDelegate
# 从 pandas.core.base 中导入 NoNewAttributesMixin、PandasObject 类
from pandas.core.base import (
    NoNewAttributesMixin,
    PandasObject,
)


# 定义函数 series_via_frame_from_dict，通过字典创建 DataFrame，并返回 'a' 列的 Series 对象
def series_via_frame_from_dict(x, **kwargs):
    return DataFrame({"a": x}, **kwargs)["a"]


# 定义函数 series_via_frame_from_scalar，通过标量创建 DataFrame，并返回第一列的元素
def series_via_frame_from_scalar(x, **kwargs):
    return DataFrame(x, **kwargs)[0]


# 定义测试用的 fixture constructor，参数化测试对象包括 Series、series_via_frame_from_dict、series_via_frame_from_scalar、Index
@pytest.fixture(
    params=[
        Series,
        series_via_frame_from_dict,
        series_via_frame_from_scalar,
        Index,
    ],
    ids=["Series", "DataFrame-dict", "DataFrame-array", "Index"],
)
def constructor(request):
    return request.param


# 定义 TestPandasDelegate 类
class TestPandasDelegate:
    # 定义内部类 Delegator
    class Delegator:
        _properties = ["prop"]
        _methods = ["test_method"]

        # 设置属性 prop 的 setter 方法
        def _set_prop(self, value):
            self.prop = value

        # 获取属性 prop 的 getter 方法
        def _get_prop(self):
            return self.prop

        # 定义 prop 属性，使用 _get_prop 和 _set_prop 方法，并添加文档字符串
        prop = property(_get_prop, _set_prop, doc="foo property")

        # 定义 test_method 方法，作为测试方法
        def test_method(self, *args, **kwargs):
            """a test method"""

    # 定义内部类 Delegate，继承自 PandasDelegate 和 PandasObject
    class Delegate(PandasDelegate, PandasObject):
        # 初始化方法，接收 obj 参数
        def __init__(self, obj) -> None:
            self.obj = obj

    # 定义 test_invalid_delegation 方法，测试无效的委托情况
    def test_invalid_delegation(self):
        # 展示委托工作需要重写 _delegate_* 方法以避免引发 TypeError
        self.Delegate._add_delegate_accessors(
            delegate=self.Delegator,
            accessors=self.Delegator._properties,
            typ="property",
        )
        self.Delegate._add_delegate_accessors(
            delegate=self.Delegator, accessors=self.Delegator._methods, typ="method"
        )

        # 创建 Delegate 对象并使用 Delegator 实例作为参数
        delegate = self.Delegate(self.Delegator())

        # 测试访问 prop 属性时抛出 TypeError 异常
        msg = "You cannot access the property prop"
        with pytest.raises(TypeError, match=msg):
            delegate.prop

        # 测试设置 prop 属性时抛出 TypeError 异常
        msg = "The property prop cannot be set"
        with pytest.raises(TypeError, match=msg):
            delegate.prop = 5

        # 再次测试访问 prop 属性时抛出 TypeError 异常
        msg = "You cannot access the property prop"
        with pytest.raises(TypeError, match=msg):
            delegate.prop

    # 标记为 pytest 的跳过测试，当 PYPY 为真时，跳过该测试
    @pytest.mark.skipif(PYPY, reason="not relevant for PyPy")
    def test_memory_usage(self):
        # Delegate 类未实现 memory_usage 方法。
        # 检查是否回退到内置的 `__sizeof__`
        # GH 12924
        # 创建 Delegate 对象并使用 Delegator 实例作为参数
        delegate = self.Delegate(self.Delegator())
        # 调用 sys.getsizeof 获取对象占用内存大小
        sys.getsizeof(delegate)


# 定义 TestNoNewAttributesMixin 类，用于测试 NoNewAttributesMixin
class TestNoNewAttributesMixin:
    pass  # Placeholder, no code yet
    # 定义一个测试方法 test_mixin，用于测试 NoNewAttributesMixin 类的行为
    def test_mixin(self):
        # 定义一个名为 T 的类，继承自 NoNewAttributesMixin
        class T(NoNewAttributesMixin):
            pass

        # 创建 T 类的一个实例 t
        t = T()
        
        # 断言 t 实例没有 "__frozen" 属性
        assert not hasattr(t, "__frozen")

        # 给 t 实例添加一个名为 'a' 的属性，赋值为 "test"
        t.a = "test"
        # 断言 t 实例的 'a' 属性值为 "test"
        assert t.a == "test"

        # 调用 t 实例的 _freeze() 方法，冻结实例，使其不可再添加新属性
        t._freeze()
        
        # 断言 "__frozen" 属性现在在 t 实例的属性列表中
        assert "__frozen" in dir(t)
        # 断言 t 实例的 "__frozen" 属性值为 True
        assert getattr(t, "__frozen")
        
        # 准备用于捕获 AttributeError 异常的消息字符串
        msg = "You cannot add any new attribute"
        # 使用 pytest 检查试图给冻结的实例 t 添加新属性时是否会抛出 AttributeError 异常，并匹配消息字符串
        with pytest.raises(AttributeError, match=msg):
            t.b = "test"

        # 断言 t 实例仍然没有 'b' 属性
        assert not hasattr(t, "b")
# 定义一个测试类 TestConstruction，用于测试构造函数在不同数据类型推断上的行为，包括 Series、Index 和 DataFrame

class TestConstruction:

    @pytest.mark.parametrize(
        "a",
        [
            np.array(["2263-01-01"], dtype="datetime64[D]"),  # 创建一个包含单个日期字符串的 numpy 数组，数据类型为 datetime64[D]
            np.array([datetime(2263, 1, 1)], dtype=object),  # 创建一个包含单个 datetime 对象的 numpy 数组，数据类型为 object
            np.array([np.datetime64("2263-01-01", "D")], dtype=object),  # 创建一个包含单个 numpy 日期标量的 numpy 数组，数据类型为 object
            np.array(["2263-01-01"], dtype=object),  # 创建一个包含单个日期字符串的 numpy 数组，数据类型为 object
        ],
        ids=[
            "datetime64[D]",
            "object-datetime.datetime",
            "object-numpy-scalar",
            "object-string",
        ],
    )
    def test_constructor_datetime_outofbound(
        self, a, constructor, request, using_infer_string
    ):
        # GH-26853 (+ bug GH-26206 out of bound non-ns unit)
        # 根据 GitHub issue GH-26853 和 GH-26206 中提到的问题，处理超出边界的非纳秒单位的情况

        # 未指定数据类型时进行推断（dtype inference）
        # datetime64[non-ns] 抛出错误，其他情况结果为 object 数据类型并保留原始数据
        result = constructor(a)
        if a.dtype.kind == "M" or isinstance(a[0], np.datetime64):
            # 无法适应纳秒边界 -> 获取最接近支持的单位
            assert result.dtype == "M8[s]"
        elif isinstance(a[0], datetime):
            assert result.dtype == "M8[us]", result.dtype
        else:
            result = constructor(a)
            if using_infer_string and "object-string" in request.node.callspec.id:
                assert result.dtype == "string"
            else:
                assert result.dtype == "object"
            tm.assert_numpy_array_equal(result.to_numpy(), a)

        # 显式指定数据类型
        # 所有情况下强制转换失败 -> 均抛出错误
        msg = "Out of bounds|Out of bounds .* present at position 0"
        with pytest.raises(pd.errors.OutOfBoundsDatetime, match=msg):
            constructor(a, dtype="datetime64[ns]")

    def test_constructor_datetime_nonns(self, constructor):
        # 创建一个包含单个微秒级 datetime 字符串的 numpy 数组
        arr = np.array(["2020-01-01T00:00:00.000000"], dtype="datetime64[us]")
        # 使用私有方法 _simple_new 创建一个 DatetimeArray 对象
        dta = pd.core.arrays.DatetimeArray._simple_new(arr, dtype=arr.dtype)
        # 用 constructor 构造函数处理 dta，并期望结果的数据类型与 arr 相同
        expected = constructor(dta)
        assert expected.dtype == arr.dtype

        # 使用 constructor 处理 arr，并比较结果与预期是否相等
        result = constructor(arr)
        tm.assert_equal(result, expected)

        # 处理 GitHub issue https://github.com/pandas-dev/pandas/issues/34843
        # 将 arr 的写入标志设为不可写
        arr.flags.writeable = False
        # 使用 constructor 处理 arr，并比较结果与预期是否相等
        result = constructor(arr)
        tm.assert_equal(result, expected)
```