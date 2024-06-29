# `D:\src\scipysrc\pandas\pandas\tests\arrays\interval\test_astype.py`

```
import pytest  # 导入 pytest 模块

from pandas import (  # 导入 pandas 库中的以下子模块：
    Categorical,  # Categorical 类，用于处理分类数据
    CategoricalDtype,  # CategoricalDtype 类，定义分类数据类型
    Index,  # Index 类，用于索引操作
    IntervalIndex,  # IntervalIndex 类，用于处理区间索引
)
import pandas._testing as tm  # 导入 pandas 内部的测试模块 pandas._testing

class TestAstype:  # 定义一个测试类 TestAstype
    @pytest.mark.parametrize("ordered", [True, False])  # 使用 pytest 的参数化装饰器，定义参数化测试，参数 ordered 取值为 True 和 False
    def test_astype_categorical_retains_ordered(self, ordered):  # 定义测试方法 test_astype_categorical_retains_ordered，接受参数 ordered
        index = IntervalIndex.from_breaks(range(5))  # 创建一个 IntervalIndex，从断点列表 range(5) 中创建

        arr = index._data  # 获取 IntervalIndex 的数据数组

        dtype = CategoricalDtype(None, ordered=ordered)  # 创建一个 CategoricalDtype 对象，没有具体类别，但指定了是否有序

        expected = Categorical(list(arr), ordered=ordered)  # 创建预期的 Categorical 对象，用数据数组 arr 初始化，指定是否有序
        result = arr.astype(dtype)  # 将数据数组 arr 转换为指定的 dtype 类型

        assert result.ordered is ordered  # 断言结果的 ordered 属性与预期的 ordered 值相等
        tm.assert_categorical_equal(result, expected)  # 使用测试模块中的方法验证 result 和 expected 是否相等

        # test IntervalIndex.astype while we're at it.
        result = index.astype(dtype)  # 将 IntervalIndex 对象 index 转换为指定的 dtype 类型
        expected = Index(expected)  # 创建一个 Index 对象，用 expected 初始化
        tm.assert_index_equal(result, expected)  # 使用测试模块中的方法验证 result 和 expected 是否相等
```