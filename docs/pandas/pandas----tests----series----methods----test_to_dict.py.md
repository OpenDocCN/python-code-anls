# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_to_dict.py`

```
# 导入必要的模块和库
import collections  # 导入collections模块，用于创建特定类型的字典
import numpy as np  # 导入numpy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试用例

from pandas import Series  # 从pandas库中导入Series数据结构
import pandas._testing as tm  # 导入pandas测试模块中的tm别名

# 定义一个测试类TestSeriesToDict，用于测试Series对象转换为字典的功能
class TestSeriesToDict:

    # 使用pytest的参数化装饰器，定义测试方法test_to_dict，用于测试Series对象转换为字典的行为
    @pytest.mark.parametrize(
        "mapping", (dict, collections.defaultdict(list), collections.OrderedDict)
    )
    def test_to_dict(self, mapping, datetime_series):
        # GH#16122
        # 将datetime_series转换为字典，并使用指定的mapping类型封装为Series对象，命名为"ts"
        result = Series(datetime_series.to_dict(into=mapping), name="ts")
        # 复制datetime_series作为期望结果
        expected = datetime_series.copy()
        # 将期望结果的索引的频率设置为None
        expected.index = expected.index._with_freq(None)
        # 断言result与expected是否相等
        tm.assert_series_equal(result, expected)

        # 从Series对象直接生成collections.Counter类型的结果，并进行断言比较
        from_method = Series(datetime_series.to_dict(into=collections.Counter))
        # 使用collections.Counter直接构造Series对象，并进行断言比较
        from_constructor = Series(collections.Counter(datetime_series.items()))
        # 断言两个Series对象是否相等
        tm.assert_series_equal(from_method, from_constructor)

    # 使用pytest的参数化装饰器，定义测试方法test_to_dict_return_types，用于测试字典中值的返回类型
    @pytest.mark.parametrize(
        "input",
        (
            {"a": np.int64(64), "b": 10},
            {"a": np.int64(64), "b": 10, "c": "ABC"},
            {"a": np.uint64(64), "b": 10, "c": "ABC"},
        ),
    )
    def test_to_dict_return_types(self, input):
        # GH25969

        # 将输入字典input转换为Series对象，并再次转换为普通字典d
        d = Series(input).to_dict()
        # 断言字典中键"a"对应的值是否为int类型
        assert isinstance(d["a"], int)
        # 断言字典中键"b"对应的值是否为int类型
        assert isinstance(d["b"], int)
```