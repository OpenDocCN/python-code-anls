# `D:\src\scipysrc\pandas\pandas\tests\generic\test_series.py`

```
from operator import methodcaller  # 导入methodcaller函数，用于调用指定名称的方法

import numpy as np  # 导入NumPy库，通常用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试

import pandas as pd  # 导入Pandas库，用于数据操作和分析
from pandas import (  # 导入Pandas中的多个模块和函数
    MultiIndex,  # 导入MultiIndex类，用于多级索引
    Series,  # 导入Series类，用于处理一维数据结构
    date_range,  # 导入date_range函数，用于生成日期范围
)
import pandas._testing as tm  # 导入Pandas内部测试模块

class TestSeries:
    @pytest.mark.parametrize("func", ["rename_axis", "_set_axis_name"])
    def test_set_axis_name_mi(self, func):
        # 创建一个带有多级索引的Series对象
        ser = Series(
            [11, 21, 31],
            index=MultiIndex.from_tuples(
                [("A", x) for x in ["a", "B", "c"]], names=["l1", "l2"]
            ),
        )

        # 使用methodcaller调用指定的方法(func参数指定)，对ser进行操作
        result = methodcaller(func, ["L1", "L2"])(ser)
        
        # 断言原始Series对象的索引名为None
        assert ser.index.name is None
        # 断言原始Series对象的索引名列表为["l1", "l2"]
        assert ser.index.names == ["l1", "l2"]
        # 断言操作后的结果Series对象的索引名为None
        assert result.index.name is None
        # 断言操作后的结果Series对象的索引名列表为["L1", "L2"]
        assert result.index.names, ["L1", "L2"]

    def test_set_axis_name_raises(self):
        # 创建一个包含单个元素的Series对象
        ser = Series([1])
        msg = "No axis named 1 for object type Series"
        # 使用pytest.raises检查是否会引发指定类型的异常，并匹配给定的错误消息
        with pytest.raises(ValueError, match=msg):
            ser._set_axis_name(name="a", axis=1)

    def test_get_bool_data_preserve_dtype(self):
        # 创建一个包含布尔值的Series对象
        ser = Series([True, False, True])
        # 调用_get_bool_data方法，获取处理后的结果
        result = ser._get_bool_data()
        # 使用Pandas的测试工具tm来断言两个Series对象是否相等
        tm.assert_series_equal(result, ser)

    @pytest.mark.parametrize("data", [np.nan, pd.NaT, True, False])
    def test_nonzero_single_element_raise_1(self, data):
        # 创建一个包含单个元素的Series对象
        series = Series([data])

        msg = "The truth value of a Series is ambiguous"
        # 使用pytest.raises检查是否会引发指定类型的异常，并匹配给定的错误消息
        with pytest.raises(ValueError, match=msg):
            bool(series)

    @pytest.mark.parametrize("data", [(True, True), (False, False)])
    def test_nonzero_multiple_element_raise(self, data):
        # 创建一个包含元组的Series对象
        msg_err = "The truth value of a Series is ambiguous"
        series = Series([data])
        # 使用pytest.raises检查是否会引发指定类型的异常，并匹配给定的错误消息
        with pytest.raises(ValueError, match=msg_err):
            bool(series)

    @pytest.mark.parametrize("data", [1, 0, "a", 0.0])
    def test_nonbool_single_element_raise(self, data):
        # 创建一个包含单个非布尔值的Series对象
        msg_err1 = "The truth value of a Series is ambiguous"
        series = Series([data])
        # 使用pytest.raises检查是否会引发指定类型的异常，并匹配给定的错误消息
        with pytest.raises(ValueError, match=msg_err1):
            bool(series)

    def test_metadata_propagation_indiv_resample(self):
        # 创建一个时间序列Series对象
        ts = Series(
            np.random.default_rng(2).random(1000),
            index=date_range("20130101", periods=1000, freq="s"),
            name="foo",
        )
        # 对时间序列进行重新采样操作，计算均值
        result = ts.resample("1min").mean()
        # 使用Pandas的测试工具tm来断言两个Series对象的元数据是否等效
        tm.assert_metadata_equivalent(ts, result)

        # 对时间序列进行重新采样操作，计算最小值
        result = ts.resample("1min").min()
        # 使用Pandas的测试工具tm来断言两个Series对象的元数据是否等效
        tm.assert_metadata_equivalent(ts, result)

        # 对时间序列进行重新采样操作，应用自定义函数（计算和）
        result = ts.resample("1min").apply(lambda x: x.sum())
        # 使用Pandas的测试工具tm来断言两个Series对象的元数据是否等效
        tm.assert_metadata_equivalent(ts, result)
    # 定义单元测试方法，用于测试元数据在操作中的传播情况，使用 monkeypatch 进行模拟
    def test_metadata_propagation_indiv(self, monkeypatch):
        # 检查结果操作中的元数据是否匹配

        # 创建一个名为 "foo" 的 Series 对象，其索引和数据均为 [0, 1, 2]
        ser = Series(range(3), range(3))
        ser.name = "foo"
        # 创建另一个名为 "bar" 的 Series 对象，其索引和数据均为 [0, 1, 2]
        ser2 = Series(range(3), range(3))
        ser2.name = "bar"

        # 对 Series 对象进行转置操作
        result = ser.T
        # 断言转置后的结果的元数据与原 Series 对象一致
        tm.assert_metadata_equivalent(ser, result)

        # 定义一个自定义的 finalize 方法，用于处理元数据的合并
        def finalize(self, other, method=None, **kwargs):
            # 遍历当前对象的元数据属性列表
            for name in self._metadata:
                # 如果是连接操作且属性名为 "filename"
                if method == "concat" and name == "filename":
                    # 将所有对象的 "filename" 属性值连接起来
                    value = "+".join(
                        [
                            getattr(obj, name)
                            for obj in other.objs
                            if getattr(obj, name, None)
                        ]
                    )
                    # 设置当前对象的 "filename" 属性为连接后的值
                    object.__setattr__(self, name, value)
                else:
                    # 否则，将当前对象的元数据属性设置为 other 对象对应的属性值
                    object.__setattr__(self, name, getattr(other, name, None))

            return self

        # 使用 monkeypatch 设置 Series 类的元数据属性列表和 __finalize__ 方法
        with monkeypatch.context() as m:
            m.setattr(Series, "_metadata", ["name", "filename"])
            m.setattr(Series, "__finalize__", finalize)

            # 设置 ser 和 ser2 的 filename 属性
            ser.filename = "foo"
            ser2.filename = "bar"

            # 进行 Series 对象的连接操作
            result = pd.concat([ser, ser2])
            # 断言连接后的结果的 filename 属性为 "foo+bar"
            assert result.filename == "foo+bar"
            # 断言连接后的结果的 name 属性为 None
            assert result.name is None
```