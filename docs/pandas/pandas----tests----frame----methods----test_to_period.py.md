# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_to_period.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import (  # 从 pandas 库中导入以下模块：
    DataFrame,  # 数据帧，用于存储二维数据
    DatetimeIndex,  # 日期时间索引，用于处理时间序列数据
    PeriodIndex,  # 周期索引，用于表示固定频率时间间隔的索引
    Series,  # 系列，一维标记数组，用于处理一维数据
    date_range,  # 生成日期范围
    period_range,  # 生成周期范围
)
import pandas._testing as tm  # 导入 pandas 内部测试模块，用于测试框架中的辅助函数和类


class TestToPeriod:
    def test_to_period(self, frame_or_series):
        K = 5  # 定义常量 K 为 5

        dr = date_range("1/1/2000", "1/1/2001", freq="D")  # 生成日期范围，从 2000-01-01 到 2001-01-01，频率为天
        obj = DataFrame(  # 创建 DataFrame 对象 obj
            np.random.default_rng(2).standard_normal((len(dr), K)),  # 用正态分布随机数填充的数组，作为数据
            index=dr,  # 指定索引为日期范围 dr
            columns=["A", "B", "C", "D", "E"],  # 指定列标签为 A, B, C, D, E
        )
        obj["mix"] = "a"  # 添加新列 mix，所有行设为字符串 'a'
        obj = tm.get_obj(obj, frame_or_series)  # 调用辅助函数 get_obj 处理 obj 对象

        pts = obj.to_period()  # 将时间序列转换为周期序列
        exp = obj.copy()  # 复制 obj 到 exp
        exp.index = period_range("1/1/2000", "1/1/2001")  # 生成期间范围的索引，与 exp 的索引匹配
        tm.assert_equal(pts, exp)  # 使用测试框架中的 assert_equal 函数比较 pts 和 exp

        pts = obj.to_period("M")  # 将时间序列转换为以月为频率的周期序列
        exp.index = exp.index.asfreq("M")  # 将 exp 的索引转换为月度频率
        tm.assert_equal(pts, exp)  # 使用测试框架中的 assert_equal 函数比较 pts 和 exp

    def test_to_period_without_freq(self, frame_or_series):
        # GH#7606 without freq
        idx = DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"])  # 创建日期时间索引 idx
        exp_idx = PeriodIndex(  # 创建期间索引 exp_idx
            ["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"], freq="D"  # 指定频率为天
        )

        obj = DataFrame(  # 创建 DataFrame 对象 obj
            np.random.default_rng(2).standard_normal((4, 4)),  # 用正态分布随机数填充的数组，作为数据
            index=idx,  # 指定索引为 idx
            columns=idx  # 指定列标签为 idx
        )
        obj = tm.get_obj(obj, frame_or_series)  # 调用辅助函数 get_obj 处理 obj 对象
        expected = obj.copy()  # 复制 obj 到 expected
        expected.index = exp_idx  # 将 expected 的索引设置为 exp_idx
        tm.assert_equal(obj.to_period(), expected)  # 使用测试框架中的 assert_equal 函数比较 obj 转换后的期间与 expected

        if frame_or_series is DataFrame:
            expected = obj.copy()  # 复制 obj 到 expected
            expected.columns = exp_idx  # 将 expected 的列标签设置为 exp_idx
            tm.assert_frame_equal(obj.to_period(axis=1), expected)  # 比较 DataFrame 对象的列转换为期间后的结果与 expected

    def test_to_period_columns(self):
        dr = date_range("1/1/2000", "1/1/2001")  # 生成日期范围，从 2000-01-01 到 2001-01-01
        df = DataFrame(np.random.default_rng(2).standard_normal((len(dr), 5)), index=dr)  # 创建 DataFrame 对象 df，用正态分布随机数填充的数组作为数据，索引为 dr
        df["mix"] = "a"  # 添加新列 mix，所有行设为字符串 'a'

        df = df.T  # 转置 DataFrame 对象 df
        pts = df.to_period(axis=1)  # 将 DataFrame 对象按列转换为周期序列
        exp = df.copy()  # 复制 df 到 exp
        exp.columns = period_range("1/1/2000", "1/1/2001")  # 生成期间范围的列标签，与 exp 的列标签匹配
        tm.assert_frame_equal(pts, exp)  # 使用测试框架中的 assert_frame_equal 函数比较 pts 和 exp

        pts = df.to_period("M", axis=1)  # 将 DataFrame 对象按列转换为以月为频率的周期序列
        tm.assert_index_equal(pts.columns, exp.columns.asfreq("M"))  # 使用测试框架中的 assert_index_equal 函数比较 pts 的列索引和 exp 列索引按月频率转换后的结果

    def test_to_period_invalid_axis(self):
        dr = date_range("1/1/2000", "1/1/2001")  # 生成日期范围，从 2000-01-01 到 2001-01-01
        df = DataFrame(np.random.default_rng(2).standard_normal((len(dr), 5)), index=dr)  # 创建 DataFrame 对象 df，用正态分布随机数填充的数组作为数据，索引为 dr
        df["mix"] = "a"  # 添加新列 mix，所有行设为字符串 'a'

        msg = "No axis named 2 for object type DataFrame"  # 定义错误消息
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 框架断言预期引发 ValueError 异常，并匹配错误消息
            df.to_period(axis=2)  # 尝试将 DataFrame 对象按不存在的轴（轴号为 2）转换为周期序列

    def test_to_period_raises(self, index, frame_or_series):
        # https://github.com/pandas-dev/pandas/issues/33327
        obj = Series(index=index, dtype=object)  # 创建索引为 index，数据类型为对象的 Series 对象 obj
        if frame_or_series is DataFrame:  # 如果 frame_or_series 是 DataFrame 类型
            obj = obj.to_frame()  # 将 Series 对象转换为 DataFrame 对象

        if not isinstance(index, DatetimeIndex):  # 如果索引不是 DatetimeIndex 类型
            msg = f"unsupported Type {type(index).__name__}"  # 定义错误消息，显示不支持的索引类型
            with pytest.raises(TypeError, match=msg):  # 使用 pytest 框架断言预期引发 TypeError 异常，并匹配错误消息
                obj.to_period()  # 尝试将对象转换为周期序列
```