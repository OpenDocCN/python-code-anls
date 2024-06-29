# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_cov_corr.py`

```
import math  # 导入数学库

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest库

import pandas as pd  # 导入Pandas库
from pandas import (  # 从Pandas中导入指定模块
    Series,  # 导入Series类
    date_range,  # 导入date_range函数
    isna,  # 导入isna函数
)
import pandas._testing as tm  # 导入Pandas测试模块

class TestSeriesCov:
    def test_cov(self, datetime_series):
        # full overlap
        tm.assert_almost_equal(
            datetime_series.cov(datetime_series), datetime_series.std() ** 2
        )  # 断言两个时间序列完全重叠时的协方差近似等于标准差的平方

        # partial overlap
        tm.assert_almost_equal(
            datetime_series[:15].cov(datetime_series[5:]),
            datetime_series[5:15].std() ** 2,
        )  # 断言两个时间序列部分重叠时的协方差近似等于部分时间序列的标准差的平方

        # No overlap
        assert np.isnan(datetime_series[::2].cov(datetime_series[1::2]))  # 断言两个时间序列没有重叠时协方差为NaN

        # all NA
        cp = datetime_series[:10].copy()
        cp[:] = np.nan
        assert isna(cp.cov(cp))  # 断言全为NaN的时间序列的协方差也为NaN

        # min_periods
        assert isna(datetime_series[:15].cov(datetime_series[5:], min_periods=12))  # 断言使用min_periods参数处理部分重叠时协方差为NaN

        ts1 = datetime_series[:15].reindex(datetime_series.index)
        ts2 = datetime_series[5:].reindex(datetime_series.index)
        assert isna(ts1.cov(ts2, min_periods=12))  # 断言重建索引后的时间序列协方差为NaN

    @pytest.mark.parametrize("test_ddof", [None, 0, 1, 2, 3])
    @pytest.mark.parametrize("dtype", ["float64", "Float64"])
    def test_cov_ddof(self, test_ddof, dtype):
        # GH#34611
        np_array1 = np.random.default_rng(2).random(10)
        np_array2 = np.random.default_rng(2).random(10)

        s1 = Series(np_array1, dtype=dtype)
        s2 = Series(np_array2, dtype=dtype)

        result = s1.cov(s2, ddof=test_ddof)  # 计算两个Series对象的协方差，使用给定的自由度修正参数
        expected = np.cov(np_array1, np_array2, ddof=test_ddof)[0][1]  # 使用NumPy计算期望的协方差
        assert math.isclose(expected, result)  # 断言计算的协方差结果与期望的值接近

class TestSeriesCorr:
    def test_corr(self, datetime_series, any_float_dtype):
        stats = pytest.importorskip("scipy.stats")  # 导入scipy.stats模块，如果导入失败则跳过测试

        datetime_series = datetime_series.astype(any_float_dtype)  # 将时间序列转换为指定的浮点类型

        # full overlap
        tm.assert_almost_equal(datetime_series.corr(datetime_series), 1)  # 断言两个时间序列完全重叠时的相关系数为1

        # partial overlap
        tm.assert_almost_equal(datetime_series[:15].corr(datetime_series[5:]), 1)  # 断言两个时间序列部分重叠时的相关系数为1

        assert isna(datetime_series[:15].corr(datetime_series[5:], min_periods=12))  # 断言使用min_periods参数处理部分重叠时相关系数为NaN

        ts1 = datetime_series[:15].reindex(datetime_series.index)
        ts2 = datetime_series[5:].reindex(datetime_series.index)
        assert isna(ts1.corr(ts2, min_periods=12))  # 断言重建索引后的时间序列相关系数为NaN

        # No overlap
        assert np.isnan(datetime_series[::2].corr(datetime_series[1::2]))  # 断言两个时间序列没有重叠时相关系数为NaN

        # all NA
        cp = datetime_series[:10].copy()
        cp[:] = np.nan
        assert isna(cp.corr(cp))  # 断言全为NaN的时间序列相关系数为NaN

        A = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        result = A.corr(A)  # 计算Series对象的自相关系数
        expected, _ = stats.pearsonr(A, A)  # 使用Pearson相关系数计算期望值
        tm.assert_almost_equal(result, expected)  # 断言计算的自相关系数与期望值近似相等
    def test_corr_rank(self):
        # 导入 pytest 并检查 scipy.stats 是否可用，若不可用则跳过测试
        stats = pytest.importorskip("scipy.stats")

        # 创建 Series B，包含从0到9的浮点数，索引为日期范围从"2020-01-01"开始的10天，名称为"ts"
        B = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        
        # 创建 Series A，包含两个 [0, 1, 2, 3, 4] 的拼接，索引和名称同上
        A = Series(
            np.concatenate([np.arange(5, dtype=np.float64)] * 2),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        
        # 计算 A 和 B 的 Kendall 相关系数并与预期值比较
        result = A.corr(B, method="kendall")
        expected = stats.kendalltau(A, B)[0]
        tm.assert_almost_equal(result, expected)

        # 计算 A 和 B 的 Spearman 相关系数并与预期值比较
        result = A.corr(B, method="spearman")
        expected = stats.spearmanr(A, B)[0]
        tm.assert_almost_equal(result, expected)

        # 创建 Series A 和 B，包含给定的浮点数列表
        # 从 R 中获取的结果
        A = Series(
            [
                -0.89926396,
                0.94209606,
                -1.03289164,
                -0.95445587,
                0.76910310,
                -0.06430576,
                -2.09704447,
                0.40660407,
                -0.89926396,
                0.94209606,
            ]
        )
        B = Series(
            [
                -1.01270225,
                -0.62210117,
                -1.56895827,
                0.59592943,
                -0.01680292,
                1.17258718,
                -1.06009347,
                -0.10222060,
                -0.89076239,
                0.89372375,
            ]
        )
        
        # 预期的 Kendall 和 Spearman 相关系数
        kexp = 0.4319297
        sexp = 0.5853767
        
        # 比较 A 和 B 的 Kendall 相关系数与预期值
        tm.assert_almost_equal(A.corr(B, method="kendall"), kexp)
        
        # 比较 A 和 B 的 Spearman 相关系数与预期值
        tm.assert_almost_equal(A.corr(B, method="spearman"), sexp)

    def test_corr_invalid_method(self):
        # GH PR #22298
        # 创建具有标准正态分布的随机 Series s1 和 s2
        s1 = Series(np.random.default_rng(2).standard_normal(10))
        s2 = Series(np.random.default_rng(2).standard_normal(10))
        
        # 预期的错误消息
        msg = "method must be either 'pearson', 'spearman', 'kendall', or a callable, "
        
        # 使用 pytest 检查 s1 和 s2 的相关方法是否触发 ValueError，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            s1.corr(s2, method="____")
    def test_corr_callable_method(self, datetime_series):
        # 定义一个简单的相关性函数，当两个输入完全相等时返回1，否则返回0
        my_corr = lambda a, b: 1.0 if (a == b).all() else 0.0

        # 创建两个Series对象作为测试数据
        s1 = Series([1, 2, 3, 4, 5])
        s2 = Series([5, 4, 3, 2, 1])
        expected = 0
        # 使用自定义的相关性函数测试s1和s2的相关性，并与期望值进行比较
        tm.assert_almost_equal(s1.corr(s2, method=my_corr), expected)

        # 测试时间序列对象与自身的相关性，完全重叠的情况
        tm.assert_almost_equal(
            datetime_series.corr(datetime_series, method=my_corr), 1.0
        )

        # 测试时间序列对象的部分重叠情况下的相关性
        tm.assert_almost_equal(
            datetime_series[:15].corr(datetime_series[5:], method=my_corr), 1.0
        )

        # 测试时间序列对象没有重叠的情况，期望结果为NaN
        assert np.isnan(
            datetime_series[::2].corr(datetime_series[1::2], method=my_corr)
        )

        # 创建DataFrame对象，并测试其转置后的相关性
        df = pd.DataFrame([s1, s2])
        expected = pd.DataFrame([{0: 1.0, 1: 0}, {0: 0, 1: 1.0}])
        tm.assert_almost_equal(df.transpose().corr(method=my_corr), expected)
```