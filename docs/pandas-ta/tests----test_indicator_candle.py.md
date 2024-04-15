# `.\pandas-ta\tests\test_indicator_candle.py`

```py
# 从项目的config模块中导入错误分析、样本数据、相关性、相关性阈值和详细程度
from .config import error_analysis, sample_data, CORRELATION, CORRELATION_THRESHOLD, VERBOSE
# 从项目的context模块中导入pandas_ta
from .context import pandas_ta

# 导入TestCase类和skip装饰器从unittest模块
from unittest import TestCase, skip
# 导入pandas测试模块作为pdt别名
import pandas.testing as pdt
# 从pandas模块导入DataFrame和Series类
from pandas import DataFrame, Series

# 导入talib库作为tal别名
import talib as tal

# 定义测试类TestCandle，继承自TestCase类
class TestCandle(TestCase):
    # 设置测试类的类方法setUpClass，在所有测试方法之前运行
    @classmethod
    def setUpClass(cls):
        # 初始化测试数据
        cls.data = sample_data
        # 将列名转换为小写
        cls.data.columns = cls.data.columns.str.lower()
        # 初始化开盘价、最高价、最低价和收盘价
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        # 如果数据中包含成交量列，则初始化成交量
        if "volume" in cls.data.columns:
            cls.volume = cls.data["volume"]

    # 设置测试类的类方法tearDownClass，在所有测试方法之后运行
    @classmethod
    def tearDownClass(cls):
        # 删除开盘价、最高价、最低价和收盘价
        del cls.open
        del cls.high
        del cls.low
        del cls.close
        # 如果测试数据中存在成交量列，则删除成交量
        if hasattr(cls, "volume"):
            del cls.volume
        # 删除测试数据
        del cls.data

    # 设置测试方法的setUp，在每个测试方法之前运行
    def setUp(self): pass
    
    # 设置测试方法的tearDown，在每个测试方法之后运行
    def tearDown(self): pass

    # 测试Heikin-Ashi指标的方法
    def test_ha(self):
        # 计算Heikin-Ashi指标
        result = pandas_ta.ha(self.open, self.high, self.low, self.close)
        # 断言结果类型为DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果名称为"Heikin-Ashi"
        self.assertEqual(result.name, "Heikin-Ashi")

    # 测试蜡烛图形态指标的方法
    def test_cdl_pattern(self):
        # 计算所有蜡烛图形态指标
        result = pandas_ta.cdl_pattern(self.open, self.high, self.low, self.close, name="all")
        # 断言结果类型为DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果列数与蜡烛图形态指标名称列表长度相等
        self.assertEqual(len(result.columns), len(pandas_ta.CDL_PATTERN_NAMES))

        # 计算特定蜡烛图形态指标（doji）的方法
        result = pandas_ta.cdl_pattern(self.open, self.high, self.low, self.close, name="doji")
        # 断言结果类型为DataFrame
        self.assertIsInstance(result, DataFrame)

        # 计算多个蜡烛图形态指标（doji、inside）的方法
        result = pandas_ta.cdl_pattern(self.open, self.high, self.low, self.close, name=["doji", "inside"])
        # 断言结果类型为DataFrame
        self.assertIsInstance(result, DataFrame)

    # 测试Doji蜡烛图形态指标的方法
    def test_cdl_doji(self):
        # 计算Doji蜡烛图形态指标
        result = pandas_ta.cdl_doji(self.open, self.high, self.low, self.close)
        # 断言结果类型为Series
        self.assertIsInstance(result, Series)
        # 断言结果名称为"CDL_DOJI_10_0.1"
        self.assertEqual(result.name, "CDL_DOJI_10_0.1")

        try:
            # 使用talib计算Doji蜡烛图形态指标
            expected = tal.CDLDOJI(self.open, self.high, self.low, self.close)
            # 断言结果与talib计算结果相等
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 如果断言失败，计算结果与talib计算结果的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于相关性阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果出现异常，调用错误分析函数
                error_analysis(result, CORRELATION, ex)

    # 测试Inside蜡烛图形态指标的方法
    def test_cdl_inside(self):
        # 计算Inside蜡烛图形态指标
        result = pandas_ta.cdl_inside(self.open, self.high, self.low, self.close)
        # 断言结果类型为Series
        self.assertIsInstance(result, Series)
        # 断言结果名称为"CDL_INSIDE"
        self.assertEqual(result.name, "CDL_INSIDE")

        # 计算Inside蜡烛图形态指标，并转换为布尔值
        result = pandas_ta.cdl_inside(self.open, self.high, self.low, self.close, asbool=True)
        # 断言结果类型为Series
        self.assertIsInstance(result, Series)
        # 断言结果名称为"CDL_INSIDE"
        self.assertEqual(result.name, "CDL_INSIDE")

    # 测试Z蜡烛图形态指标的方法
    def test_cdl_z(self):
        # 计算Z蜡烛图形态指标
        result = pandas_ta.cdl_z(self.open, self.high, self.low, self.close)
        # 断言结果类型为DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果名称为"CDL_Z_30_1"
        self.assertEqual(result.name, "CDL_Z_30_1")
```