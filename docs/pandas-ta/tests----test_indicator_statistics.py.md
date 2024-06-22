# `.\pandas-ta\tests\test_indicator_statistics.py`

```py
# 从.config模块中导入error_analysis，sample_data，CORRELATION，CORRELATION_THRESHOLD，VERBOSE变量
# 从.context模块中导入pandas_ta模块
from .config import error_analysis, sample_data, CORRELATION, CORRELATION_THRESHOLD, VERBOSE
from .context import pandas_ta

# 从unittest模块中导入skip，TestCase类
from unittest import skip, TestCase
# 导入pandas的测试模块
import pandas.testing as pdt
# 从pandas模块中导入DataFrame，Series类
from pandas import DataFrame, Series
# 导入talib库，并重命名为tal
import talib as tal

# 定义测试Statistics类，继承自TestCase类
class TestStatistics(TestCase):
    # 类方法，用于设置测试类的初始状态
    @classmethod
    def setUpClass(cls):
        # 将sample_data赋值给类属性data
        cls.data = sample_data
        # 将data的列名转换为小写
        cls.data.columns = cls.data.columns.str.lower()
        # 分别将open、high、low、close列赋值给对应的类属性
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        # 如果data的列中包含volume列，则将volume列赋值给类属性volume
        if "volume" in cls.data.columns:
            cls.volume = cls.data["volume"]

    # 类方法，用于清理测试类的状态
    @classmethod
    def tearDownClass(cls):
        # 删除类属性open、high、low、close
        del cls.open
        del cls.high
        del cls.low
        del cls.close
        # 如果类中有volume属性，则删除volume属性
        if hasattr(cls, "volume"):
            del cls.volume
        # 删除类属性data
        del cls.data

    # 空的setUp方法
    def setUp(self): pass

    # 空的tearDown方法
    def tearDown(self): pass

    # 测试entropy方法
    def test_entropy(self):
        # 调用pandas_ta模块中的entropy方法，计算close列的熵
        result = pandas_ta.entropy(self.close)
        # 断言result的类型为Series
        self.assertIsInstance(result, Series)
        # 断言result的名称为"ENTP_10"
        self.assertEqual(result.name, "ENTP_10")

    # 测试kurtosis方法
    def test_kurtosis(self):
        # 调用pandas_ta模块中的kurtosis方法，计算close列的峰度
        result = pandas_ta.kurtosis(self.close)
        # 断言result的类型为Series
        self.assertIsInstance(result, Series)
        # 断言result的名称为"KURT_30"
        self.assertEqual(result.name, "KURT_30")

    # 测试mad方法
    def test_mad(self):
        # 调用pandas_ta模块中的mad方法，计算close列的绝对平均偏差
        result = pandas_ta.mad(self.close)
        # 断言result的类型为Series
        self.assertIsInstance(result, Series)
        # 断言result的名称为"MAD_30"
        self.assertEqual(result.name, "MAD_30")

    # 测试median方法
    def test_median(self):
        # 调用pandas_ta模块中的median方法，计算close列的中位数
        result = pandas_ta.median(self.close)
        # 断言result的类型为Series
        self.assertIsInstance(result, Series)
        # 断言result的名称为"MEDIAN_30"
        self.assertEqual(result.name, "MEDIAN_30")

    # 测试quantile方法
    def test_quantile(self):
        # 调用pandas_ta模块中的quantile方法，计算close列的分位数
        result = pandas_ta.quantile(self.close)
        # 断言result的类型为Series
        self.assertIsInstance(result, Series)
        # 断言result的名称为"QTL_30_0.5"
        self.assertEqual(result.name, "QTL_30_0.5")

    # 测试skew方法
    def test_skew(self):
        # 调用pandas_ta模块中的skew方法，计算close列的偏度
        result = pandas_ta.skew(self.close)
        # 断言result的类型为Series
        self.assertIsInstance(result, Series)
        # 断言result的名称为"SKEW_30"
        self.assertEqual(result.name, "SKEW_30")

    # 测试stdev方法
    def test_stdev(self):
        # 调用pandas_ta模块中的stdev方法，计算close列的标准差（talib=False）
        result = pandas_ta.stdev(self.close, talib=False)
        # 断言result的类型为Series
        self.assertIsInstance(result, Series)
        # 断言result的名称为"STDEV_30"
        self.assertEqual(result.name, "STDEV_30")

        # 尝试使用tal.STDDEV方法计算close列的标准差
        try:
            # 期望的结果使用tal.STDDEV方法计算
            expected = tal.STDDEV(self.close, 30)
            # 断言result与expected相等，忽略名称检查
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 如果断言失败，计算result与expected之间的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于阈值CORRELATION_THRESHOLD
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果出现异常，则调用error_analysis函数，并传递result，CORRELATION，异常信息ex
                error_analysis(result, CORRELATION, ex)

        # 重新调用pandas_ta模块中的stdev方法，计算close列的标准差
        result = pandas_ta.stdev(self.close)
        # 断言result的类型为Series
        self.assertIsInstance(result, Series)
        # 断言result的名称为"STDEV_30"
        self.assertEqual(result.name, "STDEV_30")
    # 测试 TOS_STDEVALL 函数
    def test_tos_sdtevall(self):
        # 调用 TOS_STDEVALL 函数，计算标准差
        result = pandas_ta.tos_stdevall(self.close)
        # 断言结果为 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果列名为 "TOS_STDEVALL"
        self.assertEqual(result.name, "TOS_STDEVALL")
        # 断言结果列数为 7
        self.assertEqual(len(result.columns), 7)

        # 调用 TOS_STDEVALL 函数，计算长度为 30 的标准差
        result = pandas_ta.tos_stdevall(self.close, length=30)
        # 断言结果为 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果列名为 "TOS_STDEVALL_30"
        self.assertEqual(result.name, "TOS_STDEVALL_30")
        # 断言结果列数为 7
        self.assertEqual(len(result.columns), 7)

        # 调用 TOS_STDEVALL 函数，计算长度为 30，标准差为 1 和 2 的标准差
        result = pandas_ta.tos_stdevall(self.close, length=30, stds=[1, 2])
        # 断言结果为 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果列名为 "TOS_STDEVALL_30"
        self.assertEqual(result.name, "TOS_STDEVALL_30")
        # 断言结果列数为 5
        self.assertEqual(len(result.columns), 5)

    # 测试 Variance 函数
    def test_variance(self):
        # 调用 Variance 函数，计算方差
        result = pandas_ta.variance(self.close, talib=False)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果列名为 "VAR_30"
        self.assertEqual(result.name, "VAR_30")

        try:
            # 使用 Talib 计算期望结果
            expected = tal.VAR(self.close, 30)
            # 断言结果与期望结果相等
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 计算结果与期望结果的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 分析错误
                error_analysis(result, CORRELATION, ex)

        # 调用 Variance 函数，默认使用 Talib 计算，计算方差
        result = pandas_ta.variance(self.close)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果列名为 "VAR_30"
        self.assertEqual(result.name, "VAR_30")

    # 测试 Z-Score 函数
    def test_zscore(self):
        # 调用 Z-Score 函数，计算 Z 分数
        result = pandas_ta.zscore(self.close)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果列名为 "ZS_30"
        self.assertEqual(result.name, "ZS_30")
```