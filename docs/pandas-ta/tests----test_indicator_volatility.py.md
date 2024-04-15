# `.\pandas-ta\tests\test_indicator_volatility.py`

```
# 从.config模块中导入error_analysis，sample_data，CORRELATION，CORRELATION_THRESHOLD和VERBOSE变量
# 从.context模块中导入pandas_ta
from .config import error_analysis, sample_data, CORRELATION, CORRELATION_THRESHOLD, VERBOSE
from .context import pandas_ta

# 从unittest模块中导入TestCase和skip类
from unittest import TestCase, skip
# 导入pandas测试工具模块，重命名为pdt
import pandas.testing as pdt
# 从pandas模块中导入DataFrame和Series类
from pandas import DataFrame, Series
# 导入talib库，重命名为tal
import talib as tal

# 定义TestVolatility类，继承自TestCase类
class TestVolatility(TestCase):

    # 类方法，设置测试类的初始状态
    @classmethod
    def setUpClass(cls):
        # 将sample_data赋值给类属性data
        cls.data = sample_data
        # 将data的列名转换为小写
        cls.data.columns = cls.data.columns.str.lower()
        # 将data的"open"列赋值给类属性open
        cls.open = cls.data["open"]
        # 将data的"high"列赋值给类属性high
        cls.high = cls.data["high"]
        # 将data的"low"列赋值给类属性low
        cls.low = cls.data["low"]
        # 将data的"close"列赋值给类属性close
        cls.close = cls.data["close"]
        # 如果data的列中包含"volume"列
        if "volume" in cls.data.columns:
            # 将data的"volume"列赋值给类属性volume
            cls.volume = cls.data["volume"]

    # 类方法，清理测试类的状态
    @classmethod
    def tearDownClass(cls):
        # 删除类属性open
        del cls.open
        # 删除类属性high
        del cls.high
        # 删除类属性low
        del cls.low
        # 删除类属性close
        del cls.close
        # 如果类中存在volume属性
        if hasattr(cls, "volume"):
            # 删除类属性volume
            del cls.volume
        # 删除类属性data
        del cls.data

    # 实例方法，设置每个测试用例的初始状态
    def setUp(self): pass

    # 实例方法，清理每个测试用例的状态
    def tearDown(self): pass

    # 测试aberration函数
    def test_aberration(self):
        # 调用pandas_ta.aberration函数，传入self.high、self.low和self.close作为参数，返回结果赋值给result
        result = pandas_ta.aberration(self.high, self.low, self.close)
        # 断言result的类型为DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言result的name属性等于"ABER_5_15"
        self.assertEqual(result.name, "ABER_5_15")

    # 测试accbands函数
    def test_accbands(self):
        # 调用pandas_ta.accbands函数，传入self.high、self.low和self.close作为参数，返回结果赋值给result
        result = pandas_ta.accbands(self.high, self.low, self.close)
        # 断言result的类型为DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言result的name属性等于"ACCBANDS_20"
        self.assertEqual(result.name, "ACCBANDS_20")

    # 测试atr函数
    def test_atr(self):
        # 调用pandas_ta.atr函数，传入self.high、self.low、self.close和talib=False作为参数，返回结果赋值给result
        result = pandas_ta.atr(self.high, self.low, self.close, talib=False)
        # 断言result的类型为Series
        self.assertIsInstance(result, Series)
        # 断言result的name属性等于"ATRr_14"
        self.assertEqual(result.name, "ATRr_14")

        try:
            # 使用talib库计算ATR，期望结果赋值给expected
            expected = tal.ATR(self.high, self.low, self.close)
            # 断言result与expected相等，不检查名称
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 计算result与expected之间的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于CORRELATION_THRESHOLD
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 调用error_analysis函数，传入result、CORRELATION和异常对象ex作为参数
                error_analysis(result, CORRELATION, ex)

        # 调用pandas_ta.atr函数，传入self.high、self.low和self.close作为参数，返回结果赋值给result
        result = pandas_ta.atr(self.high, self.low, self.close)
        # 断言result的类型为Series
        self.assertIsInstance(result, Series)
        # 断言result的name属性等于"ATRr_14"
        self.assertEqual(result.name, "ATRr_14")
    # 测试布林带指标函数
    def test_bbands(self):
        # 调用布林带指标函数，计算布林带指标
        result = pandas_ta.bbands(self.close, talib=False)
        # 断言结果为 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "BBANDS_5_2.0"
        self.assertEqual(result.name, "BBANDS_5_2.0")

        try:
            # 使用 TA-Lib 计算布林带指标的期望值
            expected = tal.BBANDS(self.close)
            # 将期望值转换为 DataFrame
            expecteddf = DataFrame({"BBU_5_2.0": expected[0], "BBM_5_2.0": expected[1], "BBL_5_2.0": expected[2]})
            # 使用 Pandas 的 assert_frame_equal 函数比较结果和期望值
            pdt.assert_frame_equal(result, expecteddf)
        except AssertionError:
            try:
                # 计算结果与期望值的相关性
                bbl_corr = pandas_ta.utils.df_error_analysis(result.iloc[:, 0], expecteddf.iloc[:,0], col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(bbl_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 执行错误分析函数
                error_analysis(result.iloc[:, 0], CORRELATION, ex)

            try:
                bbm_corr = pandas_ta.utils.df_error_analysis(result.iloc[:, 1], expecteddf.iloc[:,1], col=CORRELATION)
                self.assertGreater(bbm_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.iloc[:, 1], CORRELATION, ex, newline=False)

            try:
                bbu_corr = pandas_ta.utils.df_error_analysis(result.iloc[:, 2], expecteddf.iloc[:,2], col=CORRELATION)
                self.assertGreater(bbu_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.iloc[:, 2], CORRELATION, ex, newline=False)

        # 调用布林带指标函数，设置自由度调整参数为 0
        result = pandas_ta.bbands(self.close, ddof=0)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "BBANDS_5_2.0")

        # 调用布林带指标函数，设置自由度调整参数为 1
        result = pandas_ta.bbands(self.close, ddof=1)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "BBANDS_5_2.0")

    # 测试唐奇安通道函数
    def test_donchian(self):
        # 调用唐奇安通道函数，计算唐奇安通道
        result = pandas_ta.donchian(self.high, self.low)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "DC_20_20")

        # 调用唐奇安通道函数，设置下界长度为 20，上界长度为 5，计算唐奇安通道
        result = pandas_ta.donchian(self.high, self.low, lower_length=20, upper_length=5)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "DC_20_5")

    # 测试 Keltner 通道函数
    def test_kc(self):
        # 调用 Keltner 通道函数，计算 Keltner 通道
        result = pandas_ta.kc(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "KCe_20_2")

        # 调用 Keltner 通道函数，设置移动平均模式为 "sma"，计算 Keltner 通道
        result = pandas_ta.kc(self.high, self.low, self.close, mamode="sma")
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "KCs_20_2")

    # 测试梅斯线指标函数
    def test_massi(self):
        # 调用梅斯线指标函数，计算梅斯线指标
        result = pandas_ta.massi(self.high, self.low)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MASSI_9_25")
    # 测试 NATR 指标函数
    def test_natr(self):
        # 调用 pandas_ta 库中的 natr 函数计算 NATR 指标
        result = pandas_ta.natr(self.high, self.low, self.close, talib=False)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "NATR_14"
        self.assertEqual(result.name, "NATR_14")

        try:
            # 使用 TA-Lib 计算 NATR 指标的预期结果
            expected = tal.NATR(self.high, self.low, self.close)
            # 断言结果与预期结果一致，不检查名称
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 计算结果与预期结果之间的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 执行错误分析
                error_analysis(result, CORRELATION, ex)

        # 重新调用 pandas_ta 库中的 natr 函数，不指定 TA-Lib
        result = pandas_ta.natr(self.high, self.low, self.close)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "NATR_14"
        self.assertEqual(result.name, "NATR_14")

    # 测试 PDIST 指标函数
    def test_pdist(self):
        # 调用 pandas_ta 库中的 pdist 函数计算 PDIST 指标
        result = pandas_ta.pdist(self.open, self.high, self.low, self.close)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "PDIST"
        self.assertEqual(result.name, "PDIST")

    # 测试 RVI 指标函数
    def test_rvi(self):
        # 调用 pandas_ta 库中的 rvi 函数计算 RVI 指标
        result = pandas_ta.rvi(self.close)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "RVI_14"

        # 调用 pandas_ta 库中的 rvi 函数计算 RVI 指标，使用高低价
        result = pandas_ta.rvi(self.close, self.high, self.low, refined=True)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "RVIr_14"

        # 调用 pandas_ta 库中的 rvi 函数计算 RVI 指标，使用三分法
        result = pandas_ta.rvi(self.close, self.high, self.low, thirds=True)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "RVIt_14"

    # 测试 THERMO 指标函数
    def test_thermo(self):
        # 调用 pandas_ta 库中的 thermo 函数计算 THERMO 指标
        result = pandas_ta.thermo(self.high, self.low)
        # 断言结果为 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "THERMO_20_2_0.5"

    # 测试 TRUE_RANGE 指标函数
    def test_true_range(self):
        # 调用 pandas_ta 库中的 true_range 函数计算 TRUE_RANGE 指标
        result = pandas_ta.true_range(self.high, self.low, self.close, talib=False)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "TRUERANGE_1"

        try:
            # 使用 TA-Lib 计算 TRUE_RANGE 指标的预期结果
            expected = tal.TRANGE(self.high, self.low, self.close)
            # 断言结果与预期结果一致，不检查名称
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 计算结果与预期结果之间的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 执行错误分析
                error_analysis(result, CORRELATION, ex)

        # 重新调用 pandas_ta 库中的 true_range 函数，不指定 TA-Lib
        result = pandas_ta.true_range(self.high, self.low, self.close)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "TRUERANGE_1"

    # 测试 UI 指标函数
    def test_ui(self):
        # 调用 pandas_ta 库中的 ui 函数计算 UI 指标
        result = pandas_ta.ui(self.close)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "UI_14"

        # 调用 pandas_ta 库中的 ui 函数计算 UI 指标，包含 everget 参数
        result = pandas_ta.ui(self.close, everget=True)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "UIe_14"
```