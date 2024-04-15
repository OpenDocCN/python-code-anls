# `.\pandas-ta\tests\test_indicator_overlap.py`

```
# 从当前目录的 config 模块中导入 CORRELATION, CORRELATION_THRESHOLD, error_analysis, sample_data, VERBOSE 变量
from .config import CORRELATION, CORRELATION_THRESHOLD, error_analysis, sample_data, VERBOSE
# 从当前目录的 context 模块中导入 pandas_ta 模块
from .context import pandas_ta

# 导入 TestCase 类
from unittest import TestCase
# 导入 pandas.testing 模块，并重命名为 pdt
import pandas.testing as pdt
# 导入 DataFrame, Series 类
from pandas import DataFrame, Series
# 导入 talib 库，并重命名为 tal
import talib as tal

# 定义测试类 TestOverlap，继承自 TestCase 类
class TestOverlap(TestCase):
    # 设置类方法 setUpClass，用于设置测试类的数据
    @classmethod
    def setUpClass(cls):
        # 设置类属性 data 为 sample_data
        cls.data = sample_data
        # 将数据列名转换为小写
        cls.data.columns = cls.data.columns.str.lower()
        # 设置类属性 open 为 data 中的 "open" 列
        cls.open = cls.data["open"]
        # 设置类属性 high 为 data 中的 "high" 列
        cls.high = cls.data["high"]
        # 设置类属性 low 为 data 中的 "low" 列
        cls.low = cls.data["low"]
        # 设置类属性 close 为 data 中的 "close" 列
        cls.close = cls.data["close"]
        # 如果数据中包含 "volume" 列，则设置类属性 volume 为 data 中的 "volume" 列
        if "volume" in cls.data.columns:
            cls.volume = cls.data["volume"]

    # 设置类方法 tearDownClass，用于清理测试类的数据
    @classmethod
    def tearDownClass(cls):
        # 删除类属性 open
        del cls.open
        # 删除类属性 high
        del cls.high
        # 删除类属性 low
        del cls.low
        # 删除类属性 close
        del cls.close
        # 如果类属性中存在 volume，则删除 volume
        if hasattr(cls, "volume"):
            del cls.volume
        # 删除类属性 data
        del cls.data

    # 设置实例方法 setUp，用于测试方法的初始化
    def setUp(self): pass

    # 设置实例方法 tearDown，用于测试方法的清理
    def tearDown(self): pass

    # 定义测试方法 test_alma，测试 alma 函数
    def test_alma(self):
        # 调用 pandas_ta.alma 函数，传入 close 列作为参数
        result = pandas_ta.alma(self.close)# , length=None, sigma=None, distribution_offset=)
        # 断言 result 是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言 result 的名称为 "ALMA_10_6.0_0.85"
        self.assertEqual(result.name, "ALMA_10_6.0_0.85")

    # 定义测试方法 test_dema，测试 dema 函数
    def test_dema(self):
        # 调用 pandas_ta.dema 函数，传入 close 列和 talib=False 参数
        result = pandas_ta.dema(self.close, talib=False)
        # 断言 result 是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言 result 的名称为 "DEMA_10"
        self.assertEqual(result.name, "DEMA_10")

        try:
            # 使用 talib 计算预期值
            expected = tal.DEMA(self.close, 10)
            # 使用 pandas.testing 模块的 assert_series_equal 函数比较 result 和 expected，不检查名称
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 调用 pandas_ta.utils.df_error_analysis 函数计算 result 和 expected 之间的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于 CORRELATION_THRESHOLD
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果计算相关性时出现异常，则调用 error_analysis 函数记录异常信息
                error_analysis(result, CORRELATION, ex)

        # 再次调用 pandas_ta.dema 函数，传入 close 列，默认参数
        result = pandas_ta.dema(self.close)
        # 断言 result 是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言 result 的名称为 "DEMA_10"
        self.assertEqual(result.name, "DEMA_10")
```  
    # 测试指数移动平均值函数
    def test_ema(self):
        # 调用指数移动平均值函数，计算不带SMA的EMA
        result = pandas_ta.ema(self.close, presma=False)
        # 断言结果为Series类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为"EMA_10"
        self.assertEqual(result.name, "EMA_10")

        # 尝试使用talib库计算EMA，并进行结果对比
        try:
            # 期望值通过talib库计算
            expected = tal.EMA(self.close, 10)
            # 断言两个Series相等，忽略名称检查
            pdt.assert_series_equal(result, expected, check_names=False)
        # 如果断言失败，则进行进一步分析
        except AssertionError:
            try:
                # 计算结果和期望值之间的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            # 如果出现异常，则执行错误分析函数
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        # 调用指数移动平均值函数，计算不使用talib的EMA
        result = pandas_ta.ema(self.close, talib=False)
        # 断言结果为Series类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为"EMA_10"
        self.assertEqual(result.name, "EMA_10")

        # 尝试断言两个Series相等，忽略名称检查
        try:
            pdt.assert_series_equal(result, expected, check_names=False)
        # 如果断言失败，则进行进一步分析
        except AssertionError:
            try:
                # 计算结果和期望值之间的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            # 如果出现异常，则执行错误分析函数
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        # 调用指数移动平均值函数，计算默认的EMA
        result = pandas_ta.ema(self.close)
        # 断言结果为Series类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为"EMA_10"
        self.assertEqual(result.name, "EMA_10")

    # 测试前加权移动平均值函数
    def test_fwma(self):
        # 调用前加权移动平均值函数
        result = pandas_ta.fwma(self.close)
        # 断言结果为Series类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为"FWMA_10"
        self.assertEqual(result.name, "FWMA_10")

    # 测试高低价通道指标函数
    def test_hilo(self):
        # 调用高低价通道指标函数
        result = pandas_ta.hilo(self.high, self.low, self.close)
        # 断言结果为DataFrame类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为"HILO_13_21"
        self.assertEqual(result.name, "HILO_13_21")

    # 测试高低价中值函数
    def test_hl2(self):
        # 调用高低价中值函数
        result = pandas_ta.hl2(self.high, self.low)
        # 断言结果为Series类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为"HL2"
        self.assertEqual(result.name, "HL2")

    # 测试高低收盘价中值函数
    def test_hlc3(self):
        # 调用高低收盘价中值函数，不使用talib
        result = pandas_ta.hlc3(self.high, self.low, self.close, talib=False)
        # 断言结果为Series类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为"HLC3"

        # 尝试使用talib库计算TYPPRICE，并进行结果对比
        try:
            # 期望值通过talib库计算
            expected = tal.TYPPRICE(self.high, self.low, self.close)
            # 断言两个Series相等，忽略名称检查
            pdt.assert_series_equal(result, expected, check_names=False)
        # 如果断言失败，则进行进一步分析
        except AssertionError:
            try:
                # 计算结果和期望值之间的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            # 如果出现异常，则执行错误分析函数
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        # 调用高低收盘价中值函数，使用talib
        result = pandas_ta.hlc3(self.high, self.low, self.close)
        # 断言结果为Series类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为"HLC3"

    # 测试Hull移动平均值函数
    def test_hma(self):
        # 调用Hull移动平均值函数
        result = pandas_ta.hma(self.close)
        # 断言结果为Series类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为"HMA_10"

    # 测试Hull加权移动平均值函数
    def test_hwma(self):
        # 调用Hull加权移动平均值函数
        result = pandas_ta.hwma(self.close)
        # 断言结果为Series类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为"HWMA_0.2_0.1_0.1"
    # 测试 KAMA 指标计算函数
    def test_kama(self):
        # 调用 pandas_ta 库中的 kama 函数计算
        result = pandas_ta.kama(self.close)
        # 断言结果为 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "KAMA_10_2_30"
        self.assertEqual(result.name, "KAMA_10_2_30")

    # 测试 JMA 指标计算函数
    def test_jma(self):
        # 调用 pandas_ta 库中的 jma 函数计算
        result = pandas_ta.jma(self.close)
        # 断言结果为 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "JMA_7_0"
        self.assertEqual(result.name, "JMA_7_0")

    # 测试 Ichimoku 指标计算函数
    def test_ichimoku(self):
        # 调用 pandas_ta 库中的 ichimoku 函数计算
        ichimoku, span = pandas_ta.ichimoku(self.high, self.low, self.close)
        # 断言 ichimoku 结果为 DataFrame 对象
        self.assertIsInstance(ichimoku, DataFrame)
        # 断言 span 结果为 DataFrame 对象
        self.assertIsInstance(span, DataFrame)
        # 断言 ichimoku 结果的名称为 "ICHIMOKU_9_26_52"
        self.assertEqual(ichimoku.name, "ICHIMOKU_9_26_52")
        # 断言 span 结果的名称为 "ICHISPAN_9_26"
        self.assertEqual(span.name, "ICHISPAN_9_26")

    # 测试 LinReg 指标计算函数
    def test_linreg(self):
        # 调用 pandas_ta 库中的 linreg 函数计算，不使用 talib
        result = pandas_ta.linreg(self.close, talib=False)
        # 断言结果为 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "LR_14"
        self.assertEqual(result.name, "LR_14")

        try:
            # 尝试使用 talib 进行结果比较
            expected = tal.LINEARREG(self.close)
            # 使用 pandas 的 assert_series_equal 检查结果
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 尝试进行结果相关性分析
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 处理异常情况
                error_analysis(result, CORRELATION, ex)

        # 再次调用 linreg 函数，使用 talib
        result = pandas_ta.linreg(self.close)
        # 断言结果为 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "LR_14"
        self.assertEqual(result.name, "LR_14")

    # 测试 LinReg Angle 指标计算函数
    def test_linreg_angle(self):
        # 调用 pandas_ta 库中的 linreg 函数计算，包括角度计算，不使用 talib
        result = pandas_ta.linreg(self.close, angle=True, talib=False)
        # 断言结果为 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "LRa_14"
        self.assertEqual(result.name, "LRa_14")

        try:
            # 尝试使用 talib 进行结果比较
            expected = tal.LINEARREG_ANGLE(self.close)
            # 使用 pandas 的 assert_series_equal 检查结果
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 尝试进行结果相关性分析
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 处理异常情况
                error_analysis(result, CORRELATION, ex)

        # 再次调用 linreg 函数，包括角度计算，使用 talib
        result = pandas_ta.linreg(self.close, angle=True)
        # 断言结果为 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "LRa_14"
        self.assertEqual(result.name, "LRa_14")

    # 测试 LinReg Intercept 指标计算函数
    def test_linreg_intercept(self):
        # 调用 pandas_ta 库中的 linreg 函数计算，包括截距计算，不使用 talib
        result = pandas_ta.linreg(self.close, intercept=True, talib=False)
        # 断言结果为 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "LRb_14"
        self.assertEqual(result.name, "LRb_14")

        try:
            # 尝试使用 talib 进行结果比较
            expected = tal.LINEARREG_INTERCEPT(self.close)
            # 使用 pandas 的 assert_series_equal 检查结果
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 尝试进行结果相关性分析
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 处理异常情况
                error_analysis(result, CORRELATION, ex)

        # 再次调用 linreg 函数，包括截距计算，使用 talib
        result = pandas_ta.linreg(self.close, intercept=True)
        # 断言结果为 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "LRb_14"
        self.assertEqual(result.name, "LRb_14")
    # 测试线性回归指标的随机性（r）计算是否正确
    def test_linreg_r(self):
        # 计算线性回归指标的随机性（r）
        result = pandas_ta.linreg(self.close, r=True)
        # 确保返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 确保返回结果的名称为 "LRr_14"
        self.assertEqual(result.name, "LRr_14")

    # 测试线性回归指标的斜率（slope）计算是否正确
    def test_linreg_slope(self):
        # 计算线性回归指标的斜率
        result = pandas_ta.linreg(self.close, slope=True, talib=False)
        # 确保返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 确保返回结果的名称为 "LRm_14"
        self.assertEqual(result.name, "LRm_14")

        try:
            # 尝试使用 talib 计算线性回归指标的斜率
            expected = tal.LINEARREG_SLOPE(self.close)
            # 对比结果与预期结果是否相等，不检查名称
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 计算结果与预期结果之间的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 确保相关性大于 CORRELATION_THRESHOLD
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 在出现异常时执行错误分析
                error_analysis(result, CORRELATION, ex)

        # 使用默认设置计算线性回归指标的斜率
        result = pandas_ta.linreg(self.close, slope=True)
        # 确保返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 确保返回结果的名称为 "LRm_14"
        self.assertEqual(result.name, "LRm_14")

    # 测试移动平均（ma）指标计算是否正确
    def test_ma(self):
        # 计算简单移动平均（SMA）指标
        result = pandas_ta.ma()
        # 确保返回结果是一个列表
        self.assertIsInstance(result, list)
        # 确保返回结果长度大于 0
        self.assertGreater(len(result), 0)

        # 计算指定类型的移动平均（EMA）指标
        result = pandas_ta.ma("ema", self.close)
        # 确保返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 确保返回结果的名称为 "EMA_10"
        self.assertEqual(result.name, "EMA_10")

        # 计算指定类型和长度的移动平均（FWMA）指标
        result = pandas_ta.ma("fwma", self.close, length=15)
        # 确保返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 确保返回结果的名称为 "FWMA_15"
        self.assertEqual(result.name, "FWMA_15")

    # 测试 MACD 指标计算是否正确
    def test_mcgd(self):
        # 计算 MACD 指标
        result = pandas_ta.mcgd(self.close)
        # 确保返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 确保返回结果的名称为 "MCGD_10"

    # 测试中点（midpoint）指标计算是否正确
    def test_midpoint(self):
        # 计算中点（midpoint）指标
        result = pandas_ta.midpoint(self.close, talib=False)
        # 确保返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 确保返回结果的名称为 "MIDPOINT_2"

        try:
            # 尝试使用 talib 计算中点（midpoint）指标
            expected = tal.MIDPOINT(self.close, 2)
            # 对比结果与预期结果是否相等，不检查名称
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 计算结果与预期结果之间的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 确保相关性大于 CORRELATION_THRESHOLD
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 在出现异常时执行错误分析
                error_analysis(result, CORRELATION, ex)

        # 使用默认设置计算中点（midpoint）指标
        result = pandas_ta.midpoint(self.close)
        # 确保返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 确保返回结果的名称为 "MIDPOINT_2"
        self.assertEqual(result.name, "MIDPOINT_2")
    # 测试中位数价格指标函数
    def test_midprice(self):
        # 调用 midprice 函数计算中位数价格
        result = pandas_ta.midprice(self.high, self.low, talib=False)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "MIDPRICE_2"
        self.assertEqual(result.name, "MIDPRICE_2")

        try:
            # 使用 Talib 库计算期望结果
            expected = tal.MIDPRICE(self.high, self.low, 2)
            # 断言结果与期望结果相等
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 分析结果与期望结果之间的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于 CORRELATION_THRESHOLD
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 在出现异常时进行错误分析
                error_analysis(result, CORRELATION, ex)

        # 未指定时间周期调用 midprice 函数
        result = pandas_ta.midprice(self.high, self.low)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "MIDPRICE_2"
        self.assertEqual(result.name, "MIDPRICE_2")

    # 测试 OHLC4 函数
    def test_ohlc4(self):
        # 调用 ohlc4 函数计算 OHLC4
        result = pandas_ta.ohlc4(self.open, self.high, self.low, self.close)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "OHLC4"
        self.assertEqual(result.name, "OHLC4")

    # 测试 PWMA 函数
    def test_pwma(self):
        # 调用 pwma 函数计算 PWMA
        result = pandas_ta.pwma(self.close)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "PWMA_10"
        self.assertEqual(result.name, "PWMA_10")

    # 测试 RMA 函数
    def test_rma(self):
        # 调用 rma 函数计算 RMA
        result = pandas_ta.rma(self.close)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "RMA_10"
        self.assertEqual(result.name, "RMA_10")

    # 测试 SINWMA 函数
    def test_sinwma(self):
        # 调用 sinwma 函数计算 SINWMA
        result = pandas_ta.sinwma(self.close)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "SINWMA_14"
        self.assertEqual(result.name, "SINWMA_14")

    # 测试 SMA 函数
    def test_sma(self):
        # 不使用 Talib 库调用 sma 函数计算 SMA
        result = pandas_ta.sma(self.close, talib=False)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "SMA_10"
        self.assertEqual(result.name, "SMA_10")

        try:
            # 使用 Talib 库计算期望结果
            expected = tal.SMA(self.close, 10)
            # 断言结果与期望结果相等
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 分析结果与期望结果之间的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于 CORRELATION_THRESHOLD
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 在出现异常时进行错误分析
                error_analysis(result, CORRELATION, ex)

        # 未指定时间周期调用 sma 函数
        result = pandas_ta.sma(self.close)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "SMA_10"
        self.assertEqual(result.name, "SMA_10")

    # 测试 SSF 函数
    def test_ssf(self):
        # 调用 ssf 函数计算 SSF
        result = pandas_ta.ssf(self.close, poles=2)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "SSF_10_2"
        self.assertEqual(result.name, "SSF_10_2")

        # 再次调用 ssf 函数计算 SSF
        result = pandas_ta.ssf(self.close, poles=3)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "SSF_10_3"
        self.assertEqual(result.name, "SSF_10_3")

    # 测试 SWMA 函数
    def test_swma(self):
        # 调用 swma 函数计算 SWMA
        result = pandas_ta.swma(self.close)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "SWMA_10"
        self.assertEqual(result.name, "SWMA_10")

    # 测试 SuperTrend 函数
    def test_supertrend(self):
        # 调用 supertrend 函数计算 SuperTrend
        result = pandas_ta.supertrend(self.high, self.low, self.close)
        # 断言返回结果是一个 DataFrame 对象
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "SUPERT_7_3.0"
        self.assertEqual(result.name, "SUPERT_7_3.0")
    # 测试 T3 指标计算函数
    def test_t3(self):
        # 使用 pandas_ta 库计算 T3 指标，关闭使用 talib
        result = pandas_ta.t3(self.close, talib=False)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "T3_10_0.7"
        self.assertEqual(result.name, "T3_10_0.7")

        # 尝试使用 talib 计算 T3 指标，并与预期结果比较
        try:
            expected = tal.T3(self.close, 10)
            # 使用 pandas.testing.assert_series_equal 函数比较两个 Series，关闭名称检查
            pdt.assert_series_equal(result, expected, check_names=False)
        # 如果断言失败
        except AssertionError:
            # 尝试进行误差分析并检查相关性
            try:
                # 调用 pandas_ta.utils.df_error_analysis 函数分析误差
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于指定阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            # 如果发生异常
            except Exception as ex:
                # 调用 error_analysis 函数处理异常
                error_analysis(result, CORRELATION, ex)

        # 重新使用 pandas_ta 库计算 T3 指标
        result = pandas_ta.t3(self.close)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "T3_10_0.7"
        self.assertEqual(result.name, "T3_10_0.7")

    # 测试 TEMA 指标计算函数
    def test_tema(self):
        # 使用 pandas_ta 库计算 TEMA 指标，关闭使用 talib
        result = pandas_ta.tema(self.close, talib=False)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "TEMA_10"
        self.assertEqual(result.name, "TEMA_10")

        # 尝试使用 talib 计算 TEMA 指标，并与预期结果比较
        try:
            expected = tal.TEMA(self.close, 10)
            # 使用 pandas.testing.assert_series_equal 函数比较两个 Series，关闭名称检查
            pdt.assert_series_equal(result, expected, check_names=False)
        # 如果断言失败
        except AssertionError:
            # 尝试进行误差分析并检查相关性
            try:
                # 调用 pandas_ta.utils.df_error_analysis 函数分析误差
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于指定阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            # 如果发生异常
            except Exception as ex:
                # 调用 error_analysis 函数处理异常
                error_analysis(result, CORRELATION, ex)

        # 重新使用 pandas_ta 库计算 TEMA 指标
        result = pandas_ta.tema(self.close)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "TEMA_10"
        self.assertEqual(result.name, "TEMA_10")

    # 测试 TRIMA 指标计算函数
    def test_trima(self):
        # 使用 pandas_ta 库计算 TRIMA 指标，关闭使用 talib
        result = pandas_ta.trima(self.close, talib=False)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "TRIMA_10"
        self.assertEqual(result.name, "TRIMA_10")

        # 尝试使用 talib 计算 TRIMA 指标，并与预期结果比较
        try:
            expected = tal.TRIMA(self.close, 10)
            # 使用 pandas.testing.assert_series_equal 函数比较两个 Series，关闭名称检查
            pdt.assert_series_equal(result, expected, check_names=False)
        # 如果断言失败
        except AssertionError:
            # 尝试进行误差分析并检查相关性
            try:
                # 调用 pandas_ta.utils.df_error_analysis 函数分析误差
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于指定阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            # 如果发生异常
            except Exception as ex:
                # 调用 error_analysis 函数处理异常
                error_analysis(result, CORRELATION, ex)

        # 重新使用 pandas_ta 库计算 TRIMA 指标
        result = pandas_ta.trima(self.close)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "TRIMA_10"
        self.assertEqual(result.name, "TRIMA_10")

    # 测试 VIDYA 指标计算函数
    def test_vidya(self):
        # 使用 pandas_ta 库计算 VIDYA 指标
        result = pandas_ta.vidya(self.close)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "VIDYA_14"
        self.assertEqual(result.name, "VIDYA_14")

    # 测试 VWAP 指标计算函数
    def test_vwap(self):
        # 使用 pandas_ta 库计算 VWAP 指标
        result = pandas_ta.vwap(self.high, self.low, self.close, self.volume)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "VWAP_D"
        self.assertEqual(result.name, "VWAP_D")

    # 测试 VWMA 指标计算函数
    def test_vwma(self):
        # 使用 pandas_ta 库计算 VWMA 指标
        result = pandas_ta.vwma(self.close, self.volume)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "VWMA_10"
        self.assertEqual(result.name, "VWMA_10")
    # 测试 Weighted Close Price (WCP) 函数
    def test_wcp(self):
        # 使用 pandas_ta 库中的 wcp 函数计算结果
        result = pandas_ta.wcp(self.high, self.low, self.close, talib=False)
        # 断言结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "WCP"
        self.assertEqual(result.name, "WCP")

        # 尝试使用 Talib 库中的 WCLPRICE 函数计算期望结果
        try:
            expected = tal.WCLPRICE(self.high, self.low, self.close)
            # 使用 pandas 测试工具 pdt 断言两个 Series 相等
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            # 如果断言失败，则进行误差分析
            try:
                # 计算结果与期望值的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果出现异常，则进行错误分析
                error_analysis(result, CORRELATION, ex)

        # 再次使用 pandas_ta 库中的 wcp 函数计算结果
        result = pandas_ta.wcp(self.high, self.low, self.close)
        # 断言结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "WCP"
        self.assertEqual(result.name, "WCP")

    # 测试 Weighted Moving Average (WMA) 函数
    def test_wma(self):
        # 使用 pandas_ta 库中的 wma 函数计算结果
        result = pandas_ta.wma(self.close, talib=False)
        # 断言结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "WMA_10"
        self.assertEqual(result.name, "WMA_10")

        # 尝试使用 Talib 库中的 WMA 函数计算期望结果
        try:
            expected = tal.WMA(self.close, 10)
            # 使用 pandas 测试工具 pdt 断言两个 Series 相等
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            # 如果断言失败，则进行误差分析
            try:
                # 计算结果与期望值的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果出现异常，则进行错误分析
                error_analysis(result, CORRELATION, ex)

        # 再次使用 pandas_ta 库中的 wma 函数计算结果
        result = pandas_ta.wma(self.close)
        # 断言结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "WMA_10"
        self.assertEqual(result.name, "WMA_10")

    # 测试 Zero-Lag Exponential Moving Average (ZLEMA) 函数
    def test_zlma(self):
        # 使用 pandas_ta 库中的 zlma 函数计算结果
        result = pandas_ta.zlma(self.close)
        # 断言结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "ZL_EMA_10"
        self.assertEqual(result.name, "ZL_EMA_10")
```