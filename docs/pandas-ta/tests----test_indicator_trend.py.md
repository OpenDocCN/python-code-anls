# `.\pandas-ta\tests\test_indicator_trend.py`

```
# 从相对路径的config模块中导入error_analysis、sample_data、CORRELATION、CORRELATION_THRESHOLD和VERBOSE变量
from .config import error_analysis, sample_data, CORRELATION, CORRELATION_THRESHOLD, VERBOSE
# 从context模块中导入pandas_ta对象
from .context import pandas_ta

# 导入TestCase类和skip函数从unittest模块中
from unittest import TestCase, skip
# 导入pandas测试模块，并用pdt别名引用
import pandas.testing as pdt
# 从pandas模块中导入DataFrame和Series类
from pandas import DataFrame, Series

# 导入talib模块，并用tal别名引用
import talib as tal

# 定义TestTrend类，继承自TestCase类
class TestTrend(TestCase):
    # 在类方法setUpClass中设置测试所需的数据
    @classmethod
    def setUpClass(cls):
        # 将sample_data赋值给cls.data，然后将列名转换为小写
        cls.data = sample_data
        cls.data.columns = cls.data.columns.str.lower()
        # 将open列、high列、low列、close列分别赋值给cls.open、cls.high、cls.low、cls.close
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        # 如果数据中存在volume列，则将其赋值给cls.volume
        if "volume" in cls.data.columns:
            cls.volume = cls.data["volume"]

    # 在类方法tearDownClass中清理测试所需的数据
    @classmethod
    def tearDownClass(cls):
        # 删除cls.open、cls.high、cls.low、cls.close四个变量的引用
        del cls.open
        del cls.high
        del cls.low
        del cls.close
        # 如果cls中存在volume属性，则删除该属性
        if hasattr(cls, "volume"):
            del cls.volume
        # 删除cls.data变量的引用
        del cls.data

    # 在setUp方法中初始化测试环境
    def setUp(self): pass
    # 在tearDown方法中清理测试环境
    def tearDown(self): pass

    # 定义测试ADX指标的方法
    def test_adx(self):
        # 调用pandas_ta.adx函数计算ADX指标，指定talib参数为False
        result = pandas_ta.adx(self.high, self.low, self.close, talib=False)
        # 断言result对象为DataFrame类型
        self.assertIsInstance(result, DataFrame)
        # 断言result对象的name属性为"ADX_14"
        self.assertEqual(result.name, "ADX_14")

        # 尝试使用tal.ADX函数计算预期结果，并断言result的第一列与预期结果相等
        try:
            expected = tal.ADX(self.high, self.low, self.close)
            pdt.assert_series_equal(result.iloc[:, 0], expected)
        # 如果断言失败，则进行异常处理
        except AssertionError:
            # 尝试计算result与预期结果的相关性，并与CORRELATION_THRESHOLD进行比较
            try:
                corr = pandas_ta.utils.df_error_analysis(result.iloc[:, 0], expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            # 如果相关性计算出现异常，则调用error_analysis函数记录异常信息
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        # 再次调用pandas_ta.adx函数计算ADX指标
        result = pandas_ta.adx(self.high, self.low, self.close)
        # 断言result对象为DataFrame类型
        self.assertIsInstance(result, DataFrame)
        # 断言result对象的name属性为"ADX_14"
        self.assertEqual(result.name, "ADX_14")

    # 定义测试AMAT指标的方法
    def test_amat(self):
        # 调用pandas_ta.amat函数计算AMAT指标
        result = pandas_ta.amat(self.close)
        # 断言result对象为DataFrame类型
        self.assertIsInstance(result, DataFrame)
        # 断言result对象的name属性为"AMATe_8_21_2"
        self.assertEqual(result.name, "AMATe_8_21_2")
    # 测试 Aroon 指标计算函数
    def test_aroon(self):
        # 调用 pandas_ta 库中的 aroon 函数计算 Aroon 指标
        result = pandas_ta.aroon(self.high, self.low, talib=False)
        # 断言返回结果为 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言返回结果的名称为 "AROON_14"

        # 尝试使用 talib 库计算 Aroon 指标
        try:
            expected = tal.AROON(self.high, self.low)
            expecteddf = DataFrame({"AROOND_14": expected[0], "AROONU_14": expected[1]})
            # 比较计算结果与预期结果是否一致
            pdt.assert_frame_equal(result, expecteddf)
        except AssertionError:
            # 如果计算结果与预期结果不一致，进行错误分析
            try:
                aroond_corr = pandas_ta.utils.df_error_analysis(result.iloc[:, 0], expecteddf.iloc[:, 0], col=CORRELATION)
                self.assertGreater(aroond_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果出现异常，进行错误分析
                error_analysis(result.iloc[:, 0], CORRELATION, ex)

            try:
                aroonu_corr = pandas_ta.utils.df_error_analysis(result.iloc[:, 1], expecteddf.iloc[:, 1], col=CORRELATION)
                self.assertGreater(aroonu_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果出现异常，进行错误分析
                error_analysis(result.iloc[:, 1], CORRELATION, ex, newline=False)

        # 重新计算 Aroon 指标
        result = pandas_ta.aroon(self.high, self.low)
        # 断言返回结果为 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言返回结果的名称为 "AROON_14"

    # 测试 Aroon Oscillator 指标计算函数
    def test_aroon_osc(self):
        # 计算 Aroon Oscillator 指标
        result = pandas_ta.aroon(self.high, self.low)
        # 断言返回结果为 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言返回结果的名称为 "AROON_14"

        try:
            expected = tal.AROONOSC(self.high, self.low)
            # 比较计算结果与预期结果是否一致
            pdt.assert_series_equal(result.iloc[:, 2], expected)
        except AssertionError:
            try:
                aroond_corr = pandas_ta.utils.df_error_analysis(result.iloc[:,2], expected,col=CORRELATION)
                self.assertGreater(aroond_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果出现异常，进行错误分析
                error_analysis(result.iloc[:, 0], CORRELATION, ex)

    # 测试 Chande Kroll Stop 指标计算函数
    def test_chop(self):
        # 计算 Chande Kroll Stop 指标
        result = pandas_ta.chop(self.high, self.low, self.close, ln=False)
        # 断言返回结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言返回结果的名称为 "CHOP_14_1_100"

        result = pandas_ta.chop(self.high, self.low, self.close, ln=True)
        # 断言返回结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言返回结果的名称为 "CHOPln_14_1_100"

    # 测试 Chande Kroll Stop Points 指标计算函数
    def test_cksp(self):
        # 计算 Chande Kroll Stop Points 指标
        result = pandas_ta.cksp(self.high, self.low, self.close, tvmode=False)
        # 断言返回结果为 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言返回结果的名称为 "CKSP_10_3_20"

    # 测试 Chande Kroll Stop Points 指标计算函数（带 True Volume 模式）
    def test_cksp_tv(self):
        # 计算 Chande Kroll Stop Points 指标（带 True Volume 模式）
        result = pandas_ta.cksp(self.high, self.low, self.close, tvmode=True)
        # 断言返回结果为 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言返回结果的名称为 "CKSP_10_1_9"
    # 测试指数加权衰减移动平均函数
    def test_decay(self):
        # 使用默认参数计算指数加权衰减移动平均
        result = pandas_ta.decay(self.close)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "LDECAY_5"
        self.assertEqual(result.name, "LDECAY_5")

        # 使用指数加权衰减移动平均模式参数计算指数加权衰减移动平均
        result = pandas_ta.decay(self.close, mode="exp")
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "EXPDECAY_5"
        self.assertEqual(result.name, "EXPDECAY_5")

    # 测试判断价格是否递减函数
    def test_decreasing(self):
        # 使用默认参数判断价格是否递减
        result = pandas_ta.decreasing(self.close)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "DEC_1"
        self.assertEqual(result.name, "DEC_1")

        # 使用指定长度和严格模式参数判断价格是否递减
        result = pandas_ta.decreasing(self.close, length=3, strict=True)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "SDEC_3"
        self.assertEqual(result.name, "SDEC_3")

    # 测试双重指数移动平均离差指标（DPO）函数
    def test_dpo(self):
        # 计算双重指数移动平均离差指标（DPO）
        result = pandas_ta.dpo(self.close)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "DPO_20"
        self.assertEqual(result.name, "DPO_20")

    # 测试判断价格是否递增函数
    def test_increasing(self):
        # 使用默认参数判断价格是否递增
        result = pandas_ta.increasing(self.close)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "INC_1"
        self.assertEqual(result.name, "INC_1")

        # 使用指定长度和严格模式参数判断价格是否递增
        result = pandas_ta.increasing(self.close, length=3, strict=True)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "SINC_3"
        self.assertEqual(result.name, "SINC_3")

    # 测试长期趋势函数
    def test_long_run(self):
        # 计算长期趋势
        result = pandas_ta.long_run(self.close, self.open)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "LR_2"
        self.assertEqual(result.name, "LR_2")

    # 测试抛物线 SAR 函数
    def test_psar(self):
        # 计算抛物线 SAR
        result = pandas_ta.psar(self.high, self.low)
        # 断言返回结果是一个 DataFrame 对象
        self.assertIsInstance(result, DataFrame)
        # 断言结果 DataFrame 的名称为 "PSAR_0.02_0.2"
        self.assertEqual(result.name, "PSAR_0.02_0.2")

        # 合并长期和短期 SAR 值为一个 SAR 值
        psar = result[result.columns[:2]].fillna(0)
        psar = psar[psar.columns[0]] + psar[psar.columns[1]]
        psar.name = result.name

        try:
            # 尝试与 talib 提供的 SAR 值进行比较
            expected = tal.SAR(self.high, self.low)
            # 断言两个 Series 对象相等
            pdt.assert_series_equal(psar, expected)
        except AssertionError:
            try:
                # 尝试进行相关性分析
                psar_corr = pandas_ta.utils.df_error_analysis(psar, expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(psar_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 发生异常时输出错误分析
                error_analysis(psar, CORRELATION, ex)

    # 测试 QStick 函数
    def test_qstick(self):
        # 计算 QStick 值
        result = pandas_ta.qstick(self.open, self.close)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "QS_10"
        self.assertEqual(result.name, "QS_10")

    # 测试短期趋势函数
    def test_short_run(self):
        # 计算短期趋势
        result = pandas_ta.short_run(self.close, self.open)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "SR_2"
        self.assertEqual(result.name, "SR_2")

    # 测试 TTM 趋势函数
    def test_ttm_trend(self):
        # 计算 TTM 趋势
        result = pandas_ta.ttm_trend(self.high, self.low, self.close)
        # 断言返回结果是一个 DataFrame 对象
        self.assertIsInstance(result, DataFrame)
        # 断言结果 DataFrame 的名称为 "TTMTREND_6"
        self.assertEqual(result.name, "TTMTREND_6")

    # 测试垂直水平通道函数
    def test_vhf(self):
        # 计算垂直水平通道值
        result = pandas_ta.vhf(self.close)
        # 断言返回结果是一个 Series 对象
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "VHF_28"
        self.assertEqual(result.name, "VHF_28")
    # 测试 Vortex 指标函数
    def test_vortex(self):
        # 调用 Vortex 指标函数，传入高、低、收盘价数据，获得结果
        result = pandas_ta.vortex(self.high, self.low, self.close)
        # 断言结果是 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "VTX_14"
        self.assertEqual(result.name, "VTX_14")
```