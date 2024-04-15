# `.\pandas-ta\tests\test_indicator_momentum.py`

```py
# 从config模块中导入error_analysis, sample_data, CORRELATION, CORRELATION_THRESHOLD, VERBOSE变量
# 从context模块中导入pandas_ta模块
from .config import error_analysis, sample_data, CORRELATION, CORRELATION_THRESHOLD, VERBOSE
from .context import pandas_ta

# 从unittest模块中导入TestCase类和skip函数
# 从pandas.testing模块中导入pdt别名
# 从pandas模块中导入DataFrame, Series类
# 导入talib模块并使用tal别名
from unittest import TestCase, skip
import pandas.testing as pdt
from pandas import DataFrame, Series

import talib as tal

# 定义TestMomentum类，继承自TestCase类
class TestMomentum(TestCase):
    # 类方法setUpClass，用于设置测试类的初始状态
    @classmethod
    def setUpClass(cls):
        # 初始化sample_data数据
        cls.data = sample_data
        # 将数据列名转换为小写
        cls.data.columns = cls.data.columns.str.lower()
        # 初始化open, high, low, close数据列
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        # 如果数据中包含"volume"列，则初始化volume数据列
        if "volume" in cls.data.columns:
            cls.volume = cls.data["volume"]

    # 类方法tearDownClass，用于清理测试类的状态
    @classmethod
    def tearDownClass(cls):
        # 删除open, high, low, close数据列
        del cls.open
        del cls.high
        del cls.low
        del cls.close
        # 如果存在volume数据列，则删除volume数据列
        if hasattr(cls, "volume"):
            del cls.volume
        # 删除数据
        del cls.data

    # setUp方法，用于设置每个测试方法的初始状态
    def setUp(self): pass
    # tearDown方法，用于清理每个测试方法的状态
    def tearDown(self): pass

    # 测试方法test_datetime_ordered
    def test_datetime_ordered(self):
        # 测试datetime64索引是否有序
        result = self.data.ta.datetime_ordered
        self.assertTrue(result)

        # 测试索引是否无序
        original = self.data.copy()
        reversal = original.ta.reverse
        result = reversal.ta.datetime_ordered
        self.assertFalse(result)

        # 测试非datetime64索引
        original = self.data.copy()
        original.reset_index(inplace=True)
        result = original.ta.datetime_ordered
        self.assertFalse(result)

    # 测试方法test_reverse
    def test_reverse(self):
        original = self.data.copy()
        result = original.ta.reverse

        # 检查第一个和最后一个时间是否被颠倒
        self.assertEqual(result.index[-1], original.index[0])
        self.assertEqual(result.index[0], original.index[-1])

    # 测试方法test_ao
    def test_ao(self):
        result = pandas_ta.ao(self.high, self.low)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "AO_5_34")

    # 测试方法test_apo
    def test_apo(self):
        result = pandas_ta.apo(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "APO_12_26")

        try:
            expected = tal.APO(self.close)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = pandas_ta.apo(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "APO_12_26")

    # 测试方法test_bias
    def test_bias(self):
        result = pandas_ta.bias(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "BIAS_SMA_26")
    # 测试 Balance of Power 指标计算函数
    def test_bop(self):
        # 使用 pandas_ta 库计算 Balance of Power 指标，不使用 TA-Lib
        result = pandas_ta.bop(self.open, self.high, self.low, self.close, talib=False)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "BOP"
        self.assertEqual(result.name, "BOP")

        try:
            # 使用 TA-Lib 计算 Balance of Power 指标
            expected = tal.BOP(self.open, self.high, self.low, self.close)
            # 断言计算结果与预期结果相等，忽略名称检查
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 分析结果数据框的误差
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 执行错误分析
                error_analysis(result, CORRELATION, ex)

        # 使用 pandas_ta 库计算 Balance of Power 指标，使用 TA-Lib
        result = pandas_ta.bop(self.open, self.high, self.low, self.close)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "BOP"
        self.assertEqual(result.name, "BOP")

    # 测试 BRAR 指标计算函数
    def test_brar(self):
        # 使用 pandas_ta 库计算 BRAR 指标
        result = pandas_ta.brar(self.open, self.high, self.low, self.close)
        # 断言结果为 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果 DataFrame 的名称为 "BRAR_26"
        self.assertEqual(result.name, "BRAR_26")

    # 测试 CCI 指标计算函数
    def test_cci(self):
        # 使用 pandas_ta 库计算 CCI 指标，不使用 TA-Lib
        result = pandas_ta.cci(self.high, self.low, self.close, talib=False)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "CCI_14_0.015"
        self.assertEqual(result.name, "CCI_14_0.015")

        try:
            # 使用 TA-Lib 计算 CCI 指标
            expected = tal.CCI(self.high, self.low, self.close)
            # 断言计算结果与预期结果相等，忽略名称检查
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 分析结果数据框的误差
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 执行错误分析
                error_analysis(result, CORRELATION, ex)

        # 使用 pandas_ta 库计算 CCI 指标，使用 TA-Lib
        result = pandas_ta.cci(self.high, self.low, self.close)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "CCI_14_0.015"
        self.assertEqual(result.name, "CCI_14_0.015")

    # 测试 CFO 指标计算函数
    def test_cfo(self):
        # 使用 pandas_ta 库计算 CFO 指标
        result = pandas_ta.cfo(self.close)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "CFO_9"
        self.assertEqual(result.name, "CFO_9")

    # 测试 CG 指标计算函数
    def test_cg(self):
        # 使用 pandas_ta 库计算 CG 指标
        result = pandas_ta.cg(self.close)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "CG_10"
        self.assertEqual(result.name, "CG_10")

    # 测试 CMO 指标计算函数
    def test_cmo(self):
        # 使用 pandas_ta 库计算 CMO 指标
        result = pandas_ta.cmo(self.close)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "CMO_14"
        self.assertEqual(result.name, "CMO_14")

        try:
            # 使用 TA-Lib 计算 CMO 指标
            expected = tal.CMO(self.close)
            # 断言计算结果与预期结果相等，忽略名称检查
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 分析结果数据框的误差
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 执行错误分析
                error_analysis(result, CORRELATION, ex)

        # 使用 pandas_ta 库计算 CMO 指标，不使用 TA-Lib
        result = pandas_ta.cmo(self.close, talib=False)
        # 断言结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "CMO_14"
        self.assertEqual(result.name, "CMO_14")
    # 测试 Coppock 指标计算函数
    def test_coppock(self):
        # 调用 coppock 函数计算结果
        result = pandas_ta.coppock(self.close)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "COPC_11_14_10"

    # 测试 CTI 指标计算函数
    def test_cti(self):
        # 调用 cti 函数计算结果
        result = pandas_ta.cti(self.close)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "CTI_12"

    # 测试 ER 指标计算函数
    def test_er(self):
        # 调用 er 函数计算结果
        result = pandas_ta.er(self.close)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "ER_10"

    # 测试 DM 指标计算函数
    def test_dm(self):
        # 调用 dm 函数计算结果
        result = pandas_ta.dm(self.high, self.low, talib=False)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "DM_14"

        try:
            # 使用 tal.PLUS_DM 和 tal.MINUS_DM 计算期望结果
            expected_pos = tal.PLUS_DM(self.high, self.low)
            expected_neg = tal.MINUS_DM(self.high, self.low)
            expecteddf = DataFrame({"DMP_14": expected_pos, "DMN_14": expected_neg})
            # 比较结果和期望结果
            pdt.assert_frame_equal(result, expecteddf)
        except AssertionError:
            try:
                # 分析结果和期望结果的相关性
                dmp = pandas_ta.utils.df_error_analysis(result.iloc[:,0], expecteddf.iloc[:,0], col=CORRELATION)
                self.assertGreater(dmp, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 处理异常情况
                error_analysis(result, CORRELATION, ex)

            try:
                # 分析结果和期望结果的相关性
                dmn = pandas_ta.utils.df_error_analysis(result.iloc[:,1], expecteddf.iloc[:,1], col=CORRELATION)
                self.assertGreater(dmn, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 处理异常情况
                error_analysis(result, CORRELATION, ex)

        # 重新调用 dm 函数计算结果
        result = pandas_ta.dm(self.high, self.low)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "DM_14"

    # 测试 ERI 指标计算函数
    def test_eri(self):
        # 调用 eri 函数计算结果
        result = pandas_ta.eri(self.high, self.low, self.close)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "ERI_13"

    # 测试 Fisher 指标计算函数
    def test_fisher(self):
        # 调用 fisher 函数计算结果
        result = pandas_ta.fisher(self.high, self.low)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "FISHERT_9_1"

    # 测试 Inertia 指标计算函数
    def test_inertia(self):
        # 调用 inertia 函数计算结果
        result = pandas_ta.inertia(self.close)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "INERTIA_20_14"

        # 调用 inertia 函数计算结果，使用高、低价数据，开启 refined 参数
        result = pandas_ta.inertia(self.close, self.high, self.low, refined=True)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断���结果的名称为 "INERTIAr_20_14"

        # 调用 inertia 函数计算结果，使用高、低价数据，开启 thirds 参数
        result = pandas_ta.inertia(self.close, self.high, self.low, thirds=True)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "INERTIAt_20_14"

    # 测试 KDJ 指标计算函数
    def test_kdj(self):
        # 调用 kdj 函数计算结果
        result = pandas_ta.kdj(self.high, self.low, self.close)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "KDJ_9_3"

    # 测试 KST 指标计算函数
    def test_kst(self):
        # 调用 kst 函数计算结果
        result = pandas_ta.kst(self.close)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "KST_10_15_20_30_10_10_10_15_9"
    # 测试 MACD 指标计算函数
    def test_macd(self):
        # 使用 pandas_ta 库计算 MACD 指标
        result = pandas_ta.macd(self.close, talib=False)
        # 断言结果是 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果 DataFrame 的名称是 "MACD_12_26_9"
        self.assertEqual(result.name, "MACD_12_26_9")

        # 尝试使用 talib 库计算 MACD 指标，并与 pandas_ta 的结果比较
        try:
            # 使用 talib 库计算 MACD 指标
            expected = tal.MACD(self.close)
            # 将 talib 计算的 MACD 结果转换为 DataFrame
            expecteddf = DataFrame({"MACD_12_26_9": expected[0], "MACDh_12_26_9": expected[2], "MACDs_12_26_9": expected[1]})
            # 断言 pandas_ta 计算的结果与 talib 计算的结果相等
            pdt.assert_frame_equal(result, expecteddf)
        except AssertionError:
            # 如果结果不相等，则进行进一步分析
            try:
                # 计算 MACD 指标数据的相关性
                macd_corr = pandas_ta.utils.df_error_analysis(result.iloc[:, 0], expecteddf.iloc[:, 0], col=CORRELATION)
                # 断言相关性大于预设阈值
                self.assertGreater(macd_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果出现异常，则进行错误分析
                error_analysis(result.iloc[:, 0], CORRELATION, ex)

            try:
                # 分析历史数据的相关性
                history_corr = pandas_ta.utils.df_error_analysis(result.iloc[:, 1], expecteddf.iloc[:, 1], col=CORRELATION)
                # 断言相关性大于预设阈值
                self.assertGreater(history_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果出现异常，则进行错误分析
                error_analysis(result.iloc[:, 1], CORRELATION, ex, newline=False)

            try:
                # 分析信号数据的相关性
                signal_corr = pandas_ta.utils.df_error_analysis(result.iloc[:, 2], expecteddf.iloc[:, 2], col=CORRELATION)
                # 断言相关性大于预设阈值
                self.assertGreater(signal_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果出现异常，则进行错误分析
                error_analysis(result.iloc[:, 2], CORRELATION, ex, newline=False)

        # 重新使用 pandas_ta 库计算 MACD 指标
        result = pandas_ta.macd(self.close)
        # 断言结果是 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果 DataFrame 的名称是 "MACD_12_26_9"
        self.assertEqual(result.name, "MACD_12_26_9")

    # 测试 MACD 指标计算函数（带 asmode 参数）
    def test_macdas(self):
        # 使用 pandas_ta 库计算 MACD 指标（带 asmode 参数）
        result = pandas_ta.macd(self.close, asmode=True)
        # 断言结果是 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果 DataFrame 的名称是 "MACDAS_12_26_9"
        self.assertEqual(result.name, "MACDAS_12_26_9")

    # 测试动量指标计算函数
    def test_mom(self):
        # 使用 pandas_ta 库计算动量指标
        result = pandas_ta.mom(self.close, talib=False)
        # 断言结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称是 "MOM_10"
        self.assertEqual(result.name, "MOM_10")

        # 尝试使用 talib 库计算动量指标，并与 pandas_ta 的结果比较
        try:
            # 使用 talib 库计算动量指标
            expected = tal.MOM(self.close)
            # 断言 pandas_ta 计算的结果与 talib 计算的结果相等
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            # 如果结果不相等，则进行进一步分析
            try:
                # 计算数据的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于预设阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果出现异常，则进行错误分析
                error_analysis(result, CORRELATION, ex)

        # 重新使用 pandas_ta 库计算动量指标
        result = pandas_ta.mom(self.close)
        # 断言结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称是 "MOM_10"
        self.assertEqual(result.name, "MOM_10")

    # 测试价格振荡器指标计算函数
    def test_pgo(self):
        # 使用 pandas_ta 库计算价格振荡器指标
        result = pandas_ta.pgo(self.high, self.low, self.close)
        # 断言结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称是 "PGO_14"
        self.assertEqual(result.name, "PGO_14")
    # 测试基于价格的振荡器（Price Percentage Oscillator，PPO），设置 talib=False 表示不使用 talib
    def test_ppo(self):
        # 调用 pandas_ta 库中的 PPO 函数，计算 PPO 指标
        result = pandas_ta.ppo(self.close, talib=False)
        # 断言返回结果是 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果 DataFrame 的名称为 "PPO_12_26_9"
        self.assertEqual(result.name, "PPO_12_26_9")

        try:
            # 尝试使用 talib 计算 PPO 指标
            expected = tal.PPO(self.close)
            # 对比结果与 talib 计算结果
            pdt.assert_series_equal(result["PPO_12_26_9"], expected, check_names=False)
        except AssertionError:
            try:
                # 若对比失败，则进行误差分析
                corr = pandas_ta.utils.df_error_analysis(result["PPO_12_26_9"], expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 若出现异常，则输出错误分析
                error_analysis(result["PPO_12_26_9"], CORRELATION, ex)

        # 重新计算 PPO 指标（使用 pandas_ta 默认设置）
        result = pandas_ta.ppo(self.close)
        # 断言返回结果是 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果 DataFrame 的名称为 "PPO_12_26_9"
        self.assertEqual(result.name, "PPO_12_26_9")

    # 测试趋势状态线（Price Speed and Length，PSL）指标
    def test_psl(self):
        # 调用 pandas_ta 库中的 PSL 函数，计算 PSL 指标
        result = pandas_ta.psl(self.close)
        # 断言返回结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "PSL_12"
        self.assertEqual(result.name, "PSL_12")

    # 测试量价震荡器（Price Volume Oscillator，PVO）指标
    def test_pvo(self):
        # 调用 pandas_ta 库中的 PVO 函数，计算 PVO 指标
        result = pandas_ta.pvo(self.volume)
        # 断言返回结果是 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果 DataFrame 的名称为 "PVO_12_26_9"
        self.assertEqual(result.name, "PVO_12_26_9")

    # 测试 QQE 指标
    def test_qqe(self):
        # 调用 pandas_ta 库中的 QQE 函数，计算 QQE 指标
        result = pandas_ta.qqe(self.close)
        # 断言返回结果是 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果 DataFrame 的名称为 "QQE_14_5_4.236"
        self.assertEqual(result.name, "QQE_14_5_4.236")

    # 测试变动率指标（Rate of Change，ROC）
    def test_roc(self):
        # 测试不使用 talib 计算 ROC 指标
        result = pandas_ta.roc(self.close, talib=False)
        # 断言返回结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "ROC_10"
        self.assertEqual(result.name, "ROC_10")

        try:
            # 尝试使用 talib 计算 ROC 指标
            expected = tal.ROC(self.close)
            # 对比结果与 talib 计算结果
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 若对比失败，则进行误差分析
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 若出现异常，则输出错误分析
                error_analysis(result, CORRELATION, ex)

        # 重新计算 ROC 指标（使用 pandas_ta 默认设置）
        result = pandas_ta.roc(self.close)
        # 断言返回结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "ROC_10"
        self.assertEqual(result.name, "ROC_10")

    # 测试相对强弱指标（Relative Strength Index，RSI）
    def test_rsi(self):
        # 测试不使用 talib 计算 RSI 指标
        result = pandas_ta.rsi(self.close, talib=False)
        # 断言返回结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "RSI_14"
        self.assertEqual(result.name, "RSI_14")

        try:
            # 尝试使用 talib 计算 RSI 指标
            expected = tal.RSI(self.close)
            # 对比结果与 talib 计算结果
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 若对比失败，则进行误差分析
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 若出现异常，则输出错误分析
                error_analysis(result, CORRELATION, ex)

        # 重新计算 RSI 指标（使用 pandas_ta 默认设置）
        result = pandas_ta.rsi(self.close)
        # 断言返回结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "RSI_14"
        self.assertEqual(result.name, "RSI_14")

    # 测试相对强弱指数（Relative Strength Index Smoothed，RSX）
    def test_rsx(self):
        # 调用 pandas_ta 库中的 RSX 函数，计算 RSX 指标
        result = pandas_ta.rsx(self.close)
        # 断言返回结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果 Series 的名称为 "RSX_14"
        self.assertEqual(result.name, "RSX_14")
    # 测试 RVGI 指标计算函数
    def test_rvgi(self):
        # 调用 rvgi 函数计算结果
        result = pandas_ta.rvgi(self.open, self.high, self.low, self.close)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果名称为 "RVGI_14_4"
        self.assertEqual(result.name, "RVGI_14_4")

    # 测试斜率指标计算函数
    def test_slope(self):
        # 调用 slope 函数计算结果
        result = pandas_ta.slope(self.close)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果名称为 "SLOPE_1"
        self.assertEqual(result.name, "SLOPE_1")

    # 测试斜率指标计算函数，返回角度值
    def test_slope_as_angle(self):
        # 调用 slope 函数计算结果，返回角度值
        result = pandas_ta.slope(self.close, as_angle=True)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果名称为 "ANGLEr_1"
        self.assertEqual(result.name, "ANGLEr_1")

    # 测试斜率指标计算函数，返回角度值并转换为度数
    def test_slope_as_angle_to_degrees(self):
        # 调用 slope 函数计算结果，返回角度值并转换为度数
        result = pandas_ta.slope(self.close, as_angle=True, to_degrees=True)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果名称为 "ANGLEd_1"
        self.assertEqual(result.name, "ANGLEd_1")

    # 测试 SMI 指标计算函数
    def test_smi(self):
        # 调用 smi 函数计算结果
        result = pandas_ta.smi(self.close)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果名称为 "SMI_5_20_5"
        self.assertEqual(result.name, "SMI_5_20_5")
        # 断言结果列数为 3
        self.assertEqual(len(result.columns), 3)

    # 测试 SMI 指标计算函数，设置 scalar 参数
    def test_smi_scalar(self):
        # 调用 smi 函数计算结果，设置 scalar 参数为 10
        result = pandas_ta.smi(self.close, scalar=10)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果名称为 "SMI_5_20_5_10.0"
        self.assertEqual(result.name, "SMI_5_20_5_10.0")
        # 断言结果列数为 3
        self.assertEqual(len(result.columns), 3)

    # 测试 Squeeze 指标计算函数
    def test_squeeze(self):
        # 调用 squeeze 函数计算结果
        result = pandas_ta.squeeze(self.high, self.low, self.close)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果名称为 "SQZ_20_2.0_20_1.5"

        # 调用 squeeze 函数计算结果，设置 tr 参数为 False
        result = pandas_ta.squeeze(self.high, self.low, self.close, tr=False)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果名称为 "SQZhlr_20_2.0_20_1.5"

        # 调用 squeeze 函数计算结果，设置 lazybear 参数为 True
        result = pandas_ta.squeeze(self.high, self.low, self.close, lazybear=True)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果名称为 "SQZ_20_2.0_20_1.5_LB"

        # 调用 squeeze 函数计算结果，设置 tr 和 lazybear 参数为 True
        result = pandas_ta.squeeze(self.high, self.low, self.close, tr=False, lazybear=True)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果名称为 "SQZhlr_20_2.0_20_1.5_LB"

    # 测试 Squeeze Pro 指标计算函数
    def test_squeeze_pro(self):
        # 调用 squeeze_pro 函数计算结果
        result = pandas_ta.squeeze_pro(self.high, self.low, self.close)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果名称为 "SQZPRO_20_2.0_20_2_1.5_1"

        # 调用 squeeze_pro 函数计算结果，设置 tr 参数为 False
        result = pandas_ta.squeeze_pro(self.high, self.low, self.close, tr=False)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果名称为 "SQZPROhlr_20_2.0_20_2_1.5_1"

        # 调用 squeeze_pro 函数计算结果，设置各参数值
        result = pandas_ta.squeeze_pro(self.high, self.low, self.close, 20, 2, 20, 3, 2, 1)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果名称为 "SQZPRO_20_2.0_20_3.0_2.0_1.0"

        # 调用 squeeze_pro 函数计算结果，设置各参数值和 tr 参数为 False
        result = pandas_ta.squeeze_pro(self.high, self.low, self.close, 20, 2, 20, 3, 2, 1, tr=False)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果名称为 "SQZPROhlr_20_2.0_20_3.0_2.0_1.0"
    # 测试 Smoothed Triple Exponential Moving Average (STC) 函数
    def test_stc(self):
        # 使用 pandas_ta 库中的 stc 函数计算结果
        result = pandas_ta.stc(self.close)
        # 断言结果是 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "STC_10_12_26_0.5"
        self.assertEqual(result.name, "STC_10_12_26_0.5")

    # 测试 Stochastic Oscillator (STOCH) 函数
    def test_stoch(self):
        # TV Correlation
        # 使用 pandas_ta 库中的 stoch 函数计算结果
        result = pandas_ta.stoch(self.high, self.low, self.close)
        # 断言结果是 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "STOCH_14_3_3"
        self.assertEqual(result.name, "STOCH_14_3_3")

        try:
            # 使用 talib 库中的 STOCH 函数计算预期结果
            expected = tal.STOCH(self.high, self.low, self.close, 14, 3, 0, 3, 0)
            # 构建预期结果的 DataFrame
            expecteddf = DataFrame({"STOCHk_14_3_0_3_0": expected[0], "STOCHd_14_3_0_3": expected[1]})
            # 断言结果与预期结果相等
            pdt.assert_frame_equal(result, expecteddf)
        except AssertionError:
            try:
                # 计算结果与预期结果的相关性
                stochk_corr = pandas_ta.utils.df_error_analysis(result.iloc[:, 0], expecteddf.iloc[:, 0], col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(stochk_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果相关性不符合要求，进行错误分析
                error_analysis(result.iloc[:, 0], CORRELATION, ex)

            try:
                # 计算结果与预期结果的相关性
                stochd_corr = pandas_ta.utils.df_error_analysis(result.iloc[:, 1], expecteddf.iloc[:, 1], col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(stochd_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果相关性不符合要求，进行错误分析
                error_analysis(result.iloc[:, 1], CORRELATION, ex, newline=False)

    # 测试 Stochastic RSI (STOCHRSI) 函数
    def test_stochrsi(self):
        # TV Correlation
        # 使用 pandas_ta 库中的 stochrsi 函数计算结果
        result = pandas_ta.stochrsi(self.close)
        # 断言结果是 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "STOCHRSI_14_14_3_3"
        self.assertEqual(result.name, "STOCHRSI_14_14_3_3")

        try:
            # 使用 talib 库中的 STOCHRSI 函数计算预期结果
            expected = tal.STOCHRSI(self.close, 14, 14, 3, 0)
            # 构建预期结果的 DataFrame
            expecteddf = DataFrame({"STOCHRSIk_14_14_0_3": expected[0], "STOCHRSId_14_14_3_0": expected[1]})
            # 断言结果与预期结果相等
            pdt.assert_frame_equal(result, expecteddf)
        except AssertionError:
            try:
                # 计算结果与预期结果的相关性
                stochrsid_corr = pandas_ta.utils.df_error_analysis(result.iloc[:, 0], expecteddf.iloc[:, 1], col=CORRELATION)
                # 断言相关性大于阈值
                self.assertGreater(stochrsid_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果相关性不符合要求，进行错误分析
                error_analysis(result.iloc[:, 0], CORRELATION, ex, newline=False)

    # 跳过测试 TS Sequential 函数
    @skip
    def test_td_seq(self):
        """TS Sequential: Working but SLOW implementation"""
        # 使用 pandas_ta 库中的 td_seq 函数计算结果（已跳过）
        result = pandas_ta.td_seq(self.close)
        # 断言结果是 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "TD_SEQ"

    # 测试 Triple Exponential Moving Average (TRIX) 函数
    def test_trix(self):
        # 使用 pandas_ta 库中的 trix 函数计算结果
        result = pandas_ta.trix(self.close)
        # 断言结果是 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "TRIX_30_9"

    # 测试 True Strength Index (TSI) 函数
    def test_tsi(self):
        # 使用 pandas_ta 库中的 tsi 函数计算结果
        result = pandas_ta.tsi(self.close)
        # 断言结果是 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "TSI_13_25_13"
    # 测试 `uo` 函数的单元测试
    def test_uo(self):
        # 使用 pandas_ta 库的 UO 函数计算结果
        result = pandas_ta.uo(self.high, self.low, self.close, talib=False)
        # 断言结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "UO_7_14_28"
        self.assertEqual(result.name, "UO_7_14_28")
    
        try:
            # 使用 TA-Lib 库的 ULTOSC 函数计算预期结果
            expected = tal.ULTOSC(self.high, self.low, self.close)
            # 比较结果和预期结果，不检查名称
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 分析结果和预期结果的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于预定义的阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 处理异常情况，调用错误分析函数
                error_analysis(result, CORRELATION, ex)
    
        # 使用 pandas_ta 库的 UO 函数计算结果（默认情况下使用 TA-Lib）
        result = pandas_ta.uo(self.high, self.low, self.close)
        # 断言结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "UO_7_14_28"
        self.assertEqual(result.name, "UO_7_14_28")
    
    # 测试 `willr` 函数的单元测试
    def test_willr(self):
        # 使用 pandas_ta 库的 WILLR 函数计算结果
        result = pandas_ta.willr(self.high, self.low, self.close, talib=False)
        # 断言结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "WILLR_14"
        self.assertEqual(result.name, "WILLR_14")
    
        try:
            # 使用 TA-Lib 库的 WILLR 函数计算预期结果
            expected = tal.WILLR(self.high, self.low, self.close)
            # 比较结果和预期结果，不检查名称
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 分析结果和预期结果的相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 断言相关性大于预定义的阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 处理异常情况，调用错误分析函数
                error_analysis(result, CORRELATION, ex)
    
        # 使用 pandas_ta 库的 WILLR 函数计算结果（默认情况下使用 TA-Lib）
        result = pandas_ta.willr(self.high, self.low, self.close)
        # 断言结果是 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "WILLR_14"
        self.assertEqual(result.name, "WILLR_14")
```