# `.\pandas-ta\tests\test_indicator_volume.py`

```py
# 从.config中导入错误分析、示例数据、相关性、相关性阈值、详细模式
from .config import error_analysis, sample_data, CORRELATION, CORRELATION_THRESHOLD, VERBOSE
# 从.context中导入pandas_ta
from .context import pandas_ta

# 导入TestCase和skip
from unittest import TestCase, skip
# 导入pandas测试工具
import pandas.testing as pdt
# 导入DataFrame和Series
from pandas import DataFrame, Series

# 导入talib库，并重命名为tal
import talib as tal

# 定义测试Volume的测试类
class TestVolume(TestCase):

    # 设置测试类的一些初始属性
    @classmethod
    def setUpClass(cls):
        cls.data = sample_data
        # 将列名转换为小写
        cls.data.columns = cls.data.columns.str.lower()
        # 设置测试数据的open、high、low、close列
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        # 如果数据中包含volume列，则设置volume_
        if "volume" in cls.data.columns:
            cls.volume_ = cls.data["volume"]

    # 清理测试类的一些属性
    @classmethod
    def tearDownClass(cls):
        del cls.open
        del cls.high
        del cls.low
        del cls.close
        # 如果存在volume属性，则删除
        if hasattr(cls, "volume"):
            del cls.volume_
        del cls.data

    # 设置测试方法的setUp方法
    def setUp(self): pass

    # 设置测试方法的tearDown方法
    def tearDown(self): pass

    # 测试ad方法
    def test_ad(self):
        # 调用pandas_ta中的ad方法，不使用talib
        result = pandas_ta.ad(self.high, self.low, self.close, self.volume_, talib=False)
        # 检查返回结果是否为Series类型
        self.assertIsInstance(result, Series)
        # 检查返回结果的名称是否为"AD"
        self.assertEqual(result.name, "AD")

        # 尝试使用talib计算AD指标并检查结果是否一致，不检查名称
        try:
            expected = tal.AD(self.high, self.low, self.close, self.volume_)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            # 如果结果不一致，则进行错误分析
            try:
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 检查相关性是否大于相关性阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果出现异常，则进行错误分析
                error_analysis(result, CORRELATION, ex)

        # 再次调用pandas_ta中的ad方法，不使用talib
        result = pandas_ta.ad(self.high, self.low, self.close, self.volume_)
        # 检查返回结果是否为Series类型
        self.assertIsInstance(result, Series)
        # 检查返回结果的名称是否为"AD"

    # 测试ad_open方法
    def test_ad_open(self):
        # 调用pandas_ta中的ad方法，不使用talib
        result = pandas_ta.ad(self.high, self.low, self.close, self.volume_, self.open)
        # 检查返回结果是否为Series类型
        self.assertIsInstance(result, Series)
        # 检查返回结果的名称是否为"ADo"

    # 测试adosc方法
    def test_adosc(self):
        # 调用pandas_ta中的adosc方法，不使用talib
        result = pandas_ta.adosc(self.high, self.low, self.close, self.volume_, talib=False)
        # 检查返回结果是否为Series类型
        self.assertIsInstance(result, Series)
        # 检查返回结果的名称是否为"ADOSC_3_10"

        # 尝试使用talib计算ADOSC指标并检查结果是否一致，不检查名称
        try:
            expected = tal.ADOSC(self.high, self.low, self.close, self.volume_)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            # 如果结果不一致，则进行错误分析
            try:
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                # 检查相关性是否大于相关性阈值
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果出现异常，则进行错误分析
                error_analysis(result, CORRELATION, ex)

        # 再次调用pandas_ta中的adosc方法，不使用talib
        result = pandas_ta.adosc(self.high, self.low, self.close, self.volume_)
        # 检查返回结果是否为Series类型
        self.assertIsInstance(result, Series)
        # 检查返回结果的名称是否为"ADOSC_3_10"

    # 测试aobv方法
    def test_aobv(self):
        # 调用pandas_ta中的aobv方法
        result = pandas_ta.aobv(self.close, self.volume_)
        # 检查返回结果是否为DataFrame类型
        self.assertIsInstance(result, DataFrame)
        # 检查返回结果的名称是否为"AOBVe_4_12_2_2_2"
    # 测试 CMF 指标计算函数
    def test_cmf(self):
        # 调用 pandas_ta 库的 CMF 函数计算结果
        result = pandas_ta.cmf(self.high, self.low, self.close, self.volume_)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "CMF_20"
        self.assertEqual(result.name, "CMF_20")

    # 测试 EFI 指标计算函数
    def test_efi(self):
        # 调用 pandas_ta 库的 EFI 函数计算结果
        result = pandas_ta.efi(self.close, self.volume_)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "EFI_13"
        self.assertEqual(result.name, "EFI_13")

    # 测试 EOM 指标计算函数
    def test_eom(self):
        # 调用 pandas_ta 库的 EOM 函数计算结果
        result = pandas_ta.eom(self.high, self.low, self.close, self.volume_)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "EOM_14_100000000"
        self.assertEqual(result.name, "EOM_14_100000000")

    # 测试 KVO 指标计算函数
    def test_kvo(self):
        # 调用 pandas_ta 库的 KVO 函数计算结果
        result = pandas_ta.kvo(self.high, self.low, self.close, self.volume_)
        # 断言结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "KVO_34_55_13"
        self.assertEqual(result.name, "KVO_34_55_13")

    # 测试 MFI 指标计算函数
    def test_mfi(self):
        # 调用 pandas_ta 库的 MFI 函数计算结果，指定不使用 talib
        result = pandas_ta.mfi(self.high, self.low, self.close, self.volume_, talib=False)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "MFI_14"
        self.assertEqual(result.name, "MFI_14")

        try:
            # 尝试使用 talib 计算 MFI，并与 pandas_ta 计算结果进行比较
            expected = tal.MFI(self.high, self.low, self.close, self.volume_)
            # 检查两个 Series 是否相等
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 如果计算结果不相等，则进行错误分析并检查相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果出现异常，则进行错误分析
                error_analysis(result, CORRELATION, ex)

        # 重新使用 pandas_ta 计算 MFI 指标
        result = pandas_ta.mfi(self.high, self.low, self.close, self.volume_)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "MFI_14"
        self.assertEqual(result.name, "MFI_14")

    # 测试 NVI 指标计算函数
    def test_nvi(self):
        # 调用 pandas_ta 库的 NVI 函数计算结果
        result = pandas_ta.nvi(self.close, self.volume_)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "NVI_1"
        self.assertEqual(result.name, "NVI_1")

    # 测试 OBV 指标计算函数
    def test_obv(self):
        # 调用 pandas_ta 库的 OBV 函数计算结果，指定不使用 talib
        result = pandas_ta.obv(self.close, self.volume_, talib=False)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "OBV"
        self.assertEqual(result.name, "OBV")

        try:
            # 尝试使用 talib 计算 OBV，并与 pandas_ta 计算结果进行比较
            expected = tal.OBV(self.close, self.volume_)
            # 检查两个 Series 是否相等
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                # 如果计算结果不相等，则进行错误分析并检查相关性
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                # 如果出现异常，则进行错误分析
                error_analysis(result, CORRELATION, ex)

        # 重新使用 pandas_ta 计算 OBV 指标
        result = pandas_ta.obv(self.close, self.volume_)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "OBV"
        self.assertEqual(result.name, "OBV")

    # 测试 PVI 指标计算函数
    def test_pvi(self):
        # 调用 pandas_ta 库的 PVI 函数计算结果
        result = pandas_ta.pvi(self.close, self.volume_)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "PVI_1"
        self.assertEqual(result.name, "PVI_1")

    # 测试 PVOL 指标计算函数
    def test_pvol(self):
        # 调用 pandas_ta 库的 PVOL 函数计算结果
        result = pandas_ta.pvol(self.close, self.volume_)
        # 断言结果类型为 Series
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "PVOL"
        self.assertEqual(result.name, "PVOL")
    # 测试 Price Volume Ratio (PVR) 指标函数
    def test_pvr(self):
        # 计算 PVR 指标
        result = pandas_ta.pvr(self.close, self.volume_)
        # 确保返回结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 确保返回结果的名称为 "PVR"
        self.assertEqual(result.name, "PVR")
        # 样本指标值来自于 SPY
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 3)
        self.assertEqual(result[4], 2)
        self.assertEqual(result[6], 4)

    # 测试 Price Volume Trend (PVT) 指标函数
    def test_pvt(self):
        # 计算 PVT 指标
        result = pandas_ta.pvt(self.close, self.volume_)
        # 确保返回结果为 Series 类型
        self.assertIsInstance(result, Series)
        # 确保返回结果的名称为 "PVT"
        self.assertEqual(result.name, "PVT")

    # 测试 Volume Price (VP) 指标函数
    def test_vp(self):
        # 计算 VP 指标
        result = pandas_ta.vp(self.close, self.volume_)
        # 确保返回结果为 DataFrame 类型
        self.assertIsInstance(result, DataFrame)
        # 确保返回结果的名称为 "VP_10"
        self.assertEqual(result.name, "VP_10")
```