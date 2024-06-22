# `.\pandas-ta\tests\test_ext_indicator_overlap_ext.py`

```py
# 从当前包中导入 sample_data 和 pandas_ta 模块
from .config import sample_data
from .context import pandas_ta
# 从 unittest 模块中导入 skip 和 TestCase 类
from unittest import skip, TestCase
# 从 pandas 模块中导入 DataFrame 类
from pandas import DataFrame

# 定义测试类 TestOverlapExtension，继承自 TestCase 类
class TestOverlapExtension(TestCase):
    # 在测试类中所有测试方法执行之前执行，用于设置测试数据
    @classmethod
    def setUpClass(cls):
        cls.data = sample_data

    # 在测试类中所有测试方法执行之后执行，用于清理测试数据
    @classmethod
    def tearDownClass(cls):
        del cls.data

    # 在每个测试方法执行之前执行的方法，此处为空
    def setUp(self): pass
    
    # 在每个测试方法执行之后执行的方法，此处为空
    def tearDown(self): pass

    # 测试 alma 方法的扩展功能
    def test_alma_ext(self):
        # 调用 data 对象的 ta 属性的 alma 方法，并将结果追加到 data 中
        self.data.ta.alma(append=True)
        # 断言 data 对象是 DataFrame 类的实例
        self.assertIsInstance(self.data, DataFrame)
        # 断言 data 对象的最后一列的列名为 "ALMA_10_6.0_0.85"
        self.assertEqual(self.data.columns[-1], "ALMA_10_6.0_0.85")

    # 测试 dema 方法的扩展功能，以下测试方法同理
    def test_dema_ext(self):
        self.data.ta.dema(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "DEMA_10")

    def test_ema_ext(self):
        self.data.ta.ema(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "EMA_10")

    def test_fwma_ext(self):
        self.data.ta.fwma(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "FWMA_10")

    def test_hilo_ext(self):
        self.data.ta.hilo(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(list(self.data.columns[-3:]), ["HILO_13_21", "HILOl_13_21", "HILOs_13_21"])

    def test_hl2_ext(self):
        self.data.ta.hl2(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "HL2")

    def test_hlc3_ext(self):
        self.data.ta.hlc3(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "HLC3")

    def test_hma_ext(self):
        self.data.ta.hma(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "HMA_10")

    def test_hwma_ext(self):
        self.data.ta.hwma(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "HWMA_0.2_0.1_0.1")

    def test_jma_ext(self):
        self.data.ta.jma(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "JMA_7_0")

    def test_kama_ext(self):
        self.data.ta.kama(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "KAMA_10_2_30")

    def test_ichimoku_ext(self):
        self.data.ta.ichimoku(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(list(self.data.columns[-5:]), ["ISA_9", "ISB_26", "ITS_9", "IKS_26", "ICS_26"])

    def test_linreg_ext(self):
        self.data.ta.linreg(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "LR_14")

    def test_mcgd_ext(self):
        self.data.ta.mcgd(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "MCGD_10")
    # 测试扩展方法：计算中间点指标
    def test_midpoint_ext(self):
        # 调用中间点指标计算方法，并将结果追加到数据帧中
        self.data.ta.midpoint(append=True)
        # 断言数据类型为数据帧
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"MIDPOINT_2"
        self.assertEqual(self.data.columns[-1], "MIDPOINT_2")

    # 测试扩展方法：计算中间价指标
    def test_midprice_ext(self):
        # 调用中间价指标计算方法，并将结果追加到数据帧中
        self.data.ta.midprice(append=True)
        # 断言数据类型为数据帧
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"MIDPRICE_2"
        self.assertEqual(self.data.columns[-1], "MIDPRICE_2")

    # 测试扩展方法：计算OHLC4指标
    def test_ohlc4_ext(self):
        # 调用OHLC4指标计算方法，并将结果追加到数据帧中
        self.data.ta.ohlc4(append=True)
        # 断言数据类型为数据帧
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"OHLC4"
        self.assertEqual(self.data.columns[-1], "OHLC4")

    # 测试扩展方法：计算PWMA指标
    def test_pwma_ext(self):
        # 调用PWMA指标计算方法，并将结果追加到数据帧中
        self.data.ta.pwma(append=True)
        # 断言数据类型为数据帧
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"PWMA_10"
        self.assertEqual(self.data.columns[-1], "PWMA_10")

    # 测试扩展方法：计算RMA指标
    def test_rma_ext(self):
        # 调用RMA指标计算方法，并将结果追加到数据帧中
        self.data.ta.rma(append=True)
        # 断言数据类型为数据帧
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"RMA_10"
        self.assertEqual(self.data.columns[-1], "RMA_10")

    # 测试扩展方法：计算SINWMA指标
    def test_sinwma_ext(self):
        # 调用SINWMA指标计算方法，并将结果追加到数据帧中
        self.data.ta.sinwma(append=True)
        # 断言数据类型为数据帧
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"SINWMA_14"
        self.assertEqual(self.data.columns[-1], "SINWMA_14")

    # 测试扩展方法：计算SMA指标
    def test_sma_ext(self):
        # 调用SMA指标计算方法，并将结果追加到数据帧中
        self.data.ta.sma(append=True)
        # 断言数据类型为数据帧
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"SMA_10"
        self.assertEqual(self.data.columns[-1], "SMA_10")

    # 测试扩展方法：计算SSF指标
    def test_ssf_ext(self):
        # 调用SSF指标计算方法，并将结果追加到数据帧中，使用两个极点
        self.data.ta.ssf(append=True, poles=2)
        # 断言数据类型为数据帧
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"SSF_10_2"
        self.assertEqual(self.data.columns[-1], "SSF_10_2")

        # 再次调用SSF指标计算方法，并将结果追加到数据帧中，使用三个极点
        self.data.ta.ssf(append=True, poles=3)
        # 断言数据类型为数据帧
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"SSF_10_3"
        self.assertEqual(self.data.columns[-1], "SSF_10_3")

    # 测试扩展方法：计算SWMA指标
    def test_swma_ext(self):
        # 调用SWMA指标计算方法，并将结果追加到数据帧中
        self.data.ta.swma(append=True)
        # 断言数据类型为数据帧
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"SWMA_10"
        self.assertEqual(self.data.columns[-1], "SWMA_10")

    # 测试扩展方法：计算超级趋势指标
    def test_supertrend_ext(self):
        # 调用超级趋势指标计算方法，并将结果追加到数据帧中
        self.data.ta.supertrend(append=True)
        # 断言数据类型为数据帧
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后四列的列名分别为"SUPERT_7_3.0", "SUPERTd_7_3.0", "SUPERTl_7_3.0", "SUPERTs_7_3.0"
        self.assertEqual(list(self.data.columns[-4:]), ["SUPERT_7_3.0", "SUPERTd_7_3.0", "SUPERTl_7_3.0", "SUPERTs_7_3.0"])

    # 测试扩展方法：计算T3指标
    def test_t3_ext(self):
        # 调用T3指标计算方法，并将结果追加到数据帧中
        self.data.ta.t3(append=True)
        # 断言数据类型为数据帧
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"T3_10_0.7"
        self.assertEqual(self.data.columns[-1], "T3_10_0.7")

    # 测试扩展方法：计算TEMA指标
    def test_tema_ext(self):
        # 调用TEMA指标计算方法，并将结果追加到数据帧中
        self.data.ta.tema(append=True)
        #
    # 测试 VWMA 扩展函数
    def test_vwma_ext(self):
        # 调用 VWMA 函数，将结果追加到 DataFrame 中
        self.data.ta.vwma(append=True)
        # 断言 DataFrame 实例化
        self.assertIsInstance(self.data, DataFrame)
        # 断言 DataFrame 最后一列的列名为 "VWMA_10"
        self.assertEqual(self.data.columns[-1], "VWMA_10")
    
    # 测试 WCP 扩展函数
    def test_wcp_ext(self):
        # 调用 WCP 函数，将结果追加到 DataFrame 中
        self.data.ta.wcp(append=True)
        # 断言 DataFrame 实例化
        self.assertIsInstance(self.data, DataFrame)
        # 断言 DataFrame 最后一列的列名为 "WCP"
        self.assertEqual(self.data.columns[-1], "WCP")
    
    # 测试 WMA 扩展函数
    def test_wma_ext(self):
        # 调用 WMA 函数，将结果追加到 DataFrame 中
        self.data.ta.wma(append=True)
        # 断言 DataFrame 实例化
        self.assertIsInstance(self.data, DataFrame)
        # 断言 DataFrame 最后一列的列名为 "WMA_10"
        self.assertEqual(self.data.columns[-1], "WMA_10")
    
    # 测试 ZLMA 扩展函数
    def test_zlma_ext(self):
        # 调用 ZLMA 函数，将结果追加到 DataFrame 中
        self.data.ta.zlma(append=True)
        # 断言 DataFrame 实例化
        self.assertIsInstance(self.data, DataFrame)
        # 断言 DataFrame 最后一列的列名为 "ZL_EMA_10"
        self.assertEqual(self.data.columns[-1], "ZL_EMA_10")
```