# `.\pandas-ta\tests\test_ext_indicator_trend.py`

```
# 从config模块中导入sample_data变量
from .config import sample_data
# 从context模块中导入pandas_ta模块
from .context import pandas_ta
# 从unittest模块中导入skip和TestCase类
from unittest import skip, TestCase
# 从pandas模块中导入DataFrame类
from pandas import DataFrame

# 定义测试TrendExtension类，继承自TestCase类
class TestTrendExtension(TestCase):
    # 设置测试类的类方法setUpClass，用于设置测试数据
    @classmethod
    def setUpClass(cls):
        cls.data = sample_data

    # 设置测试类的类方法tearDownClass，用于清理测试数据
    @classmethod
    def tearDownClass(cls):
        del cls.data

    # 设置测试方法的setUp，无操作
    def setUp(self): pass
    # 设置测试方法的tearDown，无操作
    def tearDown(self): pass

    # 测试ADX扩展功能
    def test_adx_ext(self):
        # 调用ADX扩展方法，将结果附加到数据框中
        self.data.ta.adx(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后三列的列名
        self.assertEqual(list(self.data.columns[-3:]), ["ADX_14", "DMP_14", "DMN_14"])

    # 测试AMAT扩展功能
    def test_amat_ext(self):
        # 调用AMAT扩展方法，将结果附加到数据框中
        self.data.ta.amat(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后两列的列名
        self.assertEqual(list(self.data.columns[-2:]), ["AMATe_LR_8_21_2", "AMATe_SR_8_21_2"])

    # 测试Aroon扩展功能
    def test_aroon_ext(self):
        # 调用Aroon扩展方法，将结果附加到数据框中
        self.data.ta.aroon(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后三列的列名
        self.assertEqual(list(self.data.columns[-3:]), ["AROOND_14", "AROONU_14", "AROONOSC_14"])

    # 测试Chop扩展功能
    def test_chop_ext(self):
        # 调用Chop扩展方法，将结果附加到数据框中，设置ln参数为False
        self.data.ta.chop(append=True, ln=False)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名
        self.assertEqual(self.data.columns[-1], "CHOP_14_1_100")

        # 再次调用Chop扩展方法，将结果附加到数据框中，设置ln参数为True
        self.data.ta.chop(append=True, ln=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名
        self.assertEqual(self.data.columns[-1], "CHOPln_14_1_100")

    # 测试CKSP扩展功能
    def test_cksp_ext(self):
        # 调用CKSP扩展方法，将结果附加到数据框中，设置tvmode参数为False
        self.data.ta.cksp(tvmode=False, append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后两列的列名
        self.assertEqual(list(self.data.columns[-2:]), ["CKSPl_10_3_20", "CKSPs_10_3_20"])

    # 测试CKSP_TV扩展功能
    def test_cksp_tv_ext(self):
        # 调用CKSP扩展方法，将结果附加到数据框中，设置tvmode参数为True
        self.data.ta.cksp(tvmode=True, append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后两列的列名
        self.assertEqual(list(self.data.columns[-2:]), ["CKSPl_10_1_9", "CKSPs_10_1_9"])

    # 测试Decay扩展功能
    def test_decay_ext(self):
        # 调用Decay扩展方法，将结果附加到数据框中
        self.data.ta.decay(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名
        self.assertEqual(self.data.columns[-1], "LDECAY_5")

        # 再次调用Decay扩展方法，设置mode参数为"exp"，将结果附加到数据框中
        self.data.ta.decay(mode="exp", append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名
        self.assertEqual(self.data.columns[-1], "EXPDECAY_5")

    # 测试Decreasing扩展功能
    def test_decreasing_ext(self):
        # 调用Decreasing扩展方法，将结果附加到数据框中
        self.data.ta.decreasing(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名
        self.assertEqual(self.data.columns[-1], "DEC_1")

        # 再次调用Decreasing扩展方法，设置length参数为3，strict参数为True，将结果附加到数据框中
        self.data.ta.decreasing(length=3, strict=True, append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名
        self.assertEqual(self.data.columns[-1], "SDEC_3")

    # 测试DPO扩展功能
    def test_dpo_ext(self):
        # 调用DPO扩展方法，将结果附加到数据框中
        self.data.ta.dpo(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名
        self.assertEqual(self.data.columns[-1], "DPO_20")
    # 测试增长指标计算函数的扩展功能
    def test_increasing_ext(self):
        # 调用增长指标计算函数，将结果追加到原始数据后面
        self.data.ta.increasing(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为 "INC_1"
        self.assertEqual(self.data.columns[-1], "INC_1")

        # 调用增长指标计算函数，计算窗口长度为3的严格增长指标，并将结果追加到原始数据后面
        self.data.ta.increasing(length=3, strict=True, append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为 "SINC_3"
        self.assertEqual(self.data.columns[-1], "SINC_3")

    # 测试长期趋势计算函数的扩展功能
    def test_long_run_ext(self):
        # 未传递参数，返回原始数据
        self.assertEqual(self.data.ta.long_run(append=True).shape, self.data.shape)

        # 计算快速和慢速指数移动平均线
        fast = self.data.ta.ema(8)
        slow = self.data.ta.ema(21)
        # 计算长期趋势指标，并将结果追加到原始数据后面
        self.data.ta.long_run(fast, slow, append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为 "LR_2"
        self.assertEqual(self.data.columns[-1], "LR_2")

    # 测试抛物线停止和反转指标计算函数的扩展功能
    def test_psar_ext(self):
        # 计算抛物线停止和反转指标，并将结果追加到原始数据后面
        self.data.ta.psar(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后四列的列名为指定的抛物线参数
        self.assertEqual(
            list(self.data.columns[-4:]), ["PSARl_0.02_0.2", "PSARs_0.02_0.2", "PSARaf_0.02_0.2", "PSARr_0.02_0.2"])

    # 测试 QStick 指标计算函数的扩展功能
    def test_qstick_ext(self):
        # 计算 QStick 指标，并将结果追加到原始数据后面
        self.data.ta.qstick(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为 "QS_10"
        self.assertEqual(self.data.columns[-1], "QS_10")

    # 测试短期趋势计算函数的扩展功能
    def test_short_run_ext(self):
        # 未传递参数，返回原始数据
        self.assertEqual(
            self.data.ta.short_run(append=True).shape, self.data.shape)

        # 计算快速和慢速指数移动平均线
        fast = self.data.ta.ema(8)
        slow = self.data.ta.ema(21)
        # 计算短期趋势指标，并将结果追加到原始数据后面
        self.data.ta.short_run(fast, slow, append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为 "SR_2"
        self.assertEqual(self.data.columns[-1], "SR_2")

    # 测试 TTM 趋势指标计算函数的扩展功能
    def test_ttm_trend_ext(self):
        # 计算 TTM 趋势指标，并将结果追加到原始数据后面
        self.data.ta.ttm_trend(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为 "TTM_TRND_6"
        self.assertEqual(list(self.data.columns[-1:]), ["TTM_TRND_6"])

    # 测试涡轮指标计算函数的扩展功能
    def test_vortext_ext(self):
        # 计算涡轮指标，并将结果追加到原始数据后面
        self.data.ta.vortex(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后两列的列名为 "VTXP_14" 和 "VTXM_14"
        self.assertEqual(list(self.data.columns[-2:]), ["VTXP_14", "VTXM_14"])
```