# `.\pandas-ta\tests\test_ext_indicator_volatility.py`

```py
# 导入所需模块
from .config import sample_data
from .context import pandas_ta
# 导入 TestCase 类
from unittest import TestCase
# 导入 DataFrame 类
from pandas import DataFrame

# 创建测试类 TestVolatilityExtension，继承自 TestCase 类
class TestVolatilityExtension(TestCase):
    # 在所有测试方法执行之前执行的方法
    @classmethod
    def setUpClass(cls):
        # 设置测试数据
        cls.data = sample_data

    # 在所有测试方法执行之后执行的方法
    @classmethod
    def tearDownClass(cls):
        # 删除测试数据
        del cls.data

    # 在每个测试方法执行之前执行的方法
    def setUp(self): pass
    
    # 在每个测试方法执行之后执行的方法
    def tearDown(self): pass

    # 测试 aberration 方法是否正常工作
    def test_aberration_ext(self):
        # 调用 aberration 方法
        self.data.ta.aberration(append=True)
        # 断言 data 是否为 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言 data 的最后四列是否等于给定列表
        self.assertEqual(list(self.data.columns[-4:]), ["ABER_ZG_5_15", "ABER_SG_5_15", "ABER_XG_5_15", "ABER_ATR_5_15"])

    # 测试 accbands 方法是否正常工作
    def test_accbands_ext(self):
        # 调用 accbands 方法
        self.data.ta.accbands(append=True)
        # 断言 data 是否为 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言 data 的最后三列是否等于给定列表
        self.assertEqual(list(self.data.columns[-3:]), ["ACCBL_20", "ACCBM_20", "ACCBU_20"])

    # 测试 atr 方法是否正常工作
    def test_atr_ext(self):
        # 调用 atr 方法
        self.data.ta.atr(append=True)
        # 断言 data 是否为 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言 data 的最后一列是否等于给定字符串
        self.assertEqual(self.data.columns[-1], "ATRr_14")

    # 测试 bbands 方法是否正常工作
    def test_bbands_ext(self):
        # 调用 bbands 方法
        self.data.ta.bbands(append=True)
        # 断言 data 是否为 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言 data 的最后五列是否等于给定列表
        self.assertEqual(list(self.data.columns[-5:]), ["BBL_5_2.0", "BBM_5_2.0", "BBU_5_2.0", "BBB_5_2.0", "BBP_5_2.0"])

    # 测试 donchian 方法是否正常工作
    def test_donchian_ext(self):
        # 调用 donchian 方法
        self.data.ta.donchian(append=True)
        # 断言 data 是否为 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言 data 的最后三列是否等于给定列表
        self.assertEqual(list(self.data.columns[-3:]), ["DCL_20_20", "DCM_20_20", "DCU_20_20"])

    # 测试 kc 方法是否正常工作
    def test_kc_ext(self):
        # 调用 kc 方法
        self.data.ta.kc(append=True)
        # 断言 data 是否为 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言 data 的最后三列是否等于给定列表
        self.assertEqual(list(self.data.columns[-3:]), ["KCLe_20_2", "KCBe_20_2", "KCUe_20_2"])

    # 测试 massi 方法是否正常工作
    def test_massi_ext(self):
        # 调用 massi 方法
        self.data.ta.massi(append=True)
        # 断言 data 是否为 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言 data 的最后一列是否等于给定字符串
        self.assertEqual(self.data.columns[-1], "MASSI_9_25")

    # 测试 natr 方法是否正常工作
    def test_natr_ext(self):
        # 调用 natr 方法
        self.data.ta.natr(append=True)
        # 断言 data 是否为 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言 data 的最后一列是否等于给定字符串
        self.assertEqual(self.data.columns[-1], "NATR_14")

    # 测试 pdist 方法是否正常工作
    def test_pdist_ext(self):
        # 调用 pdist 方法
        self.data.ta.pdist(append=True)
        # 断言 data 是否为 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言 data 的最后一列是否等于给定字符串
        self.assertEqual(self.data.columns[-1], "PDIST")

    # 测试 rvi 方法是否正常工作
    def test_rvi_ext(self):
        # 调用 rvi 方法
        self.data.ta.rvi(append=True)
        # 断言 data 是否为 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言 data 的最后一列是否等于给定字符串
        self.assertEqual(self.data.columns[-1], "RVI_14")

    # 测试 rvi 方法是否正常工作，且使用 refined 参数
    def test_rvi_refined_ext(self):
        # 调用 rvi 方法，传入 refined 参数为 True
        self.data.ta.rvi(refined=True, append=True)
        # 断言 data 是否为 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言 data 的最后一列是否等于给定字符串
        self.assertEqual(self.data.columns[-1], "RVIr_14")

    # 测试 rvi 方法是否正常工作，且使用 thirds 参数
    def test_rvi_thirds_ext(self):
        # 调用 rvi 方法，传入 thirds 参数为 True
        self.data.ta.rvi(thirds=True, append=True)
        # 断言 data 是否为 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言 data 的最后一列是否等于给定字符串
        self.assertEqual(self.data.columns[-1], "RVIt_14")
    # 测试扩展函数：计算热力指标
    def test_thermo_ext(self):
        # 调用热力指标计算函数，并将结果附加到原始数据上
        self.data.ta.thermo(append=True)
        # 确保结果是 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 检查结果中最后四列的列名是否符合预期
        self.assertEqual(list(self.data.columns[-4:]), ["THERMO_20_2_0.5", "THERMOma_20_2_0.5", "THERMOl_20_2_0.5", "THERMOs_20_2_0.5"])

    # 测试扩展函数：计算真实范围
    def test_true_range_ext(self):
        # 调用真实范围计算函数，并将结果附加到原始数据上
        self.data.ta.true_range(append=True)
        # 确保结果是 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 检查结果中最后一列的列名是否符合预期
        self.assertEqual(self.data.columns[-1], "TRUERANGE_1")

    # 测试扩展函数：计算 UI 指标
    def test_ui_ext(self):
        # 调用 UI 指标计算函数，并将结果附加到原始数据上
        self.data.ta.ui(append=True)
        # 确保结果是 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 检查结果中最后一列的列名是否符合预期
        self.assertEqual(self.data.columns[-1], "UI_14")

        # 调用 UI 指标计算函数，使用 everget 参数，并将结果附加到原始数据上
        self.data.ta.ui(append=True, everget=True)
        # 确保结果是 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 检查结果中最后一列的列名是否符合预期
        self.assertEqual(self.data.columns[-1], "UIe_14")
```