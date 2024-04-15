# `.\pandas-ta\tests\test_ext_indicator_candle.py`

```py
# 从config模块中导入sample_data
from .config import sample_data
# 从context模块中导入pandas_ta
from .context import pandas_ta
# 从unittest模块中导入TestCase和skip
from unittest import TestCase, skip
# 从pandas模块中导入DataFrame
from pandas import DataFrame

# 定义测试类TestCandleExtension，继承自TestCase类
class TestCandleExtension(TestCase):
    # 设置类方法setUpClass，在所有测试用例执行前执行一次
    @classmethod
    def setUpClass(cls):
        cls.data = sample_data

    # 设置类方法tearDownClass，在所有测试用例执行后执行一次
    @classmethod
    def tearDownClass(cls):
        del cls.data

    # 定义实例方法setUp，在每个测试用例执行前执行一次
    def setUp(self): pass
    # 定义实例方法tearDown，在每个测试用例执行后执行一次
    def tearDown(self): pass

    # 测试CDL_DOJI_10_0.1模式的扩展
    def test_cdl_doji_ext(self):
        self.data.ta.cdl_pattern("doji", append=True)
        # 断言self.data是DataFrame类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言self.data的最后一列的列名为"CDL_DOJI_10_0.1"
        self.assertEqual(self.data.columns[-1], "CDL_DOJI_10_0.1")

    # 测试CDL_INSIDE模式的扩展
    def test_cdl_inside_ext(self):
        self.data.ta.cdl_pattern("inside", append=True)
        # 断言self.data是DataFrame类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言self.data的最后一列的列名为"CDL_INSIDE"
        self.assertEqual(self.data.columns[-1], "CDL_INSIDE")

    # 测试CDL_Z指标的扩展
    def test_cdl_z_ext(self):
        self.data.ta.cdl_z(append=True)
        # 断言self.data是DataFrame类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言self.data的倒数第四列到最后一列的列名为["open_Z_30_1", "high_Z_30_1", "low_Z_30_1", "close_Z_30_1"]
        self.assertEqual(list(self.data.columns[-4:]), ["open_Z_30_1", "high_Z_30_1", "low_Z_30_1", "close_Z_30_1"])

    # 测试HA指标的扩展
    def test_ha_ext(self):
        self.data.ta.ha(append=True)
        # 断言self.data是DataFrame类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言self.data的倒数第四列到最后一列的列名为["HA_open", "HA_high", "HA_low", "HA_close"]
        self.assertEqual(list(self.data.columns[-4:]), ["HA_open", "HA_high", "HA_low", "HA_close"])
```