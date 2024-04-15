# `.\pandas-ta\tests\test_indicator_cycles.py`

```
# 从config模块中导入error_analysis, sample_data, CORRELATION, CORRELATION_THRESHOLD, VERBOSE变量
# 从context模块中导入pandas_ta
from .config import error_analysis, sample_data, CORRELATION, CORRELATION_THRESHOLD, VERBOSE
from .context import pandas_ta

# 从unittest模块中导入TestCase类和skip函数
# 导入pandas.testing模块并重命名为pdt
import pandas.testing as pdt
# 从pandas模块中导入DataFrame和Series类
from pandas import DataFrame, Series

# 导入talib库并重命名为tal
import talib as tal

# 定义TestCycles类，继承自TestCase类
class TestCycles(TestCase):
    # 设置类方法setUpClass，在测试类开始时执行
    @classmethod
    def setUpClass(cls):
        # 设置类属性data为sample_data
        cls.data = sample_data
        # 将data的列名转换为小写
        cls.data.columns = cls.data.columns.str.lower()
        # 设置类属性open为data的"open"列
        cls.open = cls.data["open"]
        # 设置类属性high为data的"high"列
        cls.high = cls.data["high"]
        # 设置类属性low为data的"low"列
        cls.low = cls.data["low"]
        # 设置类属性close为data的"close"列
        cls.close = cls.data["close"]
        # 如果"data"的列中包含"volume"列，设置类属性volume为"data"的"volume"列
        if "volume" in cls.data.columns:
            cls.volume = cls.data["volume"]

    # 设置类方法tearDownClass，在测试类结束时执行
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
        # 如果类有volume属性，删除类属性volume
        if hasattr(cls, "volume"):
            del cls.volume
        # 删除类属性data
        del cls.data

    # 设置实例方法setUp，在每个测试方法执行前执行
    def setUp(self): pass
    # 设置实例方法tearDown，在每个测试方法执行后执行
    def tearDown(self): pass

    # 定义测试方法test_ebsw
    def test_ebsw(self):
        # 调用pandas_ta模块中的ebsw函数，并传入close列，将结果赋给result变量
        result = pandas_ta.ebsw(self.close)
        # 断言result的类型为Series
        self.assertIsInstance(result, Series)
        # 断言result的名称为"EBSW_40_10"
        self.assertEqual(result.name, "EBSW_40_10")
```