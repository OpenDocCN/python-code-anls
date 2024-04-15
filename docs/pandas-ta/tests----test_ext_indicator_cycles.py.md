# `.\pandas-ta\tests\test_ext_indicator_cycles.py`

```py
# 从 pandas 库的 core.series 模块中导入 Series 类
from pandas.core.series import Series
# 从当前目录中的 config 模块中导入 sample_data 变量
from .config import sample_data
# 从当前目录中的 context 模块中导入 pandas_ta 模块
from .context import pandas_ta
# 从 unittest 模块中导入 TestCase 类
from unittest import TestCase
# 从 pandas 库中导入 DataFrame 类
from pandas import DataFrame

# 定义测试类 TestCylesExtension，继承自 TestCase 类
class TestCylesExtension(TestCase):
    # 在整个测试类运行之前执行的方法，设置测试数据
    @classmethod
    def setUpClass(cls):
        cls.data = sample_data

    # 在整个测试类运行之后执行的方法，清理测试数据
    @classmethod
    def tearDownClass(cls):
        del cls.data

    # 在每个测试方法运行之前执行的方法
    def setUp(self): pass
    # 在每个测试方法运行之后执行的方法
    def tearDown(self): pass

    # 定义测试方法 test_ebsw_ext，测试 EBSW 扩展函数
    def test_ebsw_ext(self):
        # 调用数据框对象的 ta 属性中的 ebsw 方法，并将结果追加到原数据框中
        self.data.ta.ebsw(append=True)
        # 断言数据框对象是 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言数据框对象的最后一列的列名为 "EBSW_40_10"
        self.assertEqual(self.data.columns[-1], "EBSW_40_10")
```