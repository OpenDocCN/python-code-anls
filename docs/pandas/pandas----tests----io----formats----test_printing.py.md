# `D:\src\scipysrc\pandas\pandas\tests\io\formats\test_printing.py`

```
# 导入必要的模块和库
# 注意！此文件专门针对 pandas.io.formats.printing 的实用函数，而非 pandas 对象的一般打印。
from collections.abc import Mapping  # 导入 Mapping 类，用于创建自定义映射类
import string  # 导入 string 模块，用于处理字符串操作

import pandas._config.config as cf  # 导入 pandas 内部配置模块

from pandas.io.formats import printing  # 导入 pandas 打印格式模块中的 printing 函数

# 测试 adjoin 函数的功能
def test_adjoin():
    # 定义测试数据和期望的输出结果
    data = [["a", "b", "c"], ["dd", "ee", "ff"], ["ggg", "hhh", "iii"]]
    expected = "a  dd  ggg\nb  ee  hhh\nc  ff  iii"

    # 调用 adjoin 函数，将数据连接为字符串
    adjoined = printing.adjoin(2, *data)

    # 断言连接后的结果与预期的输出结果相等
    assert adjoined == expected

# 自定义的 Mapping 类，继承自 collections.abc.Mapping
class MyMapping(Mapping):
    # 实现获取项目的方法
    def __getitem__(self, key):
        return 4

    # 实现迭代器方法
    def __iter__(self):
        return iter(["a", "b"])

    # 实现长度方法
    def __len__(self):
        return 2

# 测试类 TestPPrintThing
class TestPPrintThing:
    # 测试 pprint_thing 函数对二进制类型的处理
    def test_repr_binary_type(self):
        # 获取 ASCII 字母
        letters = string.ascii_letters
        try:
            # 尝试以指定编码创建原始字节串
            raw = bytes(letters, encoding=cf.get_option("display.encoding"))
        except TypeError:
            # 如果失败，则使用默认编码创建原始字节串
            raw = bytes(letters)
        # 将字节串转换为 UTF-8 编码的字符串
        b = str(raw.decode("utf-8"))
        # 测试以带引号字符串形式打印 b 的结果
        res = printing.pprint_thing(b, quote_strings=True)
        assert res == repr(b)
        # 测试以非带引号字符串形式打印 b 的结果
        res = printing.pprint_thing(b, quote_strings=False)
        assert res == b

    # 测试 pprint_thing 函数对于序列项数量限制的遵守
    def test_repr_obeys_max_seq_limit(self):
        # 在指定最大序列项数为 2000 的上下文中测试
        with cf.option_context("display.max_seq_items", 2000):
            assert len(printing.pprint_thing(list(range(1000)))) > 1000

        # 在指定最大序列项数为 5 的上下文中测试
        with cf.option_context("display.max_seq_items", 5):
            assert len(printing.pprint_thing(list(range(1000)))) < 100

        # 在指定最大序列项数为 1 的上下文中测试
        with cf.option_context("display.max_seq_items", 1):
            assert len(printing.pprint_thing(list(range(1000)))) < 9

    # 测试 pprint_thing 函数对集合的处理
    def test_repr_set(self):
        assert printing.pprint_thing({1}) == "{1}"

    # 测试 pprint_thing 函数对字典的处理
    def test_repr_dict(self):
        assert printing.pprint_thing({"a": 4, "b": 4}) == "{'a': 4, 'b': 4}"

    # 测试 pprint_thing 函数对自定义映射对象的处理
    def test_repr_mapping(self):
        assert printing.pprint_thing(MyMapping()) == "{'a': 4, 'b': 4}"

# 测试类 TestFormatBase
class TestFormatBase:
    # 测试 adjoin 函数的功能
    def test_adjoin(self):
        # 定义测试数据和期望的输出结果
        data = [["a", "b", "c"], ["dd", "ee", "ff"], ["ggg", "hhh", "iii"]]
        expected = "a  dd  ggg\nb  ee  hhh\nc  ff  iii"

        # 调用 adjoin 函数，将数据连接为字符串
        adjoined = printing.adjoin(2, *data)

        # 断言连接后的结果与预期的输出结果相等
        assert adjoined == expected

    # 测试 adjoin 函数处理包含 Unicode 字符的情况
    def test_adjoin_unicode(self):
        # 定义包含 Unicode 字符的测试数据和期望的输出结果
        data = [["あ", "b", "c"], ["dd", "ええ", "ff"], ["ggg", "hhh", "いいい"]]
        expected = "あ  dd  ggg\nb   ええ  hhh\nc   ff    いいい"
        
        # 调用 adjoin 函数，将数据连接为字符串
        adjoined = printing.adjoin(2, *data)
        
        # 断言连接后的结果与预期的输出结果相等
        assert adjoined == expected
        
        # 测试 _EastAsianTextAdjustment 类的功能
        adj = printing._EastAsianTextAdjustment()
        
        # 期望的输出结果
        expected = """あ       dd         ggg
b        ええ       hhh
c        ff         いいい"""
        
        # 调用 adjoin 方法，将数据连接为字符串
        adjoined = adj.adjoin(2, *data)
        
        # 断言连接后的结果与预期的输出结果相等
        assert adjoined == expected
        
        # 检查每一列的长度是否符合预期
        cols = adjoined.split("\n")
        assert adj.len(cols[0]) == 13
        assert adj.len(cols[1]) == 13
        assert adj.len(cols[2]) == 16

        expected = """あ       dd         ggg
b        ええ       hhh
        
        # 断言连接后的结果与预期的输出结果相等
        assert adjoined == expected
    def test_ambiguous_width(self):
        # 创建一个 _EastAsianTextAdjustment 的实例 adj
        adj = printing._EastAsianTextAdjustment()
        
        # 断言单个字符串长度为 4
        assert adj.len("¡¡ab") == 4

        # 在上下文中设置 unicode.ambiguous_as_wide 为 True
        with cf.option_context("display.unicode.ambiguous_as_wide", True):
            # 创建一个新的 _EastAsianTextAdjustment 的实例 adj
            adj = printing._EastAsianTextAdjustment()
            # 断言单个字符串长度为 6
            assert adj.len("¡¡ab") == 6

        # 定义数据和预期输出
        data = [["あ", "b", "c"], ["dd", "ええ", "ff"], ["ggg", "¡¡ab", "いいい"]]
        expected = "あ  dd    ggg \nb   ええ  ¡¡ab\nc   ff    いいい"
        # 调用 adj.adjoin 方法，将数据按照指定宽度 2 进行连接
        adjoined = adj.adjoin(2, *data)
        # 断言连接后的字符串与预期输出相等
        assert adjoined == expected
```