# `D:\src\scipysrc\pandas\pandas\tests\test_flags.py`

```
import pytest  # 导入 pytest 模块

import pandas as pd  # 导入 pandas 库，并简称为 pd


class TestFlags:  # 定义测试类 TestFlags
    def test_equality(self):  # 定义测试方法 test_equality
        a = pd.DataFrame().set_flags(allows_duplicate_labels=True).flags  # 创建 DataFrame 对象，设置 allows_duplicate_labels 标志为 True，获取其 flags 属性
        b = pd.DataFrame().set_flags(allows_duplicate_labels=False).flags  # 创建 DataFrame 对象，设置 allows_duplicate_labels 标志为 False，获取其 flags 属性

        assert a == a  # 断言 a 等于自身
        assert b == b  # 断言 b 等于自身
        assert a != b  # 断言 a 不等于 b
        assert a != 2  # 断言 a 不等于整数 2

    def test_set(self):  # 定义测试方法 test_set
        df = pd.DataFrame().set_flags(allows_duplicate_labels=True)  # 创建 DataFrame 对象，设置 allows_duplicate_labels 标志为 True
        a = df.flags  # 获取 DataFrame 对象的 flags 属性
        a.allows_duplicate_labels = False  # 设置 flags 属性中的 allows_duplicate_labels 为 False
        assert a.allows_duplicate_labels is False  # 断言 allows_duplicate_labels 属性为 False
        a["allows_duplicate_labels"] = True  # 使用字典方式设置 allows_duplicate_labels 属性为 True
        assert a.allows_duplicate_labels is True  # 断言 allows_duplicate_labels 属性为 True

    def test_repr(self):  # 定义测试方法 test_repr
        a = repr(pd.DataFrame({"A"}).set_flags(allows_duplicate_labels=True).flags)  # 创建 DataFrame 对象，并设置 allows_duplicate_labels 标志为 True，获取其 flags 属性并进行 repr 处理
        assert a == "<Flags(allows_duplicate_labels=True)>"  # 断言 repr 结果符合预期格式
        a = repr(pd.DataFrame({"A"}).set_flags(allows_duplicate_labels=False).flags)  # 创建 DataFrame 对象，并设置 allows_duplicate_labels 标志为 False，获取其 flags 属性并进行 repr 处理
        assert a == "<Flags(allows_duplicate_labels=False)>"  # 断言 repr 结果符合预期格式

    def test_obj_ref(self):  # 定义测试方法 test_obj_ref
        df = pd.DataFrame()  # 创建空的 DataFrame 对象
        flags = df.flags  # 获取 DataFrame 对象的 flags 属性
        del df  # 删除 DataFrame 对象的引用
        with pytest.raises(ValueError, match="object has been deleted"):  # 使用 pytest 断言捕获 ValueError 异常，并匹配特定错误信息
            flags.allows_duplicate_labels = True  # 尝试设置 flags 属性中的 allows_duplicate_labels

    def test_getitem(self):  # 定义测试方法 test_getitem
        df = pd.DataFrame()  # 创建空的 DataFrame 对象
        flags = df.flags  # 获取 DataFrame 对象的 flags 属性
        assert flags["allows_duplicate_labels"] is True  # 断言 flags 属性中 allows_duplicate_labels 的值为 True
        flags["allows_duplicate_labels"] = False  # 使用字典方式设置 flags 属性中 allows_duplicate_labels 的值为 False
        assert flags["allows_duplicate_labels"] is False  # 断言 flags 属性中 allows_duplicate_labels 的值为 False

        with pytest.raises(KeyError, match="a"):  # 使用 pytest 断言捕获 KeyError 异常，并匹配特定错误信息
            flags["a"]  # 尝试访问 flags 属性中不存在的键 "a"

        with pytest.raises(ValueError, match="a"):  # 使用 pytest 断言捕获 ValueError 异常，并匹配特定错误信息
            flags["a"] = 10  # 尝试使用字典方式设置 flags 属性中不存在的键 "a" 的值为 10
```