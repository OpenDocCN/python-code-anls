# `D:\src\scipysrc\pandas\pandas\tests\arrays\categorical\test_subclass.py`

```
from pandas import Categorical  # 导入 pandas 中的 Categorical 类
import pandas._testing as tm  # 导入 pandas 内部测试模块


class SubclassedCategorical(Categorical):  # 定义一个继承自 Categorical 的子类 SubclassedCategorical
    pass  # 空的类定义，未添加额外功能


class TestCategoricalSubclassing:  # 定义一个测试类 TestCategoricalSubclassing
    def test_constructor(self):  # 定义测试构造函数的方法
        sc = SubclassedCategorical(["a", "b", "c"])  # 使用 SubclassedCategorical 类创建对象 sc
        assert isinstance(sc, SubclassedCategorical)  # 断言 sc 是 SubclassedCategorical 类的实例
        tm.assert_categorical_equal(sc, Categorical(["a", "b", "c"]))  # 使用测试模块 tm 断言 sc 与 Categorical(["a", "b", "c"]) 相等

    def test_from_codes(self):  # 定义测试 from_codes 方法的方法
        sc = SubclassedCategorical.from_codes([1, 0, 2], ["a", "b", "c"])  # 使用 from_codes 方法创建 SubclassedCategorical 对象 sc
        assert isinstance(sc, SubclassedCategorical)  # 断言 sc 是 SubclassedCategorical 类的实例
        exp = Categorical.from_codes([1, 0, 2], ["a", "b", "c"])  # 创建期望的 Categorical 对象 exp
        tm.assert_categorical_equal(sc, exp)  # 使用测试模块 tm 断言 sc 与 exp 相等

    def test_map(self):  # 定义测试 map 方法的方法
        sc = SubclassedCategorical(["a", "b", "c"])  # 使用 SubclassedCategorical 类创建对象 sc
        res = sc.map(lambda x: x.upper(), na_action=None)  # 使用 map 方法对 sc 中的元素执行 lambda 函数，na_action 设置为 None
        assert isinstance(res, SubclassedCategorical)  # 断言 res 是 SubclassedCategorical 类的实例
        exp = Categorical(["A", "B", "C"])  # 创建期望的 Categorical 对象 exp
        tm.assert_categorical_equal(res, exp)  # 使用测试模块 tm 断言 res 与 exp 相等
```