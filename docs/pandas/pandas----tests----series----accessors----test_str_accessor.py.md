# `D:\src\scipysrc\pandas\pandas\tests\series\accessors\test_str_accessor.py`

```
import pytest  # 导入 pytest 模块

from pandas import Series  # 导入 pandas 库中的 Series 类
import pandas._testing as tm  # 导入 pandas 内部测试模块


class TestStrAccessor:
    def test_str_attribute(self):
        # GH#9068
        # 定义需要测试的字符串方法列表
        methods = ["strip", "rstrip", "lstrip"]
        # 创建一个包含字符串的 Series 对象
        ser = Series([" jack", "jill ", " jesse ", "frank"])
        # 遍历每个方法，生成预期结果的 Series 对象，并使用测试工具函数验证结果是否相等
        for method in methods:
            expected = Series([getattr(str, method)(x) for x in ser.values])
            tm.assert_series_equal(getattr(Series.str, method)(ser.str), expected)

        # 使用 .str 访问器仅对字符串值有效
        # 创建一个包含整数的 Series 对象
        ser = Series(range(5))
        # 使用 pytest 断言检查调用非字符串 Series 对象上的 .str 访问器是否引发了预期的异常
        with pytest.raises(AttributeError, match="only use .str accessor"):
            ser.str.repeat(2)

    def test_str_accessor_updates_on_inplace(self):
        # 创建一个包含字符列表的 Series 对象
        ser = Series(list("abc"))
        # 执行 inplace 操作并接收返回值
        return_value = ser.drop([0], inplace=True)
        # 使用普通的字符串方法来检查 inplace 操作是否成功
        assert return_value is None
        # 使用 .str 访问器来验证 inplace 操作后的 Series 对象长度
        assert len(ser.str.lower()) == 2
```