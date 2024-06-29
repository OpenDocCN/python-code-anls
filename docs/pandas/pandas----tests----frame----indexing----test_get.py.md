# `D:\src\scipysrc\pandas\pandas\tests\frame\indexing\test_get.py`

```
import pytest  # 导入 pytest 库

from pandas import DataFrame  # 从 pandas 库中导入 DataFrame 类
import pandas._testing as tm  # 导入 pandas 测试工具模块

class TestGet:  # 定义测试类 TestGet
    def test_get(self, float_frame):  # 定义测试方法 test_get，接收 float_frame 参数
        b = float_frame.get("B")  # 获取 float_frame 中列名为 "B" 的数据
        tm.assert_series_equal(b, float_frame["B"])  # 使用测试工具模块验证获取的数据与 float_frame["B"] 是否相等

        assert float_frame.get("foo") is None  # 断言获取不存在的列名 "foo" 返回 None
        tm.assert_series_equal(
            float_frame.get("foo", float_frame["B"]), float_frame["B"]
        )  # 使用测试工具模块验证获取不存在的列名 "foo" 返回默认值与 float_frame["B"] 是否相等

    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，定义参数化测试
        "columns, index",
        [
            [None, None],  # 第一组参数：列为 None，索引为 None
            [list("AB"), None],  # 第二组参数：列为 ["A", "B"]，索引为 None
            [list("AB"), range(3)],  # 第三组参数：列为 ["A", "B"]，索引为 range(3)
        ],
    )
    def test_get_none(self, columns, index):  # 定义测试方法 test_get_none，接收 columns 和 index 参数
        # see gh-5652
        assert DataFrame(columns=columns, index=index).get(None) is None  # 断言在给定的 DataFrame 中使用 get(None) 返回 None
```