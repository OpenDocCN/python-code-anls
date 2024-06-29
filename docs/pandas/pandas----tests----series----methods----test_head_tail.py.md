# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_head_tail.py`

```
# 导入 pandas 测试模块，命名为 tm
import pandas._testing as tm

# 定义一个测试函数 test_head_tail，接受一个字符串系列作为参数
def test_head_tail(string_series):
    # 断言：验证 series 的前5个元素与 string_series[:5] 是否相等
    tm.assert_series_equal(string_series.head(), string_series[:5])
    # 断言：验证 series 的前0个元素（空）与 string_series[0:0] 是否相等
    tm.assert_series_equal(string_series.head(0), string_series[0:0])
    # 断言：验证 series 的后5个元素与 string_series[-5:] 是否相等
    tm.assert_series_equal(string_series.tail(), string_series[-5:])
    # 断言：验证 series 的后0个元素（空）与 string_series[0:0] 是否相等
    tm.assert_series_equal(string_series.tail(0), string_series[0:0])
```