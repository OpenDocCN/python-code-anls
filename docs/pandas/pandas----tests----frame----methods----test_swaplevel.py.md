# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_swaplevel.py`

```
import pytest  # 导入 pytest 测试框架

from pandas import DataFrame  # 从 pandas 库中导入 DataFrame 类
import pandas._testing as tm  # 导入 pandas 内部测试模块


class TestSwaplevel:  # 定义测试类 TestSwaplevel
    def test_swaplevel(self, multiindex_dataframe_random_data):  # 定义测试方法 test_swaplevel，接受一个名为 multiindex_dataframe_random_data 的参数
        frame = multiindex_dataframe_random_data  # 将参数赋值给变量 frame

        swapped = frame["A"].swaplevel()  # 对 frame 中的 "A" 列执行 swaplevel 操作，默认交换两层索引
        swapped2 = frame["A"].swaplevel(0)  # 对 "A" 列执行 swaplevel 操作，指定要交换的第一层索引
        swapped3 = frame["A"].swaplevel(0, 1)  # 对 "A" 列执行 swaplevel 操作，指定要交换的两层索引位置
        swapped4 = frame["A"].swaplevel("first", "second")  # 对 "A" 列执行 swaplevel 操作，指定要交换的两层索引名称
        assert not swapped.index.equals(frame.index)  # 断言：交换后的索引不等于原始 frame 的索引
        tm.assert_series_equal(swapped, swapped2)  # 使用 pandas._testing 模块的 assert_series_equal 方法断言 swapped 和 swapped2 相等
        tm.assert_series_equal(swapped, swapped3)  # 使用 pandas._testing 模块的 assert_series_equal 方法断言 swapped 和 swapped3 相等
        tm.assert_series_equal(swapped, swapped4)  # 使用 pandas._testing 模块的 assert_series_equal 方法断言 swapped 和 swapped4 相等

        back = swapped.swaplevel()  # 对 swapped 执行 swaplevel 操作，默认交换两层索引
        back2 = swapped.swaplevel(0)  # 对 swapped 执行 swaplevel 操作，指定要交换的第一层索引
        back3 = swapped.swaplevel(0, 1)  # 对 swapped 执行 swaplevel 操作，指定要交换的两层索引位置
        back4 = swapped.swaplevel("second", "first")  # 对 swapped 执行 swaplevel 操作，指定要交换的两层索引名称
        assert back.index.equals(frame.index)  # 断言：交换后的索引等于原始 frame 的索引
        tm.assert_series_equal(back, back2)  # 使用 pandas._testing 模块的 assert_series_equal 方法断言 back 和 back2 相等
        tm.assert_series_equal(back, back3)  # 使用 pandas._testing 模块的 assert_series_equal 方法断言 back 和 back3 相等
        tm.assert_series_equal(back, back4)  # 使用 pandas._testing 模块的 assert_series_equal 方法断言 back 和 back4 相等

        ft = frame.T  # 取 frame 的转置
        swapped = ft.swaplevel("first", "second", axis=1)  # 对 frame 的转置执行 swaplevel 操作，指定要交换的两层索引名称，并指定操作轴为列
        exp = frame.swaplevel("first", "second").T  # 对 frame 执行 swaplevel 操作，然后再取转置
        tm.assert_frame_equal(swapped, exp)  # 使用 pandas._testing 模块的 assert_frame_equal 方法断言 swapped 和 exp 相等

        msg = "Can only swap levels on a hierarchical axis."  # 定义异常信息字符串
        with pytest.raises(TypeError, match=msg):  # 使用 pytest 的 raises 方法验证是否抛出 TypeError 异常，且异常信息匹配 msg
            DataFrame(range(3)).swaplevel()  # 创建一个新的 DataFrame 对象，并尝试对其执行 swaplevel 操作
```