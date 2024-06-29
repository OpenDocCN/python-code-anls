# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_reorder_levels.py`

```
    # 导入 numpy 库并使用别名 np
    import numpy as np
    # 导入 pytest 库用于单元测试
    import pytest
    
    # 从 pandas 库中导入 DataFrame 和 MultiIndex 类
    from pandas import (
        DataFrame,
        MultiIndex,
    )
    # 导入 pandas 测试工具集
    import pandas._testing as tm
    
    # 定义一个测试类 TestReorderLevels
    class TestReorderLevels:
        
        # 定义测试方法 test_reorder_levels，接受一个名为 frame_or_series 的参数
        def test_reorder_levels(self, frame_or_series):
            # 创建一个 MultiIndex 对象作为索引
            index = MultiIndex(
                levels=[["bar"], ["one", "two", "three"], [0, 1]],
                codes=[[0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1]],
                names=["L0", "L1", "L2"],
            )
            # 创建一个 DataFrame 对象 df，使用 MultiIndex 作为索引，包含两列 A 和 B
            df = DataFrame({"A": np.arange(6), "B": np.arange(6)}, index=index)
            # 使用测试工具集中的 get_obj 函数获取 obj 对象
            obj = tm.get_obj(df, frame_or_series)
    
            # 重新排序索引，顺序不变
            result = obj.reorder_levels([0, 1, 2])
            # 断言 obj 和 result 相等
            tm.assert_equal(obj, result)
    
            # 重新排序索引，使用标签
            result = obj.reorder_levels(["L0", "L1", "L2"])
            # 断言 obj 和 result 相等
            tm.assert_equal(obj, result)
    
            # 旋转索引，按位置重新排序
            result = obj.reorder_levels([1, 2, 0])
            # 创建期望的 MultiIndex 对象 e_idx
            e_idx = MultiIndex(
                levels=[["one", "two", "three"], [0, 1], ["bar"]],
                codes=[[0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0]],
                names=["L1", "L2", "L0"],
            )
            # 创建期望的 DataFrame 对象 expected，包含两列 A 和 B
            expected = DataFrame({"A": np.arange(6), "B": np.arange(6)}, index=e_idx)
            # 使用测试工具集中的 get_obj 函数获取 expected 对象
            expected = tm.get_obj(expected, frame_or_series)
            # 断言 result 和 expected 相等
            tm.assert_equal(result, expected)
    
            # 重新排序索引，所有级别相同
            result = obj.reorder_levels([0, 0, 0])
            # 创建期望的 MultiIndex 对象 e_idx
            e_idx = MultiIndex(
                levels=[["bar"], ["bar"], ["bar"]],
                codes=[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                names=["L0", "L0", "L0"],
            )
            # 创建期望的 DataFrame 对象 expected，包含两列 A 和 B
            expected = DataFrame({"A": np.arange(6), "B": np.arange(6)}, index=e_idx)
            # 使用测试工具集中的 get_obj 函数获取 expected 对象
            expected = tm.get_obj(expected, frame_or_series)
            # 断言 result 和 expected 相等
            tm.assert_equal(result, expected)
    
            # 重新排序索引，所有级别相同，使用标签
            result = obj.reorder_levels(["L0", "L0", "L0"])
            # 断言 result 和 expected 相等
            tm.assert_equal(result, expected)
    
        # 定义测试方法 test_reorder_levels_swaplevel_equivalence，接受一个名为 multiindex_year_month_day_dataframe_random_data 的参数
        def test_reorder_levels_swaplevel_equivalence(
            self, multiindex_year_month_day_dataframe_random_data
        ):
            # 获取参数对象 ymd
            ymd = multiindex_year_month_day_dataframe_random_data
    
            # 使用标签重新排序索引，期望结果与使用 swaplevel 方法等价
            result = ymd.reorder_levels(["month", "day", "year"])
            expected = ymd.swaplevel(0, 1).swaplevel(1, 2)
            # 断言 result 和 expected 相等
            tm.assert_frame_equal(result, expected)
    
            # 对 ymd 的列 'A' 使用标签重新排序索引，期望结果与使用 swaplevel 方法等价
            result = ymd["A"].reorder_levels(["month", "day", "year"])
            expected = ymd["A"].swaplevel(0, 1).swaplevel(1, 2)
            # 断言 result 和 expected 相等
            tm.assert_series_equal(result, expected)
    
            # 对 ymd 的转置进行标签重新排序索引，期望结果与使用 swaplevel 方法等价
            result = ymd.T.reorder_levels(["month", "day", "year"], axis=1)
            expected = ymd.T.swaplevel(0, 1, axis=1).swaplevel(1, 2, axis=1)
            # 断言 result 和 expected 相等
            tm.assert_frame_equal(result, expected)
    
            # 使用 pytest 引发 TypeError 异常，匹配错误消息 "hierarchical axis"
            with pytest.raises(TypeError, match="hierarchical axis"):
                ymd.reorder_levels([1, 2], axis=1)
    
            # 使用 pytest 引发 IndexError 异常，匹配错误消息 "Too many levels"
            with pytest.raises(IndexError, match="Too many levels"):
                ymd.index.reorder_levels([1, 2, 3])
```