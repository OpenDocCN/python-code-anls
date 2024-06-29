# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_searchsorted.py`

```
    # 导入所需的库
    import numpy as np
    import pytest
    
    # 导入 pandas 库并从中导入 Series、Timestamp 和 date_range 函数
    import pandas as pd
    from pandas import (
        Series,
        Timestamp,
        date_range,
    )
    
    # 导入 pandas 的测试模块
    import pandas._testing as tm
    
    # 从 pandas 的 api.types 模块中导入 is_scalar 函数
    from pandas.api.types import is_scalar
    
    # 定义一个测试类 TestSeriesSearchSorted，用于测试 Series 的 searchsorted 方法
    class TestSeriesSearchSorted:
        
        # 定义测试方法 test_searchsorted，测试 searchsorted 方法的基本用法
        def test_searchsorted(self):
            # 创建一个 Series 对象
            ser = Series([1, 2, 3])
            
            # 调用 searchsorted 方法，寻找值 1 的位置（左侧）
            result = ser.searchsorted(1, side="left")
            # 断言返回结果是标量
            assert is_scalar(result)
            # 断言返回的位置是 0
            assert result == 0
            
            # 调用 searchsorted 方法，寻找值 1 的位置（右侧）
            result = ser.searchsorted(1, side="right")
            # 断言返回结果是标量
            assert is_scalar(result)
            # 断言返回的位置是 1
            assert result == 1
        
        # 定义测试方法 test_searchsorted_numeric_dtypes_scalar，测试数值类型和标量值的搜索
        def test_searchsorted_numeric_dtypes_scalar(self):
            # 创建一个 Series 对象
            ser = Series([1, 2, 90, 1000, 3e9])
            
            # 调用 searchsorted 方法，寻找值 30 的位置
            res = ser.searchsorted(30)
            # 断言返回结果是标量
            assert is_scalar(res)
            # 断言返回的位置是 2
            assert res == 2
            
            # 调用 searchsorted 方法，寻找值列表 [30] 的位置
            res = ser.searchsorted([30])
            # 创建预期的 numpy 数组
            exp = np.array([2], dtype=np.intp)
            # 使用测试模块中的函数来断言两个数组相等
            tm.assert_numpy_array_equal(res, exp)
        
        # 定义测试方法 test_searchsorted_numeric_dtypes_vector，测试数值类型和向量值的搜索
        def test_searchsorted_numeric_dtypes_vector(self):
            # 创建一个 Series 对象
            ser = Series([1, 2, 90, 1000, 3e9])
            
            # 调用 searchsorted 方法，寻找值列表 [91, 2e6] 的位置
            res = ser.searchsorted([91, 2e6])
            # 创建预期的 numpy 数组
            exp = np.array([3, 4], dtype=np.intp)
            # 使用测试模块中的函数来断言两个数组相等
            tm.assert_numpy_array_equal(res, exp)
        
        # 定义测试方法 test_searchsorted_datetime64_scalar，测试 datetime64 类型和标量值的搜索
        def test_searchsorted_datetime64_scalar(self):
            # 创建一个 Series 对象，包含日期范围
            ser = Series(date_range("20120101", periods=10, freq="2D"))
            # 创建要搜索的时间戳对象
            val = Timestamp("20120102")
            # 调用 searchsorted 方法，寻找值为 val 的位置
            res = ser.searchsorted(val)
            # 断言返回结果是标量
            assert is_scalar(res)
            # 断言返回的位置是 1
            assert res == 1
        
        # 定义测试方法 test_searchsorted_datetime64_scalar_mixed_timezones，测试混合时区的 datetime64 类型和标量值的搜索
        def test_searchsorted_datetime64_scalar_mixed_timezones(self):
            # 创建一个 Series 对象，包含日期范围和时区
            ser = Series(date_range("20120101", periods=10, freq="2D", tz="UTC"))
            # 创建要搜索的带时区的时间戳对象
            val = Timestamp("20120102", tz="America/New_York")
            # 调用 searchsorted 方法，寻找值为 val 的位置
            res = ser.searchsorted(val)
            # 断言返回结果是标量
            assert is_scalar(res)
            # 断言返回的位置是 1
            assert res == 1
        
        # 定义测试方法 test_searchsorted_datetime64_list，测试 datetime64 类型和时间戳列表的搜索
        def test_searchsorted_datetime64_list(self):
            # 创建一个 Series 对象，包含日期范围
            ser = Series(date_range("20120101", periods=10, freq="2D"))
            # 创建要搜索的时间戳列表
            vals = [Timestamp("20120102"), Timestamp("20120104")]
            # 调用 searchsorted 方法，寻找值为 vals 的位置
            res = ser.searchsorted(vals)
            # 创建预期的 numpy 数组
            exp = np.array([1, 2], dtype=np.intp)
            # 使用测试模块中的函数来断言两个数组相等
            tm.assert_numpy_array_equal(res, exp)
        
        # 定义测试方法 test_searchsorted_sorter，测试使用排序器进行搜索
        def test_searchsorted_sorter(self):
            # 创建一个 Series 对象
            ser = Series([3, 1, 2])
            # 使用排序器 np.argsort(ser) 来调用 searchsorted 方法，寻找值列表 [0, 3] 的位置
            res = ser.searchsorted([0, 3], sorter=np.argsort(ser))
            # 创建预期的 numpy 数组
            exp = np.array([0, 2], dtype=np.intp)
            # 使用测试模块中的函数来断言两个数组相等
            tm.assert_numpy_array_equal(res, exp)
        
        # 定义测试方法 test_searchsorted_dataframe_fail，测试传入 DataFrame 时的错误处理
        def test_searchsorted_dataframe_fail(self):
            # 创建一个 Series 对象
            ser = Series([1, 2, 3, 4, 5])
            # 创建一个 DataFrame 对象，会引发错误
            vals = pd.DataFrame([[1, 2], [3, 4]])
            # 设置期望的错误消息
            msg = "Value must be 1-D array-like or scalar, DataFrame is not supported"
            # 使用 pytest 来断言抛出特定类型和消息的异常
            with pytest.raises(ValueError, match=msg):
                ser.searchsorted(vals)
```