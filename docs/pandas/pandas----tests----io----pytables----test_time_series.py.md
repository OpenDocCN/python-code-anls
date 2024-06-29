# `D:\src\scipysrc\pandas\pandas\tests\io\pytables\test_time_series.py`

```
# 导入 datetime 模块，用于处理日期和时间
import datetime

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 导入 pytest 模块
import pytest

# 从 pandas 库中导入以下对象：
# - DataFrame：用于操作二维表格数据
# - DatetimeIndex：用于处理时间序列索引
# - Series：用于操作一维数据结构
# - _testing as tm：导入 _testing 模块并使用 tm 别名
# - date_range：生成指定日期范围的时间序列
# - period_range：生成指定周期范围的时间序列
from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    _testing as tm,
    date_range,
    period_range,
)

# 从 pandas 的测试模块中导入 ensure_clean_store 函数
from pandas.tests.io.pytables.common import ensure_clean_store

# 设置 pytest 的标记为 single_cpu
pytestmark = pytest.mark.single_cpu


# 使用 pytest 的 parametrize 装饰器，定义测试函数 test_store_datetime_fractional_secs
@pytest.mark.parametrize("unit", ["us", "ns"])
def test_store_datetime_fractional_secs(setup_path, unit):
    # 创建一个特定的 datetime 对象，表示 2012 年 1 月 2 日 3 时 4 分 5 秒 123456 微秒
    dt = datetime.datetime(2012, 1, 2, 3, 4, 5, 123456)
    
    # 使用 DatetimeIndex 构造函数创建一个时间索引对象 dti，指定 dtype 为特定的单位（"us" 或 "ns"）
    dti = DatetimeIndex([dt], dtype=f"M8[{unit}]")
    
    # 创建一个 Series 对象，包含单个值 0，其索引为 dti
    series = Series([0], index=dti)
    
    # 使用 ensure_clean_store 函数作为上下文管理器，传入 setup_path 参数
    with ensure_clean_store(setup_path) as store:
        # 将 series 存储在 store 中，键名为 "a"
        store["a"] = series
        
        # 断言存储在 store 中的 "a" 的第一个索引值等于原始的 dt
        assert store["a"].index[0] == dt


# 使用 pytest 的 filterwarnings 装饰器，忽略特定警告信息
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_tseries_indices_series(setup_path):
    # 使用 ensure_clean_store 函数作为上下文管理器，传入 setup_path 参数
    with ensure_clean_store(setup_path) as store:
        # 创建一个日期范围索引 idx，从 "2020-01-01" 开始，包含 10 个日期
        idx = date_range("2020-01-01", periods=10)
        
        # 创建一个 Series 对象 ser，其值为随机生成的正态分布数据，索引为 idx
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        
        # 将 ser 存储在 store 中，键名为 "a"
        store["a"] = ser
        
        # 从 store 中获取存储的结果，赋值给 result
        result = store["a"]
        
        # 使用 _testing 模块中的 assert_series_equal 函数断言 result 与 ser 相等
        tm.assert_series_equal(result, ser)
        
        # 断言 result 的索引频率与 ser 的索引频率相等
        assert result.index.freq == ser.index.freq
        
        # 使用 _testing 模块中的 assert_class_equal 函数断言 result 的索引类型与 ser 的索引类型相等，指定对象为 "series index"
        tm.assert_class_equal(result.index, ser.index, obj="series index")
        
        # 创建一个周期范围索引 idx，从 "2020-01-01" 开始，包含 10 个周期，频率为 "D"（每天）
        idx = period_range("2020-01-01", periods=10, freq="D")
        
        # 创建一个 Series 对象 ser，其值为随机生成的正态分布数据，索引为 idx
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        
        # 将 ser 存储在 store 中，键名为 "a"
        store["a"] = ser
        
        # 从 store 中获取存储的结果，赋值给 result
        result = store["a"]
        
        # 使用 _testing 模块中的 assert_series_equal 函数断言 result 与 ser 相等
        tm.assert_series_equal(result, ser)
        
        # 断言 result 的索引频率与 ser 的索引频率相等
        assert result.index.freq == ser.index.freq
        
        # 使用 _testing 模块中的 assert_class_equal 函数断言 result 的索引类型与 ser 的索引类型相等，指定对象为 "series index"
        tm.assert_class_equal(result.index, ser.index, obj="series index")


# 使用 pytest 的 filterwarnings 装饰器，忽略特定警告信息
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_tseries_indices_frame(setup_path):
    # 使用 ensure_clean_store 函数作为上下文管理器，传入 setup_path 参数
    with ensure_clean_store(setup_path) as store:
        # 创建一个日期范围索引 idx，从 "2020-01-01" 开始，包含 10 个日期
        idx = date_range("2020-01-01", periods=10)
        
        # 创建一个 DataFrame 对象 df，其值为随机生成的正态分布数据，索引为 idx，列数为 3
        df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx)
        
        # 将 df 存储在 store 中，键名为 "a"
        store["a"] = df
        
        # 从 store 中获取存储的结果，赋值给 result
        result = store["a"]
        
        # 使用 _testing 模块中的 assert_frame_equal 函数断言 result 与 df 相等
        tm.assert_frame_equal(result, df)
        
        # 断言 result 的索引频率与 df 的索引频率相等
        assert result.index.freq == df.index.freq
        
        # 使用 _testing 模块中的 assert_class_equal 函数断言 result 的索引类型与 df 的索引类型相等，指定对象为 "dataframe index"
        tm.assert_class_equal(result.index, df.index, obj="dataframe index")
        
        # 创建一个周期范围索引 idx，从 "2020-01-01" 开始，包含 10 个周期，频率为 "D"（每天）
        idx = period_range("2020-01-01", periods=10, freq="D")
        
        # 创建一个 DataFrame 对象 df，其值为随机生成的正态分布数据，索引为 idx，列数为 3
        df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), idx)
        
        # 将 df 存储在 store 中，键名为 "a"
        store["a"] = df
        
        # 从 store 中获取存储的结果，赋值给 result
        result = store["a"]
        
        # 使用 _testing 模块中的 assert_frame_equal 函数断言 result 与 df 相等
        tm.assert_frame_equal(result, df)
        
        # 断言 result 的索引频率与 df 的索引频率相等
        assert result.index.freq == df.index.freq
        
        # 使用 _testing 模块中的 assert_class_equal 函数断言 result 的索引类型与 df 的索引类型相等，指定对象为 "dataframe index"
        tm.assert_class_equal(result.index, df.index, obj="dataframe index")
```