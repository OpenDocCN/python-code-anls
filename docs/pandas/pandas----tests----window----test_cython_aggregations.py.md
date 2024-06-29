# `D:\src\scipysrc\pandas\pandas\tests\window\test_cython_aggregations.py`

```
# 导入 functools 模块中的 partial 函数
# 导入 sys 模块
import sys

# 导入 numpy 库，并用 np 作为别名
import numpy as np

# 导入 pytest 库
import pytest

# 导入 pandas._libs.window.aggregations 模块中的 window_aggregations 对象
import pandas._libs.window.aggregations as window_aggregations

# 从 pandas 库导入 Series 类
from pandas import Series

# 导入 pandas._testing 模块并用 tm 作为别名
import pandas._testing as tm


# 定义一个函数 _get_rolling_aggregations
def _get_rolling_aggregations():
    # 定义一个包含名称和函数对的列表 named_roll_aggs
    # 每个函数有如下签名:
    # (const float64_t[:] values, ndarray[int64_t] start,
    #  ndarray[int64_t] end, int64_t minp) -> np.ndarray
    named_roll_aggs = (
        [
            ("roll_sum", window_aggregations.roll_sum),
            ("roll_mean", window_aggregations.roll_mean),
        ]
        + [
            (f"roll_var({ddof})", partial(window_aggregations.roll_var, ddof=ddof))
            for ddof in [0, 1]
        ]
        + [
            ("roll_skew", window_aggregations.roll_skew),
            ("roll_kurt", window_aggregations.roll_kurt),
            ("roll_median_c", window_aggregations.roll_median_c),
            ("roll_max", window_aggregations.roll_max),
            ("roll_min", window_aggregations.roll_min),
        ]
        + [
            (
                f"roll_quantile({quantile},{interpolation})",
                partial(
                    window_aggregations.roll_quantile,
                    quantile=quantile,
                    interpolation=interpolation,
                ),
            )
            for quantile in [0.0001, 0.5, 0.9999]
            for interpolation in window_aggregations.interpolation_types
        ]
        + [
            (
                f"roll_rank({percentile},{method},{ascending})",
                partial(
                    window_aggregations.roll_rank,
                    percentile=percentile,
                    method=method,
                    ascending=ascending,
                ),
            )
            for percentile in [True, False]
            for method in window_aggregations.rolling_rank_tiebreakers.keys()
            for ascending in [True, False]
        ]
    )
    # 使用 zip 解压 named_roll_aggs 列表为一个包含两个元组的列表 unzipped
    unzipped = list(zip(*named_roll_aggs))
    # 返回一个字典，包含 ids 键和 params 键，分别对应于 unzipped 的第一个和第二个元组元素
    return {"ids": unzipped[0], "params": unzipped[1]}


# 调用 _get_rolling_aggregations 函数并将结果赋值给 _rolling_aggregations 变量
_rolling_aggregations = _get_rolling_aggregations()


# 定义一个 pytest fixture，参数为 _rolling_aggregations["params"]，标识为 _rolling_aggregations["ids"]
@pytest.fixture(
    params=_rolling_aggregations["params"], ids=_rolling_aggregations["ids"]
)
# 定义 rolling_aggregation 函数，返回 request.param
def rolling_aggregation(request):
    """Make a rolling aggregation function as fixture."""
    return request.param


# 定义一个测试函数 test_rolling_aggregation_boundary_consistency，参数为 rolling_aggregation fixture
def test_rolling_aggregation_boundary_consistency(rolling_aggregation):
    # GH-45647
    # 定义变量 minp, step, width, size, selection
    minp, step, width, size, selection = 0, 1, 3, 11, [2, 7]
    # 创建一个包含 1 到 size 的浮点数值的 ndarray values
    values = np.arange(1, 1 + size, dtype=np.float64)
    # 创建一个包含 width 到 size 的整数值的 ndarray end
    end = np.arange(width, size, step, dtype=np.int64)
    # 创建一个 start 数组，其值为 end 减去 width
    start = end - width
    # 创建一个包含 selection 的整数值的 ndarray selarr
    selarr = np.array(selection, dtype=np.int32)
    # 调用 rolling_aggregation 函数，计算结果并创建 Series 对象 result
    result = Series(rolling_aggregation(values, start[selarr], end[selarr], minp))
    # 调用 rolling_aggregation 函数，计算期望值并创建 Series 对象 expected
    expected = Series(rolling_aggregation(values, start, end, minp)[selarr])
    # 使用 pandas._testing 模块中的 assert_equal 函数，比较 expected 和 result
    tm.assert_equal(expected, result)


# 定义一个测试函数 test_rolling_aggregation_with_unused_elements，参数为 rolling_aggregation fixture
def test_rolling_aggregation_with_unused_elements(rolling_aggregation):
    # GH-45647
    # 此处省略部分代码
    pass  # Placeholder for future implementation
    minp, width = 0, 5  # 初始化最小段落(minp)和窗口宽度(width)，至少需要4个值以计算峰度
    size = 2 * width + 5  # 计算数组总长度
    values = np.arange(1, size + 1, dtype=np.float64)  # 创建包含1到size的浮点数数组
    values[width : width + 2] = sys.float_info.min  # 将索引在width到width+1之间的值设为系统浮点数最小值
    values[width + 2] = np.nan  # 将索引为width+2的值设为NaN
    values[width + 3 : width + 5] = sys.float_info.max  # 将索引在width+3到width+4之间的值设为系统浮点数最大值
    start = np.array([0, size - width], dtype=np.int64)  # 创建起始位置数组，包含0和(size-width-1)
    end = np.array([width, size], dtype=np.int64)  # 创建结束位置数组，包含width和size-1
    loc = np.array(
        [j for i in range(len(start)) for j in range(start[i], end[i])],
        dtype=np.int32,
    )  # 创建位置数组loc，包含从start到end的所有索引值
    result = Series(rolling_aggregation(values, start, end, minp))  # 使用rolling_aggregation函数计算values的滚动聚合结果，并封装为Series对象
    compact_values = np.array(values[loc], dtype=np.float64)  # 从values中提取位置数组loc对应的值，封装为紧凑数组
    compact_start = np.arange(0, len(start) * width, width, dtype=np.int64)  # 创建紧凑起始位置数组，步长为width
    compact_end = compact_start + width  # 创建紧凑结束位置数组，为紧凑起始位置数组加上width
    expected = Series(
        rolling_aggregation(compact_values, compact_start, compact_end, minp)
    )  # 使用rolling_aggregation函数计算紧凑数组的滚动聚合结果，并封装为Series对象，作为期望结果
    assert np.isfinite(expected.values).all(), "Not all expected values are finite"  # 断言期望结果的所有值都是有限的，否则抛出错误信息
    tm.assert_equal(expected, result)  # 使用tm模块的assert_equal函数断言期望结果与计算结果相等
```