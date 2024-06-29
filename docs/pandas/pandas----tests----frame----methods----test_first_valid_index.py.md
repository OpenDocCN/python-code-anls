# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_first_valid_index.py`

```
"""
Includes test for last_valid_index.
"""

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
)


class TestFirstValidIndex:
    def test_first_valid_index_single_nan(self, frame_or_series):
        # GH#9752 Series/DataFrame should both return None, not raise
        # 创建包含单个 NaN 值的对象
        obj = frame_or_series([np.nan])

        # 断言首个有效索引应为 None
        assert obj.first_valid_index() is None
        # 断言空切片的首个有效索引应为 None
        assert obj.iloc[:0].first_valid_index() is None

    @pytest.mark.parametrize(
        "empty", [DataFrame(), Series(dtype=object), Series([], index=[], dtype=object)]
    )
    def test_first_valid_index_empty(self, empty):
        # GH#12800
        # 空 DataFrame 或 Series 应返回 None
        assert empty.last_valid_index() is None
        assert empty.first_valid_index() is None

    @pytest.mark.parametrize(
        "data,idx,expected_first,expected_last",
        [
            ({"A": [1, 2, 3]}, [1, 1, 2], 1, 2),
            ({"A": [1, 2, 3]}, [1, 2, 2], 1, 2),
            ({"A": [1, 2, 3, 4]}, ["d", "d", "d", "d"], "d", "d"),
            ({"A": [1, np.nan, 3]}, [1, 1, 2], 1, 2),
            ({"A": [np.nan, np.nan, 3]}, [1, 1, 2], 2, 2),
            ({"A": [1, np.nan, 3]}, [1, 2, 2], 1, 2),
        ],
    )
    def test_first_last_valid_frame(self, data, idx, expected_first, expected_last):
        # GH#21441
        # 创建 DataFrame，并验证首个和最后一个有效索引
        df = DataFrame(data, index=idx)
        assert expected_first == df.first_valid_index()
        assert expected_last == df.last_valid_index()

    @pytest.mark.parametrize(
        "index",
        [Index([str(i) for i in range(20)]), date_range("2020-01-01", periods=20)],
    )
    def test_first_last_valid(self, index):
        # GH#17400: no valid entries
        # 创建包含全部 NaN 的 DataFrame
        mat = np.random.default_rng(2).standard_normal(len(index))
        mat[:5] = np.nan
        mat[-5:] = np.nan

        frame = DataFrame({"foo": mat}, index=index)

        # 验证 DataFrame 和其 Series 的首个和最后一个有效索引
        assert frame.first_valid_index() == frame.index[5]
        assert frame.last_valid_index() == frame.index[-6]

        ser = frame["foo"]
        assert ser.first_valid_index() == frame.index[5]
        assert ser.last_valid_index() == frame.index[-6]

    @pytest.mark.parametrize(
        "index",
        [Index([str(i) for i in range(10)]), date_range("2020-01-01", periods=10)],
    )
    def test_first_last_valid_all_nan(self, index):
        # GH#17400: no valid entries
        # 创建包含全部 NaN 的 DataFrame
        frame = DataFrame(np.nan, columns=["foo"], index=index)

        # 断言全部 NaN 的 DataFrame 和其 Series 的首个和最后一个有效索引应为 None
        assert frame.last_valid_index() is None
        assert frame.first_valid_index() is None

        ser = frame["foo"]
        assert ser.first_valid_index() is None
        assert ser.last_valid_index() is None
```