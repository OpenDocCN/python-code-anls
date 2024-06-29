# `D:\src\scipysrc\pandas\pandas\tests\frame\indexing\test_get_value.py`

```
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
)


class TestGetValue:
    def test_get_set_value_no_partial_indexing(self):
        # 创建一个多级索引对象，包含元组 (0, 1), (0, 2), (1, 1), (1, 2)
        index = MultiIndex.from_tuples([(0, 1), (0, 2), (1, 1), (1, 2)])
        # 创建一个 DataFrame，使用上述多级索引和列范围为 0 到 3
        df = DataFrame(index=index, columns=range(4))
        # 使用 pytest 检查调用 _get_value(0, 1) 时是否引发 KeyError 异常，并匹配正则表达式 "^0$"
        with pytest.raises(KeyError, match=r"^0$"):
            df._get_value(0, 1)

    def test_get_value(self, float_frame):
        # 遍历 float_frame 的索引和列
        for idx in float_frame.index:
            for col in float_frame.columns:
                # 调用 DataFrame 的 _get_value 方法，获取索引 idx 和列 col 处的值
                result = float_frame._get_value(idx, col)
                # 从 float_frame 直接索引获取期望的值
                expected = float_frame[col][idx]
                # 断言实际结果与期望结果相等
                assert result == expected
```