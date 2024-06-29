# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_droplevel.py`

```
import pytest
# 导入 pytest 库，用于编写和运行测试

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
)
# 从 pandas 库中导入 DataFrame、Index 和 MultiIndex 类

import pandas._testing as tm
# 导入 pandas 内部测试模块，用于断言测试结果的一致性

class TestDropLevel:
    # 定义测试类 TestDropLevel

    def test_droplevel(self, frame_or_series):
        # 定义测试方法 test_droplevel，并接受参数 frame_or_series

        # GH#20342
        # GH 表示 GitHub 上的 issue 号，这里是 issue #20342 的问题
        # 创建一个包含两个级别的 MultiIndex 对象，列名分别为 'level_1' 和 'level_2'
        cols = MultiIndex.from_tuples(
            [("c", "e"), ("d", "f")], names=["level_1", "level_2"]
        )

        # 创建一个包含两个级别的 MultiIndex 对象，索引名分别为 'a' 和 'b'
        mi = MultiIndex.from_tuples([(1, 2), (5, 6), (9, 10)], names=["a", "b"])

        # 创建一个 DataFrame 对象，包含指定的索引 mi 和列名 cols，并填充数据
        df = DataFrame([[3, 4], [7, 8], [11, 12]], index=mi, columns=cols)

        # 如果 frame_or_series 不是 DataFrame 类型，则将 df 转换为其第一列
        if frame_or_series is not DataFrame:
            df = df.iloc[:, 0]

        # 测试删除索引中的一个级别是否有效
        expected = df.reset_index("a", drop=True)
        result = df.droplevel("a", axis="index")
        tm.assert_equal(result, expected)

        if frame_or_series is DataFrame:
            # 测试删除列中的一个级别是否有效
            expected = df.copy()
            expected.columns = Index(["c", "d"], name="level_1")
            result = df.droplevel("level_2", axis="columns")
            tm.assert_equal(result, expected)
        else:
            # 测试在 axis != 0 时，droplevel 方法是否会引发 ValueError 异常
            with pytest.raises(ValueError, match="No axis named columns"):
                df.droplevel(1, axis="columns")
```