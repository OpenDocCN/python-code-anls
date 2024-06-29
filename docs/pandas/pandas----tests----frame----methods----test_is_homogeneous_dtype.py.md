# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_is_homogeneous_dtype.py`

```
# 导入必要的库
import numpy as np
import pytest

# 从 pandas 库中导入必要的类
from pandas import (
    Categorical,
    DataFrame,
)

# 使用 pytest 提供的 parametrize 装饰器为 test_is_homogeneous_type 函数参数化测试用例
@pytest.mark.parametrize(
    "data, expected",
    [
        # 空 DataFrame
        (DataFrame(), True),
        # 多列数据类型相同的 DataFrame
        (DataFrame({"A": [1, 2], "B": [1, 2]}), True),
        # 多列数据类型为 Python 对象的 DataFrame
        (
            DataFrame(
                {
                    "A": np.array([1, 2], dtype=object),
                    "B": np.array(["a", "b"], dtype=object),
                },
                dtype="object",
            ),
            True,
        ),
        # 多列数据类型为分类数据的 DataFrame
        (
            DataFrame({"A": Categorical(["a", "b"]), "B": Categorical(["a", "b"])}),
            True,
        ),
        # 不同数据类型的 DataFrame
        (DataFrame({"A": [1, 2], "B": [1.0, 2.0]}), False),
        # 不同列大小的 DataFrame
        (
            DataFrame(
                {
                    "A": np.array([1, 2], dtype=np.int32),
                    "B": np.array([1, 2], dtype=np.int64),
                }
            ),
            False,
        ),
        # 多列数据类型为分类数据，但类型不同的 DataFrame
        (
            DataFrame({"A": Categorical(["a", "b"]), "B": Categorical(["b", "c"])}),
            False,
        ),
    ],
)
# 定义测试函数 test_is_homogeneous_type，验证 DataFrame 是否为同质类型
def test_is_homogeneous_type(data, expected):
    # 断言 data._is_homogeneous_type 的值与 expected 是否相等
    assert data._is_homogeneous_type is expected
```