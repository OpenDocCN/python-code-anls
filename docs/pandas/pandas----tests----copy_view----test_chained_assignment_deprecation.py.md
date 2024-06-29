# `D:\src\scipysrc\pandas\pandas\tests\copy_view\test_chained_assignment_deprecation.py`

```
# 导入 numpy 库，用于处理数组和数值计算
import numpy as np
# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 pandas.errors 中的 ChainedAssignmentError，用于捕获链式赋值错误
from pandas.errors import ChainedAssignmentError

# 导入 pandas 库中的 DataFrame 类
from pandas import DataFrame
# 导入 pandas._testing 模块，用于编写测试辅助函数
import pandas._testing as tm

# 使用 pytest.mark.parametrize 装饰器来定义参数化测试函数 test_series_setitem
@pytest.mark.parametrize(
    "indexer", [0, [0, 1], slice(0, 2), np.array([True, False, True])]
)
def test_series_setitem(indexer):
    # 创建一个 DataFrame 对象，包含两列 'a' 和 'b'，并赋初值
    df = DataFrame({"a": [1, 2, 3], "b": 1})

    # 使用 pytest.warns 上下文管理器捕获警告信息
    with pytest.warns() as record:  # noqa: TID251
        # 将 df["a"][indexer] 设为 0，检查是否触发了 ChainedAssignmentError 警告
        df["a"][indexer] = 0
    # 断言捕获的警告数量为 1
    assert len(record) == 1
    # 断言捕获的警告类型为 ChainedAssignmentError
    assert record[0].category == ChainedAssignmentError


# 使用 pytest.mark.parametrize 装饰器来定义参数化测试函数 test_frame_setitem
@pytest.mark.parametrize(
    "indexer", ["a", ["a", "b"], slice(0, 2), np.array([True, False, True])]
)
def test_frame_setitem(indexer):
    # 创建一个 DataFrame 对象，包含两列 'a' 和 'b'，并赋初值
    df = DataFrame({"a": [1, 2, 3, 4, 5], "b": 1})

    # 使用 tm.raises_chained_assignment_error 上下文管理器捕获链式赋值错误
    with tm.raises_chained_assignment_error():
        # 将 df[0:3][indexer] 设为 10，确保会触发链式赋值错误
        df[0:3][indexer] = 10
```