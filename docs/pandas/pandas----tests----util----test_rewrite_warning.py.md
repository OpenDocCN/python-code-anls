# `D:\src\scipysrc\pandas\pandas\tests\util\test_rewrite_warning.py`

```
# 导入警告处理模块
import warnings
# 导入 pytest 测试框架
import pytest
# 导入 pandas 内部的警告重写函数
from pandas.util._exceptions import rewrite_warning
# 导入 pandas 测试工具模块
import pandas._testing as tm

# 使用 pytest 的 parametrize 装饰器定义测试参数化
@pytest.mark.parametrize(
    "target_category, target_message, hit",
    [
        # 定义测试用例参数：目标警告类型、目标消息、预期是否命中
        (FutureWarning, "Target message", True),
        (FutureWarning, "Target", True),
        (FutureWarning, "get mess", True),
        (FutureWarning, "Missed message", False),
        (DeprecationWarning, "Target message", False),
    ],
)
# 使用 pytest 的 parametrize 装饰器定义另一组参数化
@pytest.mark.parametrize(
    "new_category",
    [
        None,
        DeprecationWarning,
    ],
)
# 定义测试函数：测试警告消息重写功能
def test_rewrite_warning(target_category, target_message, hit, new_category):
    # 新消息用于预期警告
    new_message = "Rewritten message"
    # 根据命中情况设置预期的警告类别和消息
    if hit:
        expected_category = new_category if new_category else target_category
        expected_message = new_message
    else:
        expected_category = FutureWarning
        expected_message = "Target message"
    
    # 使用 pytest 的 assert_produces_warning 上下文检查是否产生预期警告
    with tm.assert_produces_warning(expected_category, match=expected_message):
        # 使用 rewrite_warning 上下文进行警告重写
        with rewrite_warning(
            target_message, target_category, new_message, new_category
        ):
            # 触发一个特定警告消息和类别的警告
            warnings.warn(message="Target message", category=FutureWarning)
```