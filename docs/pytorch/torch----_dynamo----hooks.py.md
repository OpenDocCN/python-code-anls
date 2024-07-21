# `.\pytorch\torch\_dynamo\hooks.py`

```
# 导入 dataclasses 模块，用于创建数据类
import dataclasses

# 导入 Callable 和 Optional 类型提示
from typing import Callable, Optional

# 导入 GuardsSet 类型和 GuardFail 类型，注意这里的导入路径是相对当前模块的
from torch._guards import GuardsSet
from .types import GuardFail

# 使用 dataclasses 装饰器定义一个数据类 Hooks
@dataclasses.dataclass
class Hooks:
    # 定义一个可选的回调函数 guard_export_fn，其参数类型为 GuardsSet，无返回值
    guard_export_fn: Optional[Callable[[GuardsSet], None]] = None
    # 定义一个可选的回调函数 guard_fail_fn，其参数类型为 GuardFail，无返回值
    guard_fail_fn: Optional[Callable[[GuardFail], None]] = None
```