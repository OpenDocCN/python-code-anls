# `.\pytorch\torch\_higher_order_ops\__init__.py`

```
# 导入来自 torch 库中的条件操作（cond）、循环操作（while_loop）、灵活注意力（flex_attention）和灵活注意力反向（flex_attention_backward）
from torch._higher_order_ops.cond import cond
from torch._higher_order_ops.flex_attention import (
    flex_attention,
    flex_attention_backward,
)
from torch._higher_order_ops.while_loop import while_loop

# 定义一个包含所有导入符号名称的列表，用于模块级的导入管理
__all__ = [
    "cond",  # 条件操作函数
    "while_loop",  # 循环操作函数
    "flex_attention",  # 灵活注意力前向函数
    "flex_attention_backward",  # 灵活注意力反向函数
]
```