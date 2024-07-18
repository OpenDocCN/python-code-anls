# `.\graphrag\graphrag\index\llm\types.py`

```py
# 版权声明，版权归 2024 年微软公司所有，根据 MIT 许可证发布

# 导入必要的库和模块
"""A module containing the 'LLMtype' model."""
# 导入类型别名 TypeAlias 和 Callable 类型
from collections.abc import Callable
# 导入 TypeAlias 类型别名
from typing import TypeAlias

# 定义类型别名 TextSplitter，表示接受一个字符串参数并返回字符串列表的可调用对象
TextSplitter: TypeAlias = Callable[[str], list[str]]
# 定义类型别名 TextListSplitter，表示接受一个字符串列表参数并返回字符串列表的可调用对象
TextListSplitter: TypeAlias = Callable[[list[str]], list[str]]
```