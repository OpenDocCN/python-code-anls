# `.\graphrag\graphrag\index\verbs\text\translate\strategies\__init__.py`

```py
# 版权声明：版权所有 2024 年 Microsoft 公司。
# 使用 MIT 许可证授权

"""索引引擎翻译策略包的根目录。"""

# 从当前目录的 mock 模块导入 run 函数，并重命名为 run_mock
from .mock import run as run_mock
# 从当前目录的 openai 模块导入 run 函数，并重命名为 run_openai
from .openai import run as run_openai

# 指定模块中可以公开的对象列表，只包括 run_mock 和 run_openai
__all__ = ["run_mock", "run_openai"]
```