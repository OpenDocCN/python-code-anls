# `.\DB-GPT-src\dbgpt\util\console\__init__.py`

```py
# 从当前包的console模块中导入CliLogger类，并忽略Flake8的F401警告（表示未使用的导入）
from .console import CliLogger  # noqa: F401

# 将CliLogger类添加到__ALL__列表中，表明该类是当前包的公开接口之一
__ALL__ = ["CliLogger"]
```