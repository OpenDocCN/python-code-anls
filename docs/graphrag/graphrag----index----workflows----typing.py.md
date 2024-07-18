# `.\graphrag\graphrag\index\workflows\typing.py`

```py
# 版权声明，版权所有 (c) 2024 微软公司。
# 根据 MIT 许可证授权

"""包含 'WorkflowToRun' 模型的模块。"""

# 导入所需模块
from collections.abc import Callable
from dataclasses import dataclass as dc_dataclass
from typing import Any

# 导入自定义模块
from datashaper import TableContainer, Workflow

StepDefinition = dict[str, Any]
"""一个步骤定义。"""

VerbDefinitions = dict[str, Callable[..., TableContainer]
"""动词名称到它们的实现的映射。"""

WorkflowConfig = dict[str, Any]
"""工作流配置。"""

WorkflowDefinitions = dict[str, Callable[[WorkflowConfig], list[StepDefinition]]
"""工作流名称到它们的实现的映射。"""

VerbTiming = dict[str, float]
"""按 id 效率的动词时序。"""

# WorkflowToRun 类的数据类定义
@dc_dataclass
class WorkflowToRun:
    """要运行的工作流类的定义。"""

    workflow: Workflow
    config: dict[str, Any]
```