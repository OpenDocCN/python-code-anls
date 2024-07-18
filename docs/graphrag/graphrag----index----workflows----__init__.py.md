# `.\graphrag\graphrag\index\workflows\__init__.py`

```py
# 版权声明和许可证信息，声明代码版权归 Microsoft Corporation 所有，并遵循 MIT 许可证
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""索引引擎工作流包的根目录。"""
# 当前文件是“索引引擎工作流”模块的根目录，这里导入了所需的函数和类型

# 从当前包中导入创建工作流和加载工作流函数
from .load import create_workflow, load_workflows
# 从当前包中导入所需的类型
from .typing import (
    StepDefinition,
    VerbDefinitions,
    VerbTiming,
    WorkflowConfig,
    WorkflowDefinitions,
    WorkflowToRun,
)

# __all__ 列表定义了在使用 `from package import *` 时导入的公共接口
__all__ = [
    "StepDefinition",        # 步骤定义类型
    "VerbDefinitions",       # 动词定义类型
    "VerbTiming",            # 动词时间类型
    "WorkflowConfig",        # 工作流配置类型
    "WorkflowDefinitions",   # 工作流定义类型
    "WorkflowToRun",         # 要运行的工作流类型
    "create_workflow",       # 创建工作流函数
    "load_workflows",        # 加载工作流函数
]
```