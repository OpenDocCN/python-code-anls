# `.\graphrag\graphrag\index\errors.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""GraphRAG indexing error types."""

# 定义一个自定义异常类，继承自 ValueError，用于指示未定义工作流的异常情况
class NoWorkflowsDefinedError(ValueError):
    """Exception for no workflows defined."""

    def __init__(self):
        # 调用父类的初始化方法，传入异常消息 "No workflows defined."
        super().__init__("No workflows defined.")


# 定义一个自定义异常类，继承自 ValueError，用于指示未定义工作流名称的异常情况
class UndefinedWorkflowError(ValueError):
    """Exception for invalid verb input."""

    def __init__(self):
        # 调用父类的初始化方法，传入异常消息 "Workflow name is undefined."
        super().__init__("Workflow name is undefined.")


# 定义一个自定义异常类，继承自 ValueError，用于指示未知工作流名称的异常情况
class UnknownWorkflowError(ValueError):
    """Exception for invalid verb input."""

    def __init__(self, name: str):
        # 调用父类的初始化方法，传入异常消息 "Unknown workflow: {name}"，其中{name}为传入的名称参数
        super().__init__(f"Unknown workflow: {name}")
```