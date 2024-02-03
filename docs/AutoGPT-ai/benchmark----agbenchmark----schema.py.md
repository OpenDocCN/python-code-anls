# `.\AutoGPT\benchmark\agbenchmark\schema.py`

```py
# 导入未来的注解特性
from __future__ import annotations

# 导入类型提示模块
from typing import Optional

# 导入 Pydantic 模块中的基础模型和字段
from pydantic import BaseModel, Field

# 定义任务输入的基础模型
class TaskInput(BaseModel):
    pass

# 定义任务请求体的基础模型
class TaskRequestBody(BaseModel):
    # 输入字段，必须有最小长度为1，描述为任务的输入提示
    input: str = Field(
        ...,
        min_length=1,
        description="Input prompt for the task.",
        example="Write the words you receive to the file 'output.txt'.",
    )
    # 可选的额外输入字段，默认为空字典
    additional_input: Optional[TaskInput] = {}

# 定义任务评估请求体，继承自任务请求体
class TaskEvalRequestBody(TaskRequestBody):
    # 评估 ID 字段
    eval_id: str
```