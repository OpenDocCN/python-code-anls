# `.\AutoGPT\autogpts\forge\forge\sdk\model.py`

```py
# 从未来导入注解功能
from __future__ import annotations

# 导入所需的模块和类
from datetime import datetime
from enum import Enum
from typing import List, Optional

# 导入 Pydantic 中的基本模型和字段
from pydantic import BaseModel, Field

# 定义 ArtifactUpload 模型，用于上传文件
class ArtifactUpload(BaseModel):
    # 文件字段，必填，描述为“要上传的文件”，格式为二进制
    file: str = Field(..., description="File to upload.", format="binary")
    # 相对路径字段，必填，描述为“代理工作区中的文件相对路径”，示例为“python/code”
    relative_path: str = Field(
        ...,
        description="Relative path of the artifact in the agent's workspace.",
        example="python/code",
    )

# 定义 Pagination 模型，用于分页信息
class Pagination(BaseModel):
    # 总项目数字段，必填，描述为“项目总数”，示例为 42
    total_items: int = Field(..., description="Total number of items.", example=42)
    # 总页数字段，必填，描述为“总页数”，示例为 97
    total_pages: int = Field(..., description="Total number of pages.", example=97)
    # 当前页码字段，必填，描述为“当前页码”，示例为 1
    current_page: int = Field(..., description="Current_page page number.", example=1)
    # 每页项目数字段，必填，描述为“每页项目数”，示例为 25
    page_size: int = Field(..., description="Number of items per page.", example=25)

# 定义 Artifact 模型，用于表示工件信息
class Artifact(BaseModel):
    # 创建时间字段，必填，描述为“任务的创建时间”，示例为“2023-01-01T00:00:00Z”
    created_at: datetime = Field(
        ...,
        description="The creation datetime of the task.",
        example="2023-01-01T00:00:00Z",
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    # 修改时间字段，必填，描述为“任务的修改时间”，示例为“2023-01-01T00:00:00Z”
    modified_at: datetime = Field(
        ...,
        description="The modification datetime of the task.",
        example="2023-01-01T00:00:00Z",
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    # 工件 ID 字段，必填，描述为“工件的 ID”，示例为“b225e278-8b4c-4f99-a696-8facf19f0e56”
    artifact_id: str = Field(
        ...,
        description="ID of the artifact.",
        example="b225e278-8b4c-4f99-a696-8facf19f0e56",
    )
    # 代理创建字段，必填，描述为“工件是否由代理创建”，示例为 False
    agent_created: bool = Field(
        ...,
        description="Whether the artifact has been created by the agent.",
        example=False,
    )
    # 相对路径字段，必填，描述为“代理工作区中的工件相对路径”，示例为“/my_folder/my_other_folder/”
    relative_path: str = Field(
        ...,
        description="Relative path of the artifact in the agents workspace.",
        example="/my_folder/my_other_folder/",
    )
    # 文件名字段，必填，描述为“工件的文件名”，示例为“main.py”
    file_name: str = Field(
        ...,
        description="Filename of the artifact.",
        example="main.py",
    )
# 定义 StepOutput 类，暂时没有任何属性或方法
class StepOutput(BaseModel):
    pass


# 定义 TaskRequestBody 类，包含输入、附加输入等属性
class TaskRequestBody(BaseModel):
    input: str = Field(
        ...,
        min_length=1,
        description="Input prompt for the task.",
        example="Write the words you receive to the file 'output.txt'.",
    )
    additional_input: Optional[dict] = None


# 定义 Task 类，继承自 TaskRequestBody，包含创建时间、修改时间、任务ID、产生的工件等属性
class Task(TaskRequestBody):
    created_at: datetime = Field(
        ...,
        description="The creation datetime of the task.",
        example="2023-01-01T00:00:00Z",
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    modified_at: datetime = Field(
        ...,
        description="The modification datetime of the task.",
        example="2023-01-01T00:00:00Z",
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    task_id: str = Field(
        ...,
        description="The ID of the task.",
        example="50da533e-3904-4401-8a07-c49adf88b5eb",
    )
    artifacts: Optional[List[Artifact]] = Field(
        [],
        description="A list of artifacts that the task has produced.",
        example=[
            "7a49f31c-f9c6-4346-a22c-e32bc5af4d8e",
            "ab7b4091-2560-4692-a4fe-d831ea3ca7d6",
        ],
    )


# 定义 StepRequestBody 类，包含步骤名称、输入、附加输入等属性
class StepRequestBody(BaseModel):
    name: Optional[str] = Field(
        None, description="The name of the task step.", example="Write to file"
    )
    input: Optional[str] = Field(
        None,
        description="Input prompt for the step.",
        example="Washington",
    )
    additional_input: Optional[dict] = None


# 定义枚举类型 Status，包含任务状态：创建、运行、完成
class Status(Enum):
    created = "created"
    running = "running"
    completed = "completed"


# 定义 Step 类，继承自 StepRequestBody，包含创建时间属性
class Step(StepRequestBody):
    created_at: datetime = Field(
        ...,
        description="The creation datetime of the task.",
        example="2023-01-01T00:00:00Z",
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    # 表示任务的修改日期时间
    modified_at: datetime = Field(
        ...,
        description="The modification datetime of the task.",
        example="2023-01-01T00:00:00Z",
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    # 任务所属的任务ID
    task_id: str = Field(
        ...,
        description="The ID of the task this step belongs to.",
        example="50da533e-3904-4401-8a07-c49adf88b5eb",
    )
    # 任务步骤的ID
    step_id: str = Field(
        ...,
        description="The ID of the task step.",
        example="6bb1801a-fd80-45e8-899a-4dd723cc602e",
    )
    # 任务步骤的名称
    name: Optional[str] = Field(
        None, description="The name of the task step.", example="Write to file"
    )
    # 任务步骤的状态
    status: Status = Field(
        ..., description="The status of the task step.", example="created"
    )
    # 任务步骤的输出
    output: Optional[str] = Field(
        None,
        description="Output of the task step.",
        example="I am going to use the write_to_file command and write Washington to a file called output.txt <write_to_file('output.txt', 'Washington')",
    )
    # 任务步骤的额外输出
    additional_output: Optional[dict] = None
    # 任务步骤生成的工件列表
    artifacts: Optional[List[Artifact]] = Field(
        [], description="A list of artifacts that the step has produced."
    )
    # 是否是任务的最后一个步骤
    is_last: bool = Field(
        ..., description="Whether this is the last step in the task.", example=True
    )
# TaskListResponse 类，包含 tasks 和 pagination 两个可选字段
class TaskListResponse(BaseModel):
    tasks: Optional[List[Task]] = None
    pagination: Optional[Pagination] = None

# TaskStepsListResponse 类，包含 steps 和 pagination 两个可选字段
class TaskStepsListResponse(BaseModel):
    steps: Optional[List[Step]] = None
    pagination: Optional[Pagination] = None

# TaskArtifactsListResponse 类，包含 artifacts 和 pagination 两个可选字段
class TaskArtifactsListResponse(BaseModel):
    artifacts: Optional[List[Artifact]] = None
    pagination: Optional[Pagination] = None
```