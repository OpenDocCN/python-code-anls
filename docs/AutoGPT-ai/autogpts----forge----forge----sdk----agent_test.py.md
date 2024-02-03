# `.\AutoGPT\autogpts\forge\forge\sdk\agent_test.py`

```py
# 导入 pytest 模块
import pytest

# 导入 Agent 类
from .agent import Agent
# 导入 AgentDB 类
from .db import AgentDB
# 导入 StepRequestBody, Task, TaskListResponse, TaskRequestBody 类
from .model import StepRequestBody, Task, TaskListResponse, TaskRequestBody
# 导入 LocalWorkspace 类
from .workspace import LocalWorkspace

# 使用 pytest.fixture 装饰器定义 agent 函数
@pytest.fixture
def agent():
    # 创建 AgentDB 对象
    db = AgentDB("sqlite:///test.db")
    # 创建 LocalWorkspace 对象
    workspace = LocalWorkspace("./test_workspace")
    # 返回 Agent 对象
    return Agent(db, workspace)

# 使用 pytest.mark.skip 装饰器标记测试用例
# 使用 pytest.mark.asyncio 装饰器标记异步测试用例
# 定义 test_create_task 异步测试用例
async def test_create_task(agent):
    # 创建 TaskRequestBody 对象
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    # 调用 Agent 对象的 create_task 方法创建任务
    task: Task = await agent.create_task(task_request)
    # 断言任务的输入为 "test_input"
    assert task.input == "test_input"

# 定义 test_list_tasks 异步测试用例
async def test_list_tasks(agent):
    # 创建 TaskRequestBody 对象
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    # 调用 Agent 对象的 create_task 方法创建任务
    task = await agent.create_task(task_request)
    # 调用 Agent 对象的 list_tasks 方法获取任务列表
    tasks = await agent.list_tasks()
    # 断言 tasks 是 TaskListResponse 类型的实例
    assert isinstance(tasks, TaskListResponse)

# 定义 test_get_task 异步测试用例
async def test_get_task(agent):
    # 创建 TaskRequestBody 对象
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    # 调用 Agent 对象的 create_task 方法创建任务
    task = await agent.create_task(task_request)
    # 调用 Agent 对象的 get_task 方法获取指定任务
    retrieved_task = await agent.get_task(task.task_id)
    # 断言获取的任务的 task_id 与创建的任务的 task_id 相同
    assert retrieved_task.task_id == task.task_id

# 定义 test_create_and_execute_step 异步测试用例
async def test_create_and_execute_step(agent):
    # 创建 TaskRequestBody 对象
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    # 调用 Agent 对象的 create_task 方法创建任务
    task = await agent.create_task(task_request)
    # 创建 StepRequestBody 对象
    step_request = StepRequestBody(
        input="step_input", additional_input={"input": "additional_test_input"}
    )
    # 调用 Agent 对象的 create_and_execute_step 方法创建并执行步骤
    step = await agent.create_and_execute_step(task.task_id, step_request)
    # 断言步骤的输入为 "step_input"
    assert step.input == "step_input"
    # 断言步骤的附加输入为 {"input": "additional_test_input"}
    assert step.additional_input == {"input": "additional_test_input"}

# 定义 test_get_step 异步测试用例
async def test_get_step(agent):
    # 创建任务请求对象，包括输入和额外输入信息
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    # 使用代理创建任务，并等待返回结果
    task = await agent.create_task(task_request)
    # 创建步骤请求对象，包括输入和额外输入信息
    step_request = StepRequestBody(
        input="step_input", additional_input={"input": "additional_test_input"}
    )
    # 使用代理创建并执行步骤，并等待返回结果
    step = await agent.create_and_execute_step(task.task_id, step_request)
    # 使用代理获取特定任务的特定步骤，并等待返回结果
    retrieved_step = await agent.get_step(task.task_id, step.step_id)
    # 断言检查获取的步骤的步骤ID与创建的步骤的步骤ID是否相同
    assert retrieved_step.step_id == step.step_id
# 标记此测试用例为跳过状态
@pytest.mark.skip
# 标记此测试用例为异步测试
@pytest.mark.asyncio
async def test_list_artifacts(agent):
    # 调用 agent 对象的 list_artifacts 方法获取所有 artifacts
    artifacts = await agent.list_artifacts()
    # 断言 artifacts 的类型为 list
    assert isinstance(artifacts, list)


# 标记此测试用例为跳过状态
@pytest.mark.skip
# 标记此测试用例为异步测试
@pytest.mark.asyncio
async def test_create_artifact(agent):
    # 创建 TaskRequestBody 对象
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    # 调用 agent 对象的 create_task 方法创建任务
    task = await agent.create_task(task_request)
    # 创建 ArtifactRequestBody 对象
    artifact_request = ArtifactRequestBody(file=None, uri="test_uri")
    # 调用 agent 对象的 create_artifact 方法创建 artifact
    artifact = await agent.create_artifact(task.task_id, artifact_request)
    # 断言 artifact 的 uri 属性为 "test_uri"
    assert artifact.uri == "test_uri"


# 标记此测试用例为跳过状态
@pytest.mark.skip
# 标记此测试用例为异步测试
@pytest.mark.asyncio
async def test_get_artifact(agent):
    # 创建 TaskRequestBody 对象
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    # 调用 agent 对象的 create_task 方法创建任务
    task = await agent.create_task(task_request)
    # 创建 ArtifactRequestBody 对象
    artifact_request = ArtifactRequestBody(file=None, uri="test_uri")
    # 调用 agent 对象的 create_artifact 方法创建 artifact
    artifact = await agent.create_artifact(task.task_id, artifact_request)
    # 调用 agent 对象的 get_artifact 方法获取指定任务和 artifact 的 artifact
    retrieved_artifact = await agent.get_artifact(task.task_id, artifact.artifact_id)
    # 断言获取的 artifact 的 artifact_id 与创建的 artifact 的 artifact_id 相同
    assert retrieved_artifact.artifact_id == artifact.artifact_id
```