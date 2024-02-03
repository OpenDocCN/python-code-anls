# `.\AutoGPT\autogpts\forge\forge\sdk\db_test.py`

```py
# 导入必要的库
import os
import sqlite3
from datetime import datetime

# 导入 pytest 库
import pytest

# 导入自定义的数据库操作相关模块
from forge.sdk.db import (
    AgentDB,
    ArtifactModel,
    StepModel,
    TaskModel,
    convert_to_artifact,
    convert_to_step,
    convert_to_task,
)

# 导入自定义的错误处理模块
from forge.sdk.errors import NotFoundError as DataNotFoundError

# 导入自定义的数据模型
from forge.sdk.model import (
    Artifact,
    Status,
    Step,
    StepRequestBody,
    Task,
)

# 异步测试函数标记
@pytest.mark.asyncio
def test_table_creation():
    # 定义测试数据库名称
    db_name = "sqlite:///test_db.sqlite3"
    # 创建 AgentDB 对象
    agent_db = AgentDB(db_name)

    # 连接到测试数据库
    conn = sqlite3.connect("test_db.sqlite3")
    cursor = conn.cursor()

    # 测试 tasks 表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'")
    assert cursor.fetchone() is not None

    # 测试 steps 表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='steps'")
    assert cursor.fetchone() is not None

    # 测试 artifacts 表是否存在
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='artifacts'"
    )
    assert cursor.fetchone() is not None

    # 删除测试数据库文件
    os.remove(db_name.split("///")[1])


# 异步测试函数标记
@pytest.mark.asyncio
async def test_task_schema():
    # 获取当前时间
    now = datetime.now()
    # 创建一个 Task 对象
    task = Task(
        task_id="50da533e-3904-4401-8a07-c49adf88b5eb",
        input="Write the words you receive to the file 'output.txt'.",
        created_at=now,
        modified_at=now,
        artifacts=[
            Artifact(
                artifact_id="b225e278-8b4c-4f99-a696-8facf19f0e56",
                agent_created=True,
                file_name="main.py",
                relative_path="python/code/",
                created_at=now,
                modified_at=now,
            )
        ],
    )
    # 断言 Task 对象的属性值
    assert task.task_id == "50da533e-3904-4401-8a07-c49adf88b5eb"
    assert task.input == "Write the words you receive to the file 'output.txt'."
    assert len(task.artifacts) == 1
    # 断言任务的第一个工件的工件ID是否等于指定的值
    assert task.artifacts[0].artifact_id == "b225e278-8b4c-4f99-a696-8facf19f0e56"
# 标记为异步测试
@pytest.mark.asyncio
async def test_step_schema():
    # 获取当前时间
    now = datetime.now()
    # 创建一个步骤对象
    step = Step(
        task_id="50da533e-3904-4401-8a07-c49adf88b5eb",
        step_id="6bb1801a-fd80-45e8-899a-4dd723cc602e",
        created_at=now,
        modified_at=now,
        name="Write to file",
        input="Write the words you receive to the file 'output.txt'.",
        status=Status.created,
        output="I am going to use the write_to_file command and write Washington to a file called output.txt <write_to_file('output.txt', 'Washington')>",
        artifacts=[
            Artifact(
                artifact_id="b225e278-8b4c-4f99-a696-8facf19f0e56",
                file_name="main.py",
                relative_path="python/code/",
                created_at=now,
                modified_at=now,
                agent_created=True,
            )
        ],
        is_last=False,
    )
    # 断言步骤对象的属性值
    assert step.task_id == "50da533e-3904-4401-8a07-c49adf88b5eb"
    assert step.step_id == "6bb1801a-fd80-45e8-899a-4dd723cc602e"
    assert step.name == "Write to file"
    assert step.status == Status.created
    assert (
        step.output
        == "I am going to use the write_to_file command and write Washington to a file called output.txt <write_to_file('output.txt', 'Washington')>"
    )
    assert len(step.artifacts) == 1
    assert step.artifacts[0].artifact_id == "b225e278-8b4c-4f99-a696-8facf19f0e56"
    assert step.is_last == False

# 标记为异步测试
@pytest.mark.asyncio
async def test_convert_to_task():
    # 获取当前时间
    now = datetime.now()
    # 创建一个 TaskModel 对象，设置任务 ID、创建时间、修改时间、输入内容和相关文档信息
    task_model = TaskModel(
        task_id="50da533e-3904-4401-8a07-c49adf88b5eb",
        created_at=now,
        modified_at=now,
        input="Write the words you receive to the file 'output.txt'.",
        artifacts=[
            # 创建一个 ArtifactModel 对象，设置文档 ID、创建时间、修改时间、相对路径、是否由代理创建和文件名
            ArtifactModel(
                artifact_id="b225e278-8b4c-4f99-a696-8facf19f0e56",
                created_at=now,
                modified_at=now,
                relative_path="file:///path/to/main.py",
                agent_created=True,
                file_name="main.py",
            )
        ],
    )
    # 将 TaskModel 对象转换为 Task 对象
    task = convert_to_task(task_model)
    # 断言任务 ID 是否正确
    assert task.task_id == "50da533e-3904-4401-8a07-c49adf88b5eb"
    # 断言输入内容是否正确
    assert task.input == "Write the words you receive to the file 'output.txt'."
    # 断言文档列表长度是否为1
    assert len(task.artifacts) == 1
    # 断言第一个文档的 ID 是否正确
    assert task.artifacts[0].artifact_id == "b225e278-8b4c-4f99-a696-8facf19f0e56"
# 标记为异步测试
@pytest.mark.asyncio
async def test_convert_to_step():
    # 获取当前时间
    now = datetime.now()
    # 创建 StepModel 对象
    step_model = StepModel(
        task_id="50da533e-3904-4401-8a07-c49adf88b5eb",
        step_id="6bb1801a-fd80-45e8-899a-4dd723cc602e",
        created_at=now,
        modified_at=now,
        name="Write to file",
        status="created",
        input="Write the words you receive to the file 'output.txt'.",
        artifacts=[
            # 创建 ArtifactModel 对象
            ArtifactModel(
                artifact_id="b225e278-8b4c-4f99-a696-8facf19f0e56",
                created_at=now,
                modified_at=now,
                relative_path="file:///path/to/main.py",
                agent_created=True,
                file_name="main.py",
            )
        ],
        is_last=False,
    )
    # 转换 StepModel 对象为 Step 对象
    step = convert_to_step(step_model)
    # 断言 Step 对象的属性值
    assert step.task_id == "50da533e-3904-4401-8a07-c49adf88b5eb"
    assert step.step_id == "6bb1801a-fd80-45e8-899a-4dd723cc602e"
    assert step.name == "Write to file"
    assert step.status == Status.created
    assert len(step.artifacts) == 1
    assert step.artifacts[0].artifact_id == "b225e278-8b4c-4f99-a696-8facf19f0e56"
    assert step.is_last == False


# 标记为异步测试
@pytest.mark.asyncio
async def test_convert_to_artifact():
    # 获取当前时间
    now = datetime.now()
    # 创建 ArtifactModel 对象
    artifact_model = ArtifactModel(
        artifact_id="b225e278-8b4c-4f99-a696-8facf19f0e56",
        created_at=now,
        modified_at=now,
        relative_path="file:///path/to/main.py",
        agent_created=True,
        file_name="main.py",
    )
    # 转换 ArtifactModel 对象为 Artifact 对象
    artifact = convert_to_artifact(artifact_model)
    # 断言 Artifact 对象的属性值
    assert artifact.artifact_id == "b225e278-8b4c-4f99-a696-8facf19f0e56"
    assert artifact.relative_path == "file:///path/to/main.py"
    assert artifact.agent_created == True


# 标记为异步测试
@pytest.mark.asyncio
async def test_create_task():
    # 由于 pytest fixture 存在问题，暂时在每个测试中添加设置和清理作为快速解决方法
    # TODO: 修复这个问题！
    # 数据库名称
    db_name = "sqlite:///test_db.sqlite3"
    # 创建 AgentDB 对象
    agent_db = AgentDB(db_name)
    # 创建一个任务，并等待任务创建完成
    task = await agent_db.create_task("task_input")
    # 断言任务的输入为"task_input"
    assert task.input == "task_input"
    # 从数据库名称中提取文件名，并删除该文件
    os.remove(db_name.split("///")[1])
# 使用 pytest.mark.asyncio 标记异步测试函数
@pytest.mark.asyncio
async def test_create_and_get_task():
    # 定义数据库名称
    db_name = "sqlite:///test_db.sqlite3"
    # 创建 AgentDB 对象
    agent_db = AgentDB(db_name)
    # 创建任务并获取任务对象
    task = await agent_db.create_task("test_input")
    fetched_task = await agent_db.get_task(task.task_id)
    # 断言获取的任务输入与预期相符
    assert fetched_task.input == "test_input"
    # 删除测试数据库文件
    os.remove(db_name.split("///")[1])


@pytest.mark.asyncio
async def test_get_task_not_found():
    # 定义数据库名称
    db_name = "sqlite:///test_db.sqlite3"
    # 创建 AgentDB 对象
    agent_db = AgentDB(db_name)
    # 使用 pytest.raises 检查是否抛出 DataNotFoundError 异常
    with pytest.raises(DataNotFoundError):
        await agent_db.get_task(9999)
    # 删除测试数据库文件
    os.remove(db_name.split("///")[1])


@pytest.mark.asyncio
async def test_create_and_get_step():
    # 定义数据库名称
    db_name = "sqlite:///test_db.sqlite3"
    # 创建 AgentDB 对象
    agent_db = AgentDB(db_name)
    # 创建任务并获取任务对象
    task = await agent_db.create_task("task_input")
    # 创建步骤输入对象
    step_input = StepInput(type="python/code")
    # 创建步骤请求对象
    request = StepRequestBody(input="test_input debug", additional_input=step_input)
    # 创建步骤并获取步骤对象
    step = await agent_db.create_step(task.task_id, request)
    step = await agent_db.get_step(task.task_id, step.step_id)
    # 断言获取的步骤输入与预期相符
    assert step.input == "test_input debug"
    # 删除测试数据库文件
    os.remove(db_name.split("///")[1])


@pytest.mark.asyncio
async def test_updating_step():
    # 定义数据库名称
    db_name = "sqlite:///test_db.sqlite3"
    # 创建 AgentDB 对象
    agent_db = AgentDB(db_name)
    # 创建任务并获取任务对象
    created_task = await agent_db.create_task("task_input")
    # 创建步骤输入对象
    step_input = StepInput(type="python/code")
    # 创建步骤请求对象
    request = StepRequestBody(input="test_input debug", additional_input=step_input)
    # 创建步骤并获取步骤对象
    created_step = await agent_db.create_step(created_task.task_id, request)
    # 更新步骤状态为 "completed"
    await agent_db.update_step(created_task.task_id, created_step.step_id, "completed")

    # 获取更新后的步骤对象
    step = await agent_db.get_step(created_task.task_id, created_step.step_id)
    # 断言步骤状态为 "completed"
    assert step.status.value == "completed"
    # 删除测试数据库文件
    os.remove(db_name.split("///")[1])


@pytest.mark.asyncio
async def test_get_step_not_found():
    # 定义数据库名称
    db_name = "sqlite:///test_db.sqlite3"
    # 创建 AgentDB 对象
    agent_db = AgentDB(db_name)
    # 使用 pytest.raises 检查是否抛出 DataNotFoundError 异常
    with pytest.raises(DataNotFoundError):
        await agent_db.get_step(9999, 9999)
    # 根据指定的分隔符 "///" 将数据库名称分割成多个部分，并删除索引为1的部分对应的文件
    os.remove(db_name.split("///")[1])
# 使用 pytest.mark.asyncio 标记异步测试函数
@pytest.mark.asyncio
async def test_get_artifact():
    # 定义数据库名称
    db_name = "sqlite:///test_db.sqlite3"
    # 创建 AgentDB 对象
    db = AgentDB(db_name)

    # 给定：一个任务及其对应的 artifact
    # 创建任务
    task = await db.create_task("test_input debug")
    # 创建 StepInput 对象
    step_input = StepInput(type="python/code")
    # 创建 StepRequestBody 对象
    requst = StepRequestBody(input="test_input debug", additional_input=step_input)
    # 创建步骤
    step = await db.create_step(task.task_id, requst)

    # 创建 artifact
    artifact = await db.create_artifact(
        task_id=task.task_id,
        file_name="test_get_artifact_sample_file.txt",
        relative_path="file:///path/to/test_get_artifact_sample_file.txt",
        agent_created=True,
        step_id=step.step_id,
    )

    # 当：通过 artifact 的 ID 获取 artifact
    fetched_artifact = await db.get_artifact(artifact.artifact_id)

    # 则：获取的 artifact 与原始 artifact 匹配
    assert fetched_artifact.artifact_id == artifact.artifact_id
    assert (
        fetched_artifact.relative_path
        == "file:///path/to/test_get_artifact_sample_file.txt"
    )

    # 删除数据库文件
    os.remove(db_name.split("///")[1])


# 使用 pytest.mark.asyncio 标记异步测试函数
@pytest.mark.asyncio
async def test_list_tasks():
    # 定义数据库名称
    db_name = "sqlite:///test_db.sqlite3"
    # 创建 AgentDB 对象
    db = AgentDB(db_name)

    # 给定：数据库中存在多个任务
    # 创建任务1
    task1 = await db.create_task("test_input_1")
    # 创建任务2
    task2 = await db.create_task("test_input_2")

    # 当：获取所有任务
    fetched_tasks, pagination = await db.list_tasks()

    # 则：获取的任务列表包括创建的任务
    task_ids = [task.task_id for task in fetched_tasks]
    assert task1.task_id in task_ids
    assert task2.task_id in task_ids
    # 删除数据库文件
    os.remove(db_name.split("///")[1])


# 使用 pytest.mark.asyncio 标记异步测试函数
@pytest.mark.asyncio
async def test_list_steps():
    # 定义数据库名称
    db_name = "sqlite:///test_db.sqlite3"
    # 创建 AgentDB 对象
    db = AgentDB(db_name)

    # 创建 StepInput 对象
    step_input = StepInput(type="python/code")
    # 创建 StepRequestBody 对象
    requst = StepRequestBody(input="test_input debug", additional_input=step_input)

    # 给定：一个任务及该任务的多个步骤
    # 创建一个名为 "test_input" 的任务，并将其赋值给变量 task
    task = await db.create_task("test_input")
    # 创建一个步骤，将请求 requst 与任务 task 相关联，并将其赋值给变量 step1
    step1 = await db.create_step(task.task_id, requst)
    # 创建一个请求 requst，包含输入为 "step two" 和额外输入为 step_input
    requst = StepRequestBody(input="step two", additional_input=step_input)
    # 创建一个步骤，将请求 requst 与任务 task 相关联，并将其赋值给变量 step2
    step2 = await db.create_step(task.task_id, requst)

    # 当：获取任务的所有步骤
    fetched_steps, pagination = await db.list_steps(task.task_id)

    # 那么：获取的步骤列表应包含创建的步骤
    # 从 fetched_steps 中提取所有步骤的 step_id 组成列表 step_ids
    step_ids = [step.step_id for step in fetched_steps]
    # 断言 step1 的 step_id 在 step_ids 中
    assert step1.step_id in step_ids
    # 断言 step2 的 step_id 在 step_ids 中
    assert step2.step_id in step_ids
    # 移除 db_name 中以 "///" 分割后的第二部分文件
    os.remove(db_name.split("///")[1])
```