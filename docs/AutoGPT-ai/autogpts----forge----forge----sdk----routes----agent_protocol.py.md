# `.\AutoGPT\autogpts\forge\forge\sdk\routes\agent_protocol.py`

```py
"""
Routes for the Agent Service.

This module defines the API routes for the Agent service. While there are multiple endpoints provided by the service,
the ones that require special attention due to their complexity are:

1. `execute_agent_task_step`:
   This route is significant because this is where the agent actually performs the work. The function handles
   executing the next step for a task based on its current state, and it requires careful implementation to ensure
   all scenarios (like the presence or absence of steps or a step marked as `last_step`) are handled correctly.

2. `upload_agent_task_artifacts`:
   This route allows for the upload of artifacts, supporting various URI types (e.g., s3, gcs, ftp, http).
   The support for different URI types makes it a bit more complex, and it's important to ensure that all
   supported URI types are correctly managed. NOTE: The AutoGPT team will eventually handle the most common
   uri types for you.

3. `create_agent_task`:
   While this is a simpler route, it plays a crucial role in the workflow, as it's responsible for the creation
   of a new task.

Developers and contributors should be especially careful when making modifications to these routes to ensure
consistency and correctness in the system's behavior.
"""
import json
from typing import Optional

from fastapi import APIRouter, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse

from forge.sdk.errors import *
from forge.sdk.forge_log import ForgeLogger
from forge.sdk.model import *

# 创建一个 APIRouter 实例
base_router = APIRouter()

# 创建一个 ForgeLogger 实例，用于记录日志
LOG = ForgeLogger(__name__)

# 定义根路由，返回欢迎消息
@base_router.get("/", tags=["root"])
async def root():
    """
    Root endpoint that returns a welcome message.
    """
    return Response(content="Welcome to the AutoGPT Forge")

# 定义心跳检测路由，检查服务器是否运行
@base_router.get("/heartbeat", tags=["server"])
async def check_server_status():
    """
    Check if the server is running.
    """
    return Response(content="Server is running.", status_code=200)
# 创建一个 POST 请求路由，用于创建一个新的任务，并返回一个 Task 对象
@base_router.post("/agent/tasks", tags=["agent"], response_model=Task)
async def create_agent_task(request: Request, task_request: TaskRequestBody) -> Task:
    """
    Creates a new task using the provided TaskRequestBody and returns a Task.

    Args:
        request (Request): FastAPI request object.
        task (TaskRequestBody): The task request containing input and additional input data.

    Returns:
        Task: A new task with task_id, input, additional_input, and empty lists for artifacts and steps.

    Example:
        Request (TaskRequestBody defined in schema.py):
            {
                "input": "Write the words you receive to the file 'output.txt'.",
                "additional_input": "python/code"
            }

        Response (Task defined in schema.py):
            {
                "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                "input": "Write the word 'Washington' to a .txt file",
                "additional_input": "python/code",
                "artifacts": [],
            }
    """
    # 从请求中获取代理对象
    agent = request["agent"]

    try:
        # 使用代理对象创建任务
        task_request = await agent.create_task(task_request)
        # 返回成功响应
        return Response(
            content=task_request.json(),
            status_code=200,
            media_type="application/json",
        )
    except Exception:
        # 捕获异常并记录错误日志
        LOG.exception(f"Error whilst trying to create a task: {task_request}")
        # 返回内部服务器错误响应
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


# 创建一个 GET 请求路由，用于获取所有任务的分页列表
@base_router.get("/agent/tasks", tags=["agent"], response_model=TaskListResponse)
async def list_agent_tasks(
    request: Request,
    page: Optional[int] = Query(1, ge=1),
    page_size: Optional[int] = Query(10, ge=1),
) -> TaskListResponse:
    """
    Retrieves a paginated list of all tasks.
    """
    Args:
        request (Request): FastAPI request object.  # 定义函数参数 request，类型为 FastAPI 的请求对象
        page (int, optional): The page number for pagination. Defaults to 1.  # 定义函数参数 page，表示分页的页数，默认为 1
        page_size (int, optional): The number of tasks per page for pagination. Defaults to 10.  # 定义函数参数 page_size，表示每页任务的数量，默认为 10

    Returns:
        TaskListResponse: A response object containing a list of tasks and pagination details.  # 返回类型为 TaskListResponse 的响应对象，包含任务列表和分页详情

    Example:
        Request:
            GET /agent/tasks?page=1&pageSize=10  # 示例请求，获取第一页，每页显示 10 个任务

        Response (TaskListResponse defined in schema.py):
            {  # 返回的响应对象格式如下
                "items": [  # 任务列表
                    {
                        "input": "Write the word 'Washington' to a .txt file",  # 任务描述
                        "additional_input": null,  # 附加输入
                        "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",  # 任务 ID
                        "artifacts": [],  # 任务产生的文件
                        "steps": []  # 任务步骤
                    },
                    ...
                ],
                "pagination": {  # 分页信息
                    "total": 100,  # 总任务数
                    "pages": 10,  # 总页数
                    "current": 1,  # 当前页数
                    "pageSize": 10  # 每页任务数量
                }
            }
    """
    agent = request["agent"]  # 获取请求中的代理信息
    try:
        tasks = await agent.list_tasks(page, page_size)  # 调用代理对象的 list_tasks 方法获取任务列表
        return Response(
            content=tasks.json(),  # 返回任务列表的 JSON 格式数据
            status_code=200,  # 返回状态码 200 表示成功
            media_type="application/json",  # 返回数据类型为 JSON
        )
    except NotFoundError:
        LOG.exception("Error whilst trying to list tasks")  # 记录未找到任务的异常信息
        return Response(
            content=json.dumps({"error": "Tasks not found"}),  # 返回未找到任务的错误信息
            status_code=404,  # 返回状态码 404 表示未找到
            media_type="application/json",  # 返回数据类型为 JSON
        )
    except Exception:
        LOG.exception("Error whilst trying to list tasks")  # 记录获取任务列表时的异常信息
        return Response(
            content=json.dumps({"error": "Internal server error"}),  # 返回内部服务器错误的错误信息
            status_code=500,  # 返回状态码 500 表示内部服务器错误
            media_type="application/json",  # 返回数据类型为 JSON
        )
# 定义一个路由处理函数，用于处理 GET 请求，路径为 "/agent/tasks/{task_id}"，标签为 "agent"，响应模型为 Task
async def get_agent_task(request: Request, task_id: str) -> Task:
    """
    获取指定 ID 的任务的详细信息。

    Args:
        request (Request): FastAPI 请求对象。
        task_id (str): 任务的 ID。

    Returns:
        Task: 指定 ID 的任务。
    """
    # 定义示例请求和响应，展示了任务的结构和内容
    Example:
        Request:
            GET /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb

        Response (Task defined in schema.py):
            {
                "input": "Write the word 'Washington' to a .txt file",
                "additional_input": null,
                "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                "artifacts": [
                    {
                        "artifact_id": "7a49f31c-f9c6-4346-a22c-e32bc5af4d8e",
                        "file_name": "output.txt",
                        "agent_created": true,
                        "relative_path": "file://50da533e-3904-4401-8a07-c49adf88b5eb/output.txt"
                    }
                ],
                "steps": [
                    {
                        "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                        "step_id": "6bb1801a-fd80-45e8-899a-4dd723cc602e",
                        "input": "Write the word 'Washington' to a .txt file",
                        "additional_input": "challenge:write_to_file",
                        "name": "Write to file",
                        "status": "completed",
                        "output": "I am going to use the write_to_file command and write Washington to a file called output.txt <write_to_file('output.txt', 'Washington')>",
                        "additional_output": "Do you want me to continue?",
                        "artifacts": [
                            {
                                "artifact_id": "7a49f31c-f9c6-4346-a22c-e32bc5af4d8e",
                                "file_name": "output.txt",
                                "agent_created": true,
                                "relative_path": "file://50da533e-3904-4401-8a07-c49adf88b5eb/output.txt"
                            }
                        ],
                        "is_last": true
                    }
                ]
            }
    """
    # 从请求中获取代理信息
    agent = request["agent"]
    # 尝试从代理获取任务信息
    try:
        task = await agent.get_task(task_id)
        # 返回包含任务信息的响应
        return Response(
            content=task.json(),
            status_code=200,
            media_type="application/json",
        )
    # 如果任务未找到，则捕获 NotFoundError 异常
    except NotFoundError:
        # 记录异常信息
        LOG.exception(f"Error whilst trying to get task: {task_id}")
        # 返回包含错误信息的响应
        return Response(
            content=json.dumps({"error": "Task not found"}),
            status_code=404,
            media_type="application/json",
        )
    # 捕获其他异常
    except Exception:
        # 记录异常信息
        LOG.exception(f"Error whilst trying to get task: {task_id}")
        # 返回包含内部服务器错误信息的响应
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )
# 定义一个路由处理函数，用于获取特定任务关联的步骤列表
@base_router.get(
    "/agent/tasks/{task_id}/steps", tags=["agent"], response_model=TaskStepsListResponse
)
async def list_agent_task_steps(
    request: Request,
    task_id: str,
    page: Optional[int] = Query(1, ge=1),
    page_size: Optional[int] = Query(10, ge=1, alias="pageSize"),
) -> TaskStepsListResponse:
    """
    Retrieves a paginated list of steps associated with a specific task.

    Args:
        request (Request): FastAPI request object.
        task_id (str): The ID of the task.
        page (int, optional): The page number for pagination. Defaults to 1.
        page_size (int, optional): The number of steps per page for pagination. Defaults to 10.

    Returns:
        TaskStepsListResponse: A response object containing a list of steps and pagination details.

    Example:
        Request:
            GET /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/steps?page=1&pageSize=10

        Response (TaskStepsListResponse defined in schema.py):
            {
                "items": [
                    {
                        "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                        "step_id": "step1_id",
                        ...
                    },
                    ...
                ],
                "pagination": {
                    "total": 100,
                    "pages": 10,
                    "current": 1,
                    "pageSize": 10
                }
            }
    """
    # 从请求对象中获取代理信息
    agent = request["agent"]
    try:
        # 调用代理对象的方法，获取特定任务的步骤列表
        steps = await agent.list_steps(task_id, page, page_size)
        # 返回包含步骤列表的响应对象
        return Response(
            content=steps.json(),
            status_code=200,
            media_type="application/json",
        )
    except NotFoundError:
        # 如果出现未找到错误，记录异常信息并返回包含错误信息的响应对象
        LOG.exception("Error whilst trying to list steps")
        return Response(
            content=json.dumps({"error": "Steps not found"}),
            status_code=404,
            media_type="application/json",
        )
    # 捕获任何异常情况
    except Exception:
        # 记录异常信息到日志
        LOG.exception("Error whilst trying to list steps")
        # 返回一个包含错误信息的响应对象
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )
# 定义一个 POST 请求处理函数，用于执行指定任务的下一步，并返回执行的步骤及附加反馈字段
async def execute_agent_task_step(
    request: Request, task_id: str, step: Optional[StepRequestBody] = None
) -> Step:
    """
    Executes the next step for a specified task based on the current task status and returns the
    executed step with additional feedback fields.

    Depending on the current state of the task, the following scenarios are supported:

    1. No steps exist for the task.
    2. There is at least one step already for the task, and the task does not have a completed step marked as `last_step`.
    3. There is a completed step marked as `last_step` already on the task.

    In each of these scenarios, a step object will be returned with two additional fields: `output` and `additional_output`.
    - `output`: Provides the primary response or feedback to the user.
    - `additional_output`: Supplementary information or data. Its specific content is not strictly defined and can vary based on the step or agent's implementation.

    Args:
        request (Request): FastAPI request object.
        task_id (str): The ID of the task.
        step (StepRequestBody): The details for executing the step.

    Returns:
        Step: Details of the executed step with additional feedback.

    Example:
        Request:
            POST /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/steps
            {
                "input": "Step input details...",
                ...
            }

        Response:
            {
                "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                "step_id": "step1_id",
                "output": "Primary feedback...",
                "additional_output": "Supplementary details...",
                ...
            }
    """
    # 从请求对象中获取代理信息
    agent = request["agent"]
    # 尝试执行任务步骤
    try:
        # 如果步骤为空，则表示继续执行的确认
        if not step:
            step = StepRequestBody(input="y")

        # 执行任务步骤，并等待结果
        step = await agent.execute_step(task_id, step)
        # 返回执行结果的 JSON 格式响应
        return Response(
            content=step.json(),
            status_code=200,
            media_type="application/json",
        )
    # 如果任务未找到，则捕获 NotFoundError 异常
    except NotFoundError:
        # 记录异常信息
        LOG.exception(f"Error whilst trying to execute a task step: {task_id}")
        # 返回任务未找到的 JSON 格式响应
        return Response(
            content=json.dumps({"error": f"Task not found {task_id}"}),
            status_code=404,
            media_type="application/json",
        )
    # 捕获其他异常
    except Exception as e:
        # 记录异常信息
        LOG.exception(f"Error whilst trying to execute a task step: {task_id}")
        # 返回内部服务器错误的 JSON 格式响应
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )
# 定义一个异步函数，用于处理获取特定任务步骤的请求，返回步骤的详细信息
@base_router.get(
    "/agent/tasks/{task_id}/steps/{step_id}", tags=["agent"], response_model=Step
)
async def get_agent_task_step(request: Request, task_id: str, step_id: str) -> Step:
    """
    Retrieves the details of a specific step for a given task.

    Args:
        request (Request): FastAPI request object.
        task_id (str): The ID of the task.
        step_id (str): The ID of the step.

    Returns:
        Step: Details of the specific step.

    Example:
        Request:
            GET /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/steps/step1_id

        Response:
            {
                "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                "step_id": "step1_id",
                ...
            }
    """
    # 从请求对象中获取代理信息
    agent = request["agent"]
    try:
        # 尝试获取特定任务步骤的详细信息
        step = await agent.get_step(task_id, step_id)
        # 返回步骤的详细信息
        return Response(content=step.json(), status_code=200)
    except NotFoundError:
        # 如果步骤未找到，则记录异常并返回相应的错误响应
        LOG.exception(f"Error whilst trying to get step: {step_id}")
        return Response(
            content=json.dumps({"error": "Step not found"}),
            status_code=404,
            media_type="application/json",
        )
    except Exception:
        # 处理其他异常情况，记录异常并返回相应的错误响应
        LOG.exception(f"Error whilst trying to get step: {step_id}")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


# 定义一个异步函数，用于处理获取特定任务关联的工件列表的请求，返回工件列表的分页信息
@base_router.get(
    "/agent/tasks/{task_id}/artifacts",
    tags=["agent"],
    response_model=TaskArtifactsListResponse,
)
async def list_agent_task_artifacts(
    request: Request,
    task_id: str,
    page: Optional[int] = Query(1, ge=1),
    page_size: Optional[int] = Query(10, ge=1, alias="pageSize"),
) -> TaskArtifactsListResponse:
    """
    Retrieves a paginated list of artifacts associated with a specific task.
    """
    # 从 FastAPI 请求对象中获取代理信息，根据任务 ID、页码和每页数量获取任务工件列表，并返回包含工件列表和分页详情的响应对象
    async def get_task_artifacts(request: Request, task_id: str, page: int = 1, page_size: int = 10) -> TaskArtifactsListResponse:
        # 从请求对象中获取代理信息
        agent = request["agent"]
        try:
            # 调用代理对象的 list_artifacts 方法获取任务工件列表
            artifacts: TaskArtifactsListResponse = await agent.list_artifacts(
                task_id, page, page_size
            )
            # 返回获取到的任务工件列表
            return artifacts
        except NotFoundError:
            # 如果任务工件未找到，则记录异常并返回包含错误信息的 404 响应
            LOG.exception("Error whilst trying to list artifacts")
            return Response(
                content=json.dumps({"error": "Artifacts not found for task_id"}),
                status_code=404,
                media_type="application/json",
            )
        except Exception:
            # 处理其他异常情况，记录异常并返回包含错误信息的 500 响应
            LOG.exception("Error whilst trying to list artifacts")
            return Response(
                content=json.dumps({"error": "Internal server error"}),
                status_code=500,
                media_type="application/json",
            )
# 定义一个异步函数，用于处理上传代理任务的文件，返回上传的文件元数据
@base_router.post(
    "/agent/tasks/{task_id}/artifacts", tags=["agent"], response_model=Artifact
)
async def upload_agent_task_artifacts(
    request: Request, task_id: str, file: UploadFile, relative_path: Optional[str] = ""
) -> Artifact:
    """
    This endpoint is used to upload an artifact associated with a specific task. The artifact is provided as a file.

    Args:
        request (Request): The FastAPI request object.
        task_id (str): The unique identifier of the task for which the artifact is being uploaded.
        file (UploadFile): The file being uploaded as an artifact.
        relative_path (str): The relative path for the file. This is a query parameter.

    Returns:
        Artifact: An object containing metadata of the uploaded artifact, including its unique identifier.

    Example:
        Request:
            POST /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/artifacts?relative_path=my_folder/my_other_folder
            File: <uploaded_file>

        Response:
            {
                "artifact_id": "b225e278-8b4c-4f99-a696-8facf19f0e56",
                "created_at": "2023-01-01T00:00:00Z",
                "modified_at": "2023-01-01T00:00:00Z",
                "agent_created": false,
                "relative_path": "/my_folder/my_other_folder/",
                "file_name": "main.py"
            }
    """
    # 从请求中获取代理对象
    agent = request["agent"]

    # 如果文件为空，则返回错误响应
    if file is None:
        return Response(
            content=json.dumps({"error": "File must be specified"}),
            status_code=404,
            media_type="application/json",
        )
    try:
        # 调用代理对象的方法创建一个文件元数据对象
        artifact = await agent.create_artifact(task_id, file, relative_path)
        # 返回成功响应
        return Response(
            content=artifact.json(),
            status_code=200,
            media_type="application/json",
        )
    # 捕获任何异常并记录异常信息，包括尝试上传 artifact 的任务 ID
    except Exception:
        LOG.exception(f"Error whilst trying to upload artifact: {task_id}")
        # 返回一个包含错误信息的响应对象，状态码为 500，媒体类型为 JSON
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )
# 定义一个路由处理函数，用于下载与特定任务相关的文件
@base_router.get(
    "/agent/tasks/{task_id}/artifacts/{artifact_id}", tags=["agent"], response_model=str
)
async def download_agent_task_artifact(
    request: Request, task_id: str, artifact_id: str
) -> FileResponse:
    """
    Downloads an artifact associated with a specific task.

    Args:
        request (Request): FastAPI request object. # 请求对象
        task_id (str): The ID of the task. # 任务的ID
        artifact_id (str): The ID of the artifact. # 文件的ID

    Returns:
        FileResponse: The downloaded artifact file. # 下载的文件响应对象

    Example:
        Request:
            GET /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/artifacts/artifact1_id

        Response:
            <file_content_of_artifact>
    """
    # 从请求中获取代理对象
    agent = request["agent"]
    try:
        # 尝试获取任务ID和文件ID对应的文件
        return await agent.get_artifact(task_id, artifact_id)
    except NotFoundError:
        # 如果文件未找到，则记录错误并返回相应
        LOG.exception(f"Error whilst trying to download artifact: {task_id}")
        return Response(
            content=json.dumps(
                {
                    "error": f"Artifact not found - task_id: {task_id}, artifact_id: {artifact_id}"
                }
            ),
            status_code=404,
            media_type="application/json",
        )
    except Exception:
        # 处理其他异常情况，记录错误并返回相应
        LOG.exception(f"Error whilst trying to download artifact: {task_id}")
        return Response(
            content=json.dumps(
                {
                    "error": f"Internal server error - task_id: {task_id}, artifact_id: {artifact_id}"
                }
            ),
            status_code=500,
            media_type="application/json",
        )
```