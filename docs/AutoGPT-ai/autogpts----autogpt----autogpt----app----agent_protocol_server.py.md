# `.\AutoGPT\autogpts\autogpt\autogpt\app\agent_protocol_server.py`

```py
# 导入日志、操作系统、路径、字节流、唯一标识等模块
import logging
import os
import pathlib
from io import BytesIO
from uuid import uuid4

# 导入 JSON 序列化模块、FastAPI 框架、上传文件、跨域中间件、重定向响应、流响应、静态文件等模块
import orjson
from fastapi import APIRouter, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# 导入 Forge SDK 中的数据库、错误、中间件、模型等模块
from forge.sdk.db import AgentDB
from forge.sdk.errors import NotFoundError
from forge.sdk.middlewares import AgentMiddleware
from forge.sdk.model import (
    Artifact,
    Step,
    StepRequestBody,
    Task,
    TaskArtifactsListResponse,
    TaskListResponse,
    TaskRequestBody,
    TaskStepsListResponse,
)
from forge.sdk.routes.agent_protocol import base_router

# 导入 Hypercorn 中的异步服务和配置模块
from hypercorn.asyncio import serve as hypercorn_serve
from hypercorn.config import Config as HypercornConfig

# 导入自动生成 GPT 相关模块
from autogpt.agent_factory.configurators import configure_agent_with_state
from autogpt.agent_factory.generators import generate_agent_for_task
from autogpt.agent_manager import AgentManager
from autogpt.commands.system import finish
from autogpt.commands.user_interaction import ask_user
from autogpt.config import Config
from autogpt.core.resource.model_providers import ChatModelProvider
from autogpt.core.resource.model_providers.openai import OpenAIProvider
from autogpt.core.resource.model_providers.schema import ModelProviderBudget
from autogpt.file_workspace import (
    FileWorkspace,
    FileWorkspaceBackendName,
    get_workspace,
)
from autogpt.logs.utils import fmt_kwargs
from autogpt.models.action_history import ActionErrorResult, ActionSuccessResult

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义 AgentProtocolServer 类
class AgentProtocolServer:
    # 任务预算字典，键为任务 ID，值为模型提供者预算
    _task_budgets: dict[str, ModelProviderBudget]

    # 初始化方法，接受应用配置、数据库、语言模型提供者等参数
    def __init__(
        self,
        app_config: Config,
        database: AgentDB,
        llm_provider: ChatModelProvider,
        ):
        # 初始化应用配置
        self.app_config = app_config
        # 初始化数据库
        self.db = database
        # 初始化LLM提供者
        self.llm_provider = llm_provider
        # 初始化代理管理器，传入应用数据目录
        self.agent_manager = AgentManager(app_data_dir=app_config.app_data_dir)
        # 初始化任务预算字典
        self._task_budgets = {}
    # 异步方法，用于启动代理服务器
    async def start(self, port: int = 8000, router: APIRouter = base_router):
        """Start the agent server."""
        # 调试信息，表示正在启动代理服务器
        logger.debug("Starting the agent server...")
        # 创建 Hypercorn 配置对象
        config = HypercornConfig()
        # 绑定端口号
        config.bind = [f"localhost:{port}"]
        # 创建 FastAPI 应用
        app = FastAPI(
            title="AutoGPT Server",
            description="Forked from AutoGPT Forge; "
            "Modified version of The Agent Protocol.",
            version="v0.4",
        )

        # 添加 CORS 中间件
        origins = [
            "*",
            # 添加要允许的其他来源
        ]

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 包含路由器
        app.include_router(router, prefix="/ap/v1")
        # 获取当前脚本的目录路径
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # 前端路径
        frontend_path = (
            pathlib.Path(script_dir)
            .joinpath("../../../../frontend/build/web")
            .resolve()
        )

        # 如果前端路径存在，则挂载静态文件
        if os.path.exists(frontend_path):
            app.mount("/app", StaticFiles(directory=frontend_path), name="app")

            # 根路径重定向到前端页面
            @app.get("/", include_in_schema=False)
            async def root():
                return RedirectResponse(url="/app/index.html", status_code=307)

        else:
            # 警告信息，表示前端文件不存在
            logger.warning(
                f"Frontend not found. {frontend_path} does not exist. "
                "The frontend will not be available."
            )

        # 用于从 API 路由处理程序访问此类的方法
        app.add_middleware(AgentMiddleware, agent=self)

        # 设置日志级别和绑定地址
        config.loglevel = "ERROR"
        config.bind = [f"0.0.0.0:{port}"]

        # 信息日志，表示 AutoGPT 服务器正在启动
        logger.info(f"AutoGPT server starting on http://localhost:{port}")
        # 启动 Hypercorn 服务器
        await hypercorn_serve(app, config)
    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        Create a task for the agent.
        """
        # 调用数据库方法创建一个任务
        task = await self.db.create_task(
            input=task_request.input,
            additional_input=task_request.additional_input,
        )
        # 记录创建任务的日志
        logger.debug(f"Creating agent for task: '{task.input}'")
        # 为任务生成一个代理
        task_agent = await generate_agent_for_task(
            task=task.input,
            app_config=self.app_config,
            llm_provider=self._get_task_llm_provider(task),
        )

        # 为代理分配一个ID和文件夹，并持久化
        agent_id = task_agent.state.agent_id = task_agent_id(task.task_id)
        logger.debug(f"New agent ID: {agent_id}")
        task_agent.attach_fs(self.app_config.app_data_dir / "agents" / agent_id)
        task_agent.state.save_to_json_file(task_agent.file_manager.state_file_path)

        return task

    async def list_tasks(self, page: int = 1, pageSize: int = 10) -> TaskListResponse:
        """
        List all tasks that the agent has created.
        """
        # 记录列出所有任务的日志
        logger.debug("Listing all tasks...")
        # 调用数据库方法列出所有任务
        tasks, pagination = await self.db.list_tasks(page, pageSize)
        response = TaskListResponse(tasks=tasks, pagination=pagination)
        return response

    async def get_task(self, task_id: str) -> Task:
        """
        Get a task by ID.
        """
        # 根据任务ID获取任务
        logger.debug(f"Getting task with ID: {task_id}...")
        task = await self.db.get_task(task_id)
        return task

    async def list_steps(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskStepsListResponse:
        """
        List the IDs of all steps that the task has created.
        """
        # 记录列出任务创建的所有步骤的日志
        logger.debug(f"Listing all steps created by task with ID: {task_id}...")
        # 调用数据库方法列出任务创建的所有步骤
        steps, pagination = await self.db.list_steps(task_id, page, pageSize)
        response = TaskStepsListResponse(steps=steps, pagination=pagination)
        return response
    # 异步方法，用于在代理写入文件时创建或更新文件的 Artifact
    async def _on_agent_write_file(
        self, task: Task, step: Step, relative_path: pathlib.Path
    ) -> None:
        """
        创建一个 Artifact 用于写入的文件，如果存在则更新 Artifact。
        """
        # 检查相对路径是否为绝对路径，如果是则抛出数值错误
        if relative_path.is_absolute():
            raise ValueError(f"File path '{relative_path}' is not relative")
        # 遍历任务的所有 Artifact
        for a in task.artifacts or []:
            # 如果 Artifact 的相对路径与给定相对路径相同
            if a.relative_path == str(relative_path):
                # 记录日志，更新已存在文件的 Artifact
                logger.debug(f"Updating Artifact after writing to existing file: {a}")
                # 如果 Artifact 尚未由代理创建，则更新为已创建
                if not a.agent_created:
                    await self.db.update_artifact(a.artifact_id, agent_created=True)
                break
        else:
            # 记录日志，为新文件创建 Artifact
            logger.debug(f"Creating Artifact for new file '{relative_path}'")
            # 创建新的 Artifact
            await self.db.create_artifact(
                task_id=step.task_id,
                step_id=step.step_id,
                file_name=relative_path.parts[-1],
                agent_created=True,
                relative_path=str(relative_path),
            )

    # 异步方法，通过 ID 获取一个步骤
    async def get_step(self, task_id: str, step_id: str) -> Step:
        """
        通过 ID 获取一个步骤。
        """
        # 从数据库中获取步骤信息
        step = await self.db.get_step(task_id, step_id)
        return step

    # 异步方法，列出任务创建的所有 Artifact
    async def list_artifacts(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskArtifactsListResponse:
        """
        列出任务创建的所有 Artifact。
        """
        # 从数据库中获取任务的 Artifact 列表和分页信息
        artifacts, pagination = await self.db.list_artifacts(task_id, page, pageSize)
        return TaskArtifactsListResponse(artifacts=artifacts, pagination=pagination)

    # 异步方法，创建一个 Artifact
    async def create_artifact(
        self, task_id: str, file: UploadFile, relative_path: str
    ) -> Artifact:
        """
        Create an artifact for the task.
        """
        # 如果文件名为空，则使用随机生成的 UUID 作为文件名
        file_name = file.filename or str(uuid4())
        # 初始化数据为空字节串
        data = b""
        # 逐块读取文件内容，每次读取 1MB，拼接到数据中
        while contents := file.file.read(1024 * 1024):
            data += contents
        # 检查相对路径是否以文件名结尾
        if relative_path.endswith(file_name):
            file_path = relative_path
        else:
            file_path = os.path.join(relative_path, file_name)

        # 获取任务代理文件工作空间，写入文件数据
        workspace = self._get_task_agent_file_workspace(task_id, self.agent_manager)
        await workspace.write_file(file_path, data)

        # 创建任务的 artifact，并返回
        artifact = await self.db.create_artifact(
            task_id=task_id,
            file_name=file_name,
            relative_path=relative_path,
            agent_created=False,
        )
        return artifact

    async def get_artifact(self, task_id: str, artifact_id: str) -> StreamingResponse:
        """
        Download a task artifact by ID.
        """
        try:
            # 获取指定 artifact
            artifact = await self.db.get_artifact(artifact_id)
            # 如果文件名不在相对路径中，则拼接文件路径
            if artifact.file_name not in artifact.relative_path:
                file_path = os.path.join(artifact.relative_path, artifact.file_name)
            else:
                file_path = artifact.relative_path
            # 获取任务代理文件工作空间，读取文件数据
            workspace = self._get_task_agent_file_workspace(task_id, self.agent_manager)
            retrieved_artifact = workspace.read_file(file_path, binary=True)
        except NotFoundError:
            raise
        except FileNotFoundError:
            raise

        # 返回文件流响应
        return StreamingResponse(
            BytesIO(retrieved_artifact),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{artifact.file_name}"'
            },
        )

    def _get_task_agent_file_workspace(
        self,
        task_id: str | int,
        agent_manager: AgentManager,
    # 返回一个 FileWorkspace 对象
    ) -> FileWorkspace:
        # 检查是否使用本地工作空间
        use_local_ws = (
            self.app_config.workspace_backend == FileWorkspaceBackendName.LOCAL
        )
        # 获取任务代理的 ID
        agent_id = task_agent_id(task_id)
        # 获取工作空间对象
        workspace = get_workspace(
            backend=self.app_config.workspace_backend,
            id=agent_id if not use_local_ws else "",
            root_path=agent_manager.get_agent_dir(
                agent_id=agent_id,
                must_exist=True,
            )
            / "workspace"
            if use_local_ws
            else None,
        )
        # 初始化工作空间
        workspace.initialize()
        # 返回工作空间对象
        return workspace

    # 获取任务的 LLM 提供者
    def _get_task_llm_provider(
        self, task: Task, step_id: str = ""
    ) -> ChatModelProvider:
        """
        为 LLM 提供程序配置头部，以将传出请求与任务关联起来。
        """
        # 获取任务的 LLM 预算，如果不存在则使用默认设置的预算
        task_llm_budget = self._task_budgets.get(
            task.task_id, self.llm_provider.default_settings.budget.copy(deep=True)
        )

        # 复制 LLM 提供程序的配置
        task_llm_provider_config = self.llm_provider._configuration.copy(deep=True)
        _extra_request_headers = task_llm_provider_config.extra_request_headers
        # 设置额外的请求头部，包括任务 ID
        _extra_request_headers["AP-TaskID"] = task.task_id
        if step_id:
            _extra_request_headers["AP-StepID"] = step_id
        if task.additional_input and (user_id := task.additional_input.get("user_id")):
            _extra_request_headers["AutoGPT-UserID"] = user_id

        task_llm_provider = None
        # 如果 LLM 提供程序是 OpenAIProvider 类型
        if isinstance(self.llm_provider, OpenAIProvider):
            settings = self.llm_provider._settings.copy()
            settings.budget = task_llm_budget
            settings.configuration = task_llm_provider_config  # type: ignore
            # 创建一个新的 OpenAIProvider 实例
            task_llm_provider = OpenAIProvider(
                settings=settings,
                logger=logger.getChild(f"Task-{task.task_id}_OpenAIProvider"),
            )

        # 如果任务的 LLM 提供程序存在且有预算，则更新任务的预算
        if task_llm_provider and task_llm_provider._budget:
            self._task_budgets[task.task_id] = task_llm_provider._budget

        # 返回任务的 LLM 提供程序，如果不存在则返回默认的 LLM 提供程序
        return task_llm_provider or self.llm_provider
# 定义一个函数，用于生成任务代理ID，接受一个字符串或整数类型的参数，返回一个字符串类型的结果
def task_agent_id(task_id: str | int) -> str:
    # 使用 f-string 格式化字符串，生成任务代理ID，格式为 "AutoGPT-任务ID"
    return f"AutoGPT-{task_id}"
```