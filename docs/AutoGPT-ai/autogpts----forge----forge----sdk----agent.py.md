# `.\AutoGPT\autogpts\forge\forge\sdk\agent.py`

```py
# 导入必要的模块
import os
import pathlib
from io import BytesIO
from uuid import uuid4

# 导入 FastAPI 框架相关模块
import uvicorn
from fastapi import APIRouter, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# 导入自定义模块
from .db import AgentDB
from .errors import NotFoundError
from .forge_log import ForgeLogger
from .middlewares import AgentMiddleware
from .model import (
    Artifact,
    Step,
    StepRequestBody,
    Task,
    TaskArtifactsListResponse,
    TaskListResponse,
    TaskRequestBody,
    TaskStepsListResponse,
)
from .routes.agent_protocol import base_router
from .workspace import Workspace

# 创建日志对象
LOG = ForgeLogger(__name__)

# 定义 Agent 类
class Agent:
    # 初始化 Agent 类
    def __init__(self, database: AgentDB, workspace: Workspace):
        # 初始化数据库对象
        self.db = database
        # 初始化工作空间对象
        self.workspace = workspace
    # 获取代理应用程序，可以传入自定义的路由器，默认为基础路由器
    def get_agent_app(self, router: APIRouter = base_router):
        """
        Start the agent server.
        """

        # 创建 FastAPI 应用程序实例，设置标题、描述和版本号
        app = FastAPI(
            title="AutoGPT Forge",
            description="Modified version of The Agent Protocol.",
            version="v0.4",
        )

        # 添加 CORS 中间件，设置允许的来源列表
        origins = [
            "http://localhost:5000",
            "http://127.0.0.1:5000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            # 添加其他需要允许的来源
        ]

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 将路由器包含到应用程序中，设置路由前缀
        app.include_router(router, prefix="/ap/v1")
        script_dir = os.path.dirname(os.path.realpath(__file__))
        frontend_path = pathlib.Path(
            os.path.join(script_dir, "../../../../frontend/build/web")
        ).resolve()

        # 如果前端路径存在，则挂载静态文件服务
        if os.path.exists(frontend_path):
            app.mount("/app", StaticFiles(directory=frontend_path), name="app")

            # 定义根路由，重定向到前端页面
            @app.get("/", include_in_schema=False)
            async def root():
                return RedirectResponse(url="/app/index.html", status_code=307)

        else:
            # 如果前端路径不存在，则记录警告信息
            LOG.warning(
                f"Frontend not found. {frontend_path} does not exist. The frontend will not be served"
            )
        
        # 添加代理中间件到应用程序
        app.add_middleware(AgentMiddleware, agent=self)

        return app

    # 启动应用程序，使用 uvicorn 运行应用程序
    def start(self, port):
        uvicorn.run(
            "forge.app:app", host="localhost", port=port, log_level="error", reload=True
        )
    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        Create a task for the agent.
        """
        # 尝试创建一个任务
        try:
            # 调用数据库对象的create_task方法创建任务
            task = await self.db.create_task(
                input=task_request.input,
                additional_input=task_request.additional_input,
            )
            # 返回创建的任务
            return task
        except Exception as e:
            # 如果出现异常则抛出
            raise

    async def list_tasks(self, page: int = 1, pageSize: int = 10) -> TaskListResponse:
        """
        List all tasks that the agent has created.
        """
        # 尝试列出所有代理创建的任务
        try:
            # 调用数据库对象的list_tasks方法获取任务列表和分页信息
            tasks, pagination = await self.db.list_tasks(page, pageSize)
            # 创建任务列表响应对象
            response = TaskListResponse(tasks=tasks, pagination=pagination)
            # 返回响应对象
            return response
        except Exception as e:
            # 如果出现异常则抛出
            raise

    async def get_task(self, task_id: str) -> Task:
        """
        Get a task by ID.
        """
        # 尝试通过ID获取任务
        try:
            # 调用数据库对象的get_task方法获取任务
            task = await self.db.get_task(task_id)
        except Exception as e:
            # 如果出现异常则抛出
            raise
        # 返回获取的任务
        return task

    async def list_steps(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskStepsListResponse:
        """
        List the IDs of all steps that the task has created.
        """
        # 尝试列出任务创建的所有步骤的ID
        try:
            # 调用数据库对象的list_steps方法获取步骤列表和分页信息
            steps, pagination = await self.db.list_steps(task_id, page, pageSize)
            # 创建步骤列表响应对象
            response = TaskStepsListResponse(steps=steps, pagination=pagination)
            # 返回响应对象
            return response
        except Exception as e:
            # 如果出现异常则抛出
            raise

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """
        Create a step for the task.
        """
        # 抛出未实现的错误，表示该方法需要在子类中实现
        raise NotImplementedError

    async def get_step(self, task_id: str, step_id: str) -> Step:
        """
        Get a step by ID.
        """
        # 尝试通过ID获取步骤
        try:
            # 调用数据库对象的get_step方法获取步骤
            step = await self.db.get_step(task_id, step_id)
            # 返回获取的步骤
            return step
        except Exception as e:
            # 如果出现异常则抛出
            raise
    # 异步函数，用于列出任务创建的所有工件
    async def list_artifacts(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskArtifactsListResponse:
        """
        列出任务创建的工件。
        """
        try:
            # 调用数据库方法列出任务的工件和分页信息
            artifacts, pagination = await self.db.list_artifacts(
                task_id, page, pageSize
            )
            # 返回包含工件和分页信息的响应对象
            return TaskArtifactsListResponse(artifacts=artifacts, pagination=pagination)

        except Exception as e:
            # 捕获异常并重新抛出
            raise

    # 异步函数，用于为任务创建工件
    async def create_artifact(
        self, task_id: str, file: UploadFile, relative_path: str
    ) -> Artifact:
        """
        为任务创建工件。
        """
        data = None
        # 获取文件名，如果没有则生成一个随机的文件名
        file_name = file.filename or str(uuid4())
        try:
            # 初始化数据为空字节串
            data = b""
            # 读取文件内容并拼接到数据中，每次读取 1MB
            while contents := file.file.read(1024 * 1024):
                data += contents
            # 检查相对路径是否以文件名结尾
            if relative_path.endswith(file_name):
                file_path = relative_path
            else:
                file_path = os.path.join(relative_path, file_name)

            # 将数据写入工作空间
            self.workspace.write(task_id, file_path, data)

            # 调用数据库方法创建工件
            artifact = await self.db.create_artifact(
                task_id=task_id,
                file_name=file_name,
                relative_path=relative_path,
                agent_created=False,
            )
        except Exception as e:
            # 捕获异常并重新抛出
            raise
        # 返回创建的工件对象
        return artifact
    # 异步函数，通过任务ID和artifact ID获取artifact对象
    async def get_artifact(self, task_id: str, artifact_id: str) -> Artifact:
        """
        Get an artifact by ID.
        """
        # 尝试从数据库中获取artifact对象
        try:
            artifact = await self.db.get_artifact(artifact_id)
            # 如果artifact的文件名不在相对路径中，则拼接文件路径
            if artifact.file_name not in artifact.relative_path:
                file_path = os.path.join(artifact.relative_path, artifact.file_name)
            else:
                file_path = artifact.relative_path
            # 从工作空间中读取指定任务ID和文件路径的artifact数据
            retrieved_artifact = self.workspace.read(task_id=task_id, path=file_path)
        except NotFoundError as e:
            # 如果出现NotFoundError异常，则抛出
            raise
        except FileNotFoundError as e:
            # 如果出现FileNotFoundError异常，则抛出
            raise
        except Exception as e:
            # 如果出现其他异常，则抛出
            raise

        # 返回StreamingResponse对象，包含artifact数据的字节流、媒体类型和文件名
        return StreamingResponse(
            BytesIO(retrieved_artifact),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={artifact.file_name}"
            },
        )
```