# `.\AutoGPT\benchmark\agbenchmark\app.py`

```py
# 导入所需的模块
import datetime
import glob
import json
import logging
import sys
import time
import uuid
from collections import deque
from multiprocessing import Process
from pathlib import Path
from typing import Optional

import httpx
import psutil
from agent_protocol_client import AgentApi, ApiClient, ApiException, Configuration
from agent_protocol_client.models import Task, TaskRequestBody
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Extra, ValidationError

# 导入自定义模块
from agbenchmark.challenges import ChallengeInfo
from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.reports.processing.report_types_v2 import (
    BenchmarkRun,
    Metrics,
    RepositoryInfo,
    RunDetails,
    TaskInfo,
)
from agbenchmark.schema import TaskEvalRequestBody
from agbenchmark.utils.utils import write_pretty_json

# 将当前文件的父目录添加到系统路径中
sys.path.append(str(Path(__file__).parent.parent))

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义一个空字典用于存储挑战信息
CHALLENGES: dict[str, ChallengeInfo] = {}
# 获取挑战文件所在路径
challenges_path = Path(__file__).parent / "challenges"
# 使用双向队列存储挑战规范文件的路径
challenge_spec_files = deque(
    glob.glob(
        f"{challenges_path}/**/data.json",
        recursive=True,
    )
)

# 记录日志，表示正在加载挑战信息
logger.debug("Loading challenges...")
# 遍历挑战规范文件路径
while challenge_spec_files:
    # 获取当前挑战规范文件的路径
    challenge_spec_file = Path(challenge_spec_files.popleft())
    # 获取挑战规范文件相对于挑战路径的相对路径
    challenge_relpath = challenge_spec_file.relative_to(challenges_path.parent)
    # 如果挑战规范文件在"challenges/deprecated"目录下，则跳过
    if challenge_relpath.is_relative_to("challenges/deprecated"):
        continue

    # 记录日志，表示正在加载当前挑战信息
    logger.debug(f"Loading {challenge_relpath}...")
    try:
        # 解析挑战规范文件，获取挑战信息
        challenge_info = ChallengeInfo.parse_file(challenge_spec_file)
    except ValidationError as e:
        # 如果解析出错，记录警告日志
        if logging.getLogger().level == logging.DEBUG:
            logger.warning(f"Spec file {challenge_relpath} failed to load:\n{e}")
        # 记录调试日志，显示无效的挑战规范文件内容
        logger.debug(f"Invalid challenge spec: {challenge_spec_file.read_text()}")
        continue
    # 将挑战规范文件路径存储到挑战信息中
    challenge_info.spec_file = challenge_spec_file
    # 如果评估ID为空，则生成一个UUID作为评估ID
    if not challenge_info.eval_id:
        challenge_info.eval_id = str(uuid.uuid4())
        # 这将按照系统化的方式对JSON的所有键进行排序
        # 以确保顺序始终相同
        write_pretty_json(challenge_info.dict(), challenge_spec_file)

    # 将评估ID和对应的挑战信息添加到CHALLENGES字典中
    CHALLENGES[challenge_info.eval_id] = challenge_info
# 定义 BenchmarkTaskInfo 类，包含任务 ID、开始时间和挑战信息
class BenchmarkTaskInfo(BaseModel):
    task_id: str
    start_time: datetime.datetime
    challenge_info: ChallengeInfo

# 创建一个空字典，用于存储任务信息
task_informations: dict[str, BenchmarkTaskInfo] = {}

# 查找没有 uvicorn 的 agbenchmark 进程
def find_agbenchmark_without_uvicorn():
    # 存储符合条件的进程 ID
    pids = []
    # 遍历所有进程
    for process in psutil.process_iter(
        attrs=[
            "pid",
            "cmdline",
            "name",
            "username",
            "status",
            "cpu_percent",
            "memory_info",
            "create_time",
            "cwd",
            "connections",
        ]
    ):
        try:
            # 将进程信息字典的值转换为字符串并连接起来
            full_info = " ".join([str(v) for k, v in process.as_dict().items()])

            # 如果进程信息中包含 "agbenchmark" 且不包含 "uvicorn"，则将进程 ID 添加到列表中
            if "agbenchmark" in full_info and "uvicorn" not in full_info:
                pids.append(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return pids

# 定义 CreateReportRequest 类，包含测试名称、测试运行 ID、是否模拟等信息
class CreateReportRequest(BaseModel):
    test: str = None
    test_run_id: str = None
    mock: Optional[bool] = False

    class Config:
        extra = Extra.forbid  # 禁止额外字段

# 初始化更新列表
updates_list = []

# 定义允许的来源列表
origins = [
    "http://localhost:8000",
    "http://localhost:8080",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
]

# 定义函数 stream_output，用于输出管道中的内容
def stream_output(pipe):
    for line in pipe:
        print(line, end="")

# 设置 FastAPI 应用程序
def setup_fastapi_app(agbenchmark_config: AgentBenchmarkConfig) -> FastAPI:
    # 导入所需模块和函数
    from agbenchmark.agent_api_interface import upload_artifacts
    from agbenchmark.challenges import get_challenge_from_source_uri
    from agbenchmark.main import run_benchmark

    # 配置信息
    configuration = Configuration(
        host=agbenchmark_config.host or "http://localhost:8000"
    )
    # 创建 FastAPI 应用程序
    app = FastAPI()
    # 添加 CORS 中间件，允许指定的来源、方法和头部信息
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # 创建一个APIRouter对象，用于定义路由
    router = APIRouter()

    # 定义一个POST请求处理函数，用于处理"/reports"路由
    # 同时也定义了"/agent/tasks"和"/agent/tasks/{task_id}/steps"路由，设置了标签为"agent"
    async def proxy(request: Request, task_id: str):
        # 设置请求超时时间为5分钟
        timeout = httpx.Timeout(300.0, read=300.0)  # 5 minutes
        # 创建一个异步HTTP客户端对象
        async with httpx.AsyncClient(timeout=timeout) as client:
            # 构建新的URL
            new_url = f"{configuration.host}/ap/v1/agent/tasks/{task_id}/steps"

            # 转发请求
            response = await client.post(
                new_url,
                data=await request.body(),
                headers=dict(request.headers),
            )

            # 返回转发请求的响应
            return Response(content=response.content, status_code=response.status_code)

    # 定义一个"/agent/tasks/{task_id}/evaluations"的POST请求处理函数
    @router.post("/agent/tasks/{task_id}/evaluations")
    # 异步函数，用于创建评估任务并返回 BenchmarkRun 对象
    async def create_evaluation(task_id: str) -> BenchmarkRun:
        # 获取任务信息
        task_info = task_informations[task_id]
        # 从任务信息中获取挑战信息
        challenge = get_challenge_from_source_uri(task_info.challenge_info.source_uri)
        try:
            # 使用配置创建 ApiClient，并异步打开
            async with ApiClient(configuration) as api_client:
                # 创建 AgentApi 实例
                api_instance = AgentApi(api_client)
                # 异步评估任务状态
                eval_results = await challenge.evaluate_task_state(
                    api_instance, task_id
                )

            # 构建 BenchmarkRun 对象
            eval_info = BenchmarkRun(
                repository_info=RepositoryInfo(),
                run_details=RunDetails(
                    command=f"agbenchmark --test={challenge.info.name}",
                    benchmark_start_time=(
                        task_info.start_time.strftime("%Y-%m-%dT%H:%M:%S+00:00")
                    ),
                    test_name=challenge.info.name,
                ),
                task_info=TaskInfo(
                    data_path=challenge.info.source_uri,
                    is_regression=None,
                    category=[c.value for c in challenge.info.category],
                    task=challenge.info.task,
                    answer=challenge.info.reference_answer or "",
                    description=challenge.info.description or "",
                ),
                metrics=Metrics(
                    success=all(e.passed for e in eval_results),
                    success_percentage=(
                        100 * sum(e.score for e in eval_results) / len(eval_results)
                        if eval_results  # 避免除以 0
                        else 0
                    ),
                    attempted=True,
                ),
                config={},
            )

            # 记录调试信息
            logger.debug(f"Returning evaluation data:\n{eval_info.json(indent=4)}")
            # 返回评估信息
            return eval_info
        except ApiException as e:
            # 捕获异常并记录错误信息
            logger.error(f"Error {e} whilst trying to evaluate task: {task_id}")
            # 抛出 HTTP 异常
            raise HTTPException(500)
    # 将路由器 router 包含到应用程序中，并指定路由前缀为 "/ap/v1"
    app.include_router(router, prefix="/ap/v1")

    # 返回包含路由器的应用程序
    return app
```