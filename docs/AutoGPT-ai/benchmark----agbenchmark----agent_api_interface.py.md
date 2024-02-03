# `.\AutoGPT\benchmark\agbenchmark\agent_api_interface.py`

```py
# 导入日志、操作系统、时间、路径等必要模块
import logging
import os
import time
from pathlib import Path
from typing import AsyncIterator, Optional

# 导入AgentApi、ApiClient、Configuration、Step、TaskRequestBody等类
from agent_protocol_client import (
    AgentApi,
    ApiClient,
    Configuration,
    Step,
    TaskRequestBody,
)

# 导入获取文件路径列表和AgentBenchmarkConfig类
from agbenchmark.agent_interface import get_list_of_file_paths
from agbenchmark.config import AgentBenchmarkConfig

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 异步函数，运行API代理
async def run_api_agent(
    task: str,
    config: AgentBenchmarkConfig,
    timeout: int,
    artifacts_location: Optional[Path] = None,
) -> AsyncIterator[Step]:
    # 创建API配置对象
    configuration = Configuration(host=config.host)
    # 使用API客户端进行异步操作
    async with ApiClient(configuration) as api_client:
        # 创建AgentApi实例
        api_instance = AgentApi(api_client)
        # 创建任务请求体
        task_request_body = TaskRequestBody(input=task)

        # 记录开始时间
        start_time = time.time()
        # 创建代理任务
        response = await api_instance.create_agent_task(
            task_request_body=task_request_body
        )
        # 获取任务ID
        task_id = response.task_id

        # 如果有存储位置，上传文件
        if artifacts_location:
            await upload_artifacts(
                api_instance, artifacts_location, task_id, "artifacts_in"
            )

        # 循环执行代理任务步骤
        while True:
            step = await api_instance.execute_agent_task_step(task_id=task_id)
            yield step

            # 如果超时，抛出超时错误
            if time.time() - start_time > timeout:
                raise TimeoutError("Time limit exceeded")
            # 如果步骤为空或为最后一步，跳出循环
            if not step or step.is_last:
                break

        # 如果有存储位置，根据环境变量上传或下载文件
        if artifacts_location:
            # 在“mock”模式下，通过上传正确的文件来欺骗测试
            if os.getenv("IS_MOCK"):
                await upload_artifacts(
                    api_instance, artifacts_location, task_id, "artifacts_out"
                )

            # 下载代理任务的文件到指定文件夹
            await download_agent_artifacts_into_folder(
                api_instance, task_id, config.temp_folder
            )

# 异步函数，将代理任务的文件下载到指定文件夹
async def download_agent_artifacts_into_folder(
    api_instance: AgentApi, task_id: str, folder: Path
):
    # 获取代理任务的文件列表
    artifacts = await api_instance.list_agent_task_artifacts(task_id=task_id)
    # 遍历artifacts.artifacts列表中的每个artifact对象
    for artifact in artifacts.artifacts:
        # 如果artifact对象有相对路径
        if artifact.relative_path:
            # 将相对路径存储在变量path中，如果路径不是以"/"开头，则保持不变，否则去掉开头的"/"
            path: str = (
                artifact.relative_path
                if not artifact.relative_path.startswith("/")
                else artifact.relative_path[1:]
            )
            # 更新文件夹路径为当前路径加上相对路径的父目录
            folder = (folder / path).parent

        # 如果文件夹路径不存在，则创建文件夹
        if not folder.exists():
            folder.mkdir(parents=True)

        # 构建文件的完整路径
        file_path = folder / artifact.file_name
        # 记录下载agent artifact的日志信息
        logger.debug(f"Downloading agent artifact {artifact.file_name} to {folder}")
        # 以二进制写模式打开文件
        with open(file_path, "wb") as f:
            # 从API实例中下载agent任务artifact的内容
            content = await api_instance.download_agent_task_artifact(
                task_id=task_id, artifact_id=artifact.artifact_id
            )
            # 将内容写入文件
            f.write(content)
# 异步函数，用于上传构件
async def upload_artifacts(
    api_instance: AgentApi, artifacts_location: Path, task_id: str, type: str
) -> None:
    # 遍历指定类型的构件文件路径列表
    for file_path in get_list_of_file_paths(artifacts_location, type):
        # 获取构件文件相对路径
        relative_path: Optional[str] = "/".join(
            str(file_path).split(f"{type}/", 1)[-1].split("/")[:-1]
        )
        # 如果相对路径为空，则设为 None
        if not relative_path:
            relative_path = None

        # 调用 API 实例的上传构件方法，传入任务 ID、文件路径和相对路径
        await api_instance.upload_agent_task_artifacts(
            task_id=task_id, file=str(file_path), relative_path=relative_path
        )
```