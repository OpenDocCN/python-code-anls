# `.\DB-GPT-src\dbgpt\model\cluster\worker\remote_worker.py`

```py
import json  # 导入json模块，用于处理JSON数据
import logging  # 导入logging模块，用于日志记录
from typing import Dict, Iterator, List  # 导入类型提示相关的模块

from dbgpt.core import ModelMetadata, ModelOutput  # 导入dbgpt核心模块中的ModelMetadata和ModelOutput类
from dbgpt.model.cluster.worker_base import ModelWorker  # 导入dbgpt集群模型中的ModelWorker类
from dbgpt.model.parameter import ModelParameters  # 导入dbgpt模型参数相关模块
from dbgpt.util.tracer import DBGPT_TRACER_SPAN_ID, root_tracer  # 导入dbgpt追踪相关模块中的常量和函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class RemoteModelWorker(ModelWorker):
    def __init__(self) -> None:
        self.headers = {}  # 初始化headers为空字典
        # TODO Configured by ModelParameters
        self.timeout = 3600  # 初始化超时时间为3600秒
        self.host = None  # 初始化主机地址为None
        self.port = None  # 初始化端口号为None

    @property
    def worker_addr(self) -> str:
        return f"http://{self.host}:{self.port}/api/worker"
        # 返回拼接后的worker地址字符串，格式为"http://主机地址:端口号/api/worker"

    def support_async(self) -> bool:
        return True
        # 返回True，表示支持异步操作

    def parse_parameters(self, command_args: List[str] = None) -> ModelParameters:
        return None
        # 返回None，表示解析参数方法暂不实现

    def load_worker(self, model_name: str, model_path: str, **kwargs):
        self.host = kwargs.get("host")  # 设置主机地址为关键字参数中的host值
        self.port = kwargs.get("port")  # 设置端口号为关键字参数中的port值

    def start(
        self, model_params: ModelParameters = None, command_args: List[str] = None
    ) -> None:
        """Start model worker"""
        pass
        # 占位符方法，暂不实现，用于启动模型工作器

    def stop(self) -> None:
        raise NotImplementedError("Remote model worker not support stop methods")
        # 抛出未实现错误，表示不支持停止方法

    def generate_stream(self, params: Dict) -> Iterator[ModelOutput]:
        """Generate stream"""
        raise NotImplementedError
        # 抛出未实现错误，表示不支持流式生成方法

    async def async_generate_stream(self, params: Dict) -> Iterator[ModelOutput]:
        """Asynchronous generate stream"""
        import httpx  # 导入httpx模块，用于异步HTTP请求

        async with httpx.AsyncClient() as client:
            delimiter = b"\0"  # 设置分隔符为字节串\0
            buffer = b""  # 初始化缓冲区为空字节串
            url = self.worker_addr + "/generate_stream"  # 构建生成流的完整URL地址
            logger.debug(f"Send async_generate_stream to url {url}, params: {params}")
            # 记录调试信息，包含URL和参数

            async with client.stream(
                "POST",
                url,
                headers=self._get_trace_headers(),  # 获取追踪信息的请求头
                json=params,  # 将参数转换为JSON格式发送
                timeout=self.timeout,  # 设置超时时间
            ) as response:
                async for raw_chunk in response.aiter_raw():
                    buffer += raw_chunk  # 将接收到的原始数据块添加到缓冲区

                    while delimiter in buffer:
                        chunk, buffer = buffer.split(delimiter, 1)  # 按分隔符切割数据
                        if not chunk:
                            continue
                        chunk = chunk.decode()  # 解码数据块为字符串
                        data = json.loads(chunk)  # 解析JSON数据
                        yield ModelOutput(**data)  # 生成ModelOutput对象

    def generate(self, params: Dict) -> ModelOutput:
        """Generate non stream"""
        raise NotImplementedError
        # 抛出未实现错误，表示不支持非流式生成方法
    async def async_generate(self, params: Dict) -> ModelOutput:
        """异步生成非流式输出"""
        import httpx

        async with httpx.AsyncClient() as client:
            url = self.worker_addr + "/generate"
            logger.debug(f"Send async_generate to url {url}, params: {params}")
            # 发送异步 POST 请求到指定 URL，传递参数和超时设置，并获取响应
            response = await client.post(
                url,
                headers=self._get_trace_headers(),  # 获取追踪信息的请求头
                json=params,  # 将参数转换为 JSON 格式发送
                timeout=self.timeout,  # 设置请求超时时间
            )
            # 将 JSON 响应解析为 ModelOutput 对象并返回
            return ModelOutput(**response.json())

    def count_token(self, prompt: str) -> int:
        """计算给定提示文本的标记数"""
        raise NotImplementedError

    async def async_count_token(self, prompt: str) -> int:
        import httpx

        async with httpx.AsyncClient() as client:
            url = self.worker_addr + "/count_token"
            logger.debug(f"Send async_count_token to url {url}, params: {prompt}")
            # 发送异步 POST 请求到指定 URL，传递提示文本参数和超时设置，并获取响应
            response = await client.post(
                url,
                headers=self._get_trace_headers(),  # 获取追踪信息的请求头
                json={"prompt": prompt},  # 将提示文本封装为 JSON 格式发送
                timeout=self.timeout,  # 设置请求超时时间
            )
            # 返回 JSON 响应中的整数结果
            return response.json()

    async def async_get_model_metadata(self, params: Dict) -> ModelMetadata:
        """异步获取模型元数据"""
        import httpx

        async with httpx.AsyncClient() as client:
            url = self.worker_addr + "/model_metadata"
            logger.debug(
                f"Send async_get_model_metadata to url {url}, params: {params}"
            )
            # 发送异步 POST 请求到指定 URL，传递模型参数和超时设置，并获取响应
            response = await client.post(
                url,
                headers=self._get_trace_headers(),  # 获取追踪信息的请求头
                json=params,  # 将模型参数封装为 JSON 格式发送
                timeout=self.timeout,  # 设置请求超时时间
            )
            # 将 JSON 响应解析为 ModelMetadata 对象并返回
            return ModelMetadata.from_dict(response.json())

    def get_model_metadata(self, params: Dict) -> ModelMetadata:
        """获取模型元数据"""
        raise NotImplementedError

    def embeddings(self, params: Dict) -> List[List[float]]:
        """获取输入的嵌入向量"""
        import requests

        url = self.worker_addr + "/embeddings"
        logger.debug(f"Send embeddings to url {url}, params: {params}")
        # 发送 POST 请求到指定 URL，传递参数和超时设置，并获取响应
        response = requests.post(
            url,
            headers=self._get_trace_headers(),  # 获取追踪信息的请求头
            json=params,  # 将参数转换为 JSON 格式发送
            timeout=self.timeout,  # 设置请求超时时间
        )
        # 返回 JSON 响应中的嵌入向量列表
        return response.json()

    async def async_embeddings(self, params: Dict) -> List[List[float]]:
        """异步获取输入的嵌入向量"""
        import httpx

        async with httpx.AsyncClient() as client:
            url = self.worker_addr + "/embeddings"
            logger.debug(f"Send async_embeddings to url {url}")
            # 发送异步 POST 请求到指定 URL，传递参数和超时设置，并获取响应
            response = await client.post(
                url,
                headers=self._get_trace_headers(),  # 获取追踪信息的请求头
                json=params,  # 将参数转换为 JSON 格式发送
                timeout=self.timeout,  # 设置请求超时时间
            )
            # 返回 JSON 响应中的嵌入向量列表
            return response.json()
    # 定义一个方法 `_get_trace_headers`，用于获取追踪相关的头信息
    def _get_trace_headers(self):
        # 调用全局的 root_tracer 对象的方法，获取当前追踪的 span ID
        span_id = root_tracer.get_current_span_id()
        # 复制当前对象的 headers 属性，确保不修改原始 headers
        headers = self.headers.copy()
        # 如果存在有效的 span_id，则将其添加到 headers 中
        if span_id:
            headers.update({DBGPT_TRACER_SPAN_ID: span_id})
        # 返回更新后的 headers
        return headers
```