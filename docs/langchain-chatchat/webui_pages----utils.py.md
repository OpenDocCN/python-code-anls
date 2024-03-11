# `.\Langchain-Chatchat\webui_pages\utils.py`

```
# 该文件封装了对api.py的请求，可以被不同的webui使用
# 通过ApiRequest和AsyncApiRequest支持同步/异步调用

# 导入类型提示相关的模块
from typing import *
# 导入处理路径相关的模块
from pathlib import Path
# 导入配置文件，用于设置默认值
from configs import (
    EMBEDDING_MODEL,
    DEFAULT_VS_TYPE,
    LLM_MODELS,
    TEMPERATURE,
    SCORE_THRESHOLD,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    ZH_TITLE_ENHANCE,
    VECTOR_SEARCH_TOP_K,
    SEARCH_ENGINE_TOP_K,
    HTTPX_DEFAULT_TIMEOUT,
    logger, log_verbose,
)
# 导入httpx模块，用于发起HTTP请求
import httpx
# 导入上下文管理相关的模块
import contextlib
# 导入处理JSON数据相关的模块
import json
# 导入操作系统相关的模块
import os
# 导入处理字节流相关的模块
from io import BytesIO
# 导入设置httpx配置的函数
from server.utils import set_httpx_config, api_address, get_httpx_client

# 导入用于打印输出的模块
from pprint import pprint
# 导入标记为弃用的装饰器
from langchain_core._api import deprecated

# 设置httpx配置
set_httpx_config()

# 定义ApiRequest类
class ApiRequest:
    '''
    api.py调用的封装（同步模式）,简化api调用方式
    '''

    # 初始化方法
    def __init__(
            self,
            base_url: str = api_address(),
            timeout: float = HTTPX_DEFAULT_TIMEOUT,
    ):
        # 设置基础URL和超时时间
        self.base_url = base_url
        self.timeout = timeout
        self._use_async = False
        self._client = None

    # 定义client属性
    @property
    def client(self):
        # 如果客户端为空或已关闭，则重新创建客户端
        if self._client is None or self._client.is_closed:
            self._client = get_httpx_client(base_url=self.base_url,
                                            use_async=self._use_async,
                                            timeout=self.timeout)
        return self._client

    # 定义get方法
    def get(
            self,
            url: str,
            params: Union[Dict, List[Tuple], bytes] = None,
            retry: int = 3,
            stream: bool = False,
            **kwargs: Any,
    # 定义一个方法用于发送 GET 请求，支持重试机制
    def get(
            self,
            url: str,
            params: Dict = None,
            retry: int = 3,
            stream: bool = False,
            **kwargs: Any
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        # 当重试次数大于 0 时，执行请求
        while retry > 0:
            try:
                # 如果需要流式传输数据，则使用 stream 方法发送 GET 请求
                if stream:
                    return self.client.stream("GET", url, params=params, **kwargs)
                else:
                    # 否则使用 get 方法发送 GET 请求
                    return self.client.get(url, params=params, **kwargs)
            except Exception as e:
                # 捕获异常并记录错误信息
                msg = f"error when get {url}: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                # 重试次数减一
                retry -= 1

    # 定义一个方法用于发送 POST 请求，支持重试机制
    def post(
            self,
            url: str,
            data: Dict = None,
            json: Dict = None,
            retry: int = 3,
            stream: bool = False,
            **kwargs: Any
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        # 当重试次数大于 0 时，执行请求
        while retry > 0:
            try:
                # 如果需要流式传输数据，则使用 stream 方法发送 POST 请求
                if stream:
                    return self.client.stream("POST", url, data=data, json=json, **kwargs)
                else:
                    # 否则使用 post 方法发送 POST 请求
                    return self.client.post(url, data=data, json=json, **kwargs)
            except Exception as e:
                # 捕获异常并记录错误信息
                msg = f"error when post {url}: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                # 重试次数减一
                retry -= 1

    # 定义一个方法用于发送 DELETE 请求，支持重试机制
    def delete(
            self,
            url: str,
            data: Dict = None,
            json: Dict = None,
            retry: int = 3,
            stream: bool = False,
            **kwargs: Any
    # 定义一个方法，用于发送 DELETE 请求，并返回响应或响应迭代器
    def delete_request(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json: Any = None,
        stream: bool = False,
        retry: int = 3,
        log_verbose: bool = False,
        **kwargs
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        # 循环执行请求，直到成功或重试次数用尽
        while retry > 0:
            try:
                # 如果需要流式传输数据，则使用 stream 方法发送 DELETE 请求
                if stream:
                    return self.client.stream("DELETE", url, data=data, json=json, **kwargs)
                else:
                    # 否则使用 delete 方法发送 DELETE 请求
                    return self.client.delete(url, data=data, json=json, **kwargs)
            except Exception as e:
                # 捕获异常并记录错误信息
                msg = f"error when delete {url}: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                # 减少重试次数
                retry -= 1

    # 将异步请求的响应转换为生成器
    def _httpx_stream2generator(
            self,
            response: contextlib._GeneratorContextManager,
            as_json: bool = False,
    # 获取响应值
    def _get_response_value(
            self,
            response: httpx.Response,
            as_json: bool = False,
            value_func: Callable = None,
    ):
        '''
        转换同步或异步请求返回的响应
        `as_json`: 返回json
        `value_func`: 用户可以自定义返回值，该函数接受response或json
        '''

        # 将响应转换为 JSON 格式
        def to_json(r):
            try:
                return r.json()
            except Exception as e:
                msg = "API未能返回正确的JSON。" + str(e)
                if log_verbose:
                    logger.error(f'{e.__class__.__name__}: {msg}',
                                 exc_info=e if log_verbose else None)
                return {"code": 500, "msg": msg, "data": None}

        # 如果未指定自定义返回值函数，则使用默认函数
        if value_func is None:
            value_func = (lambda r: r)

        # 异步返回响应
        async def ret_async(response):
            if as_json:
                return value_func(to_json(await response))
            else:
                return value_func(await response)

        # 根据是否使用异步请求返回相应结果
        if self._use_async:
            return ret_async(response)
        else:
            if as_json:
                return value_func(to_json(response))
            else:
                return value_func(response)

    # 服务器信息
    # 获取服务器配置信息
    def get_server_configs(self, **kwargs) -> Dict:
        # 发送 POST 请求获取服务器配置信息
        response = self.post("/server/configs", **kwargs)
        # 返回响应结果中的 JSON 数据
        return self._get_response_value(response, as_json=True)

    # 列出搜索引擎
    def list_search_engines(self, **kwargs) -> List:
        # 发送 POST 请求列出搜索引擎
        response = self.post("/server/list_search_engines", **kwargs)
        # 返回响应结果中的数据字段作为列表
        return self._get_response_value(response, as_json=True, value_func=lambda r: r["data"])

    # 获取提示模板
    def get_prompt_template(
            self,
            type: str = "llm_chat",
            name: str = "default",
            **kwargs,
    ) -> str:
        # 构建请求数据
        data = {
            "type": type,
            "name": name,
        }
        # 发送 POST 请求获取提示模板
        response = self.post("/server/get_prompt_template", json=data, **kwargs)
        # 返回响应结果中的文本数据
        return self._get_response_value(response, value_func=lambda r: r.text)

    # 对话操作
    def chat_chat(
            self,
            query: str,
            conversation_id: str = None,
            history_len: int = -1,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
            **kwargs,
    ):
        '''
        对应api.py/chat/chat接口
        '''
        # 构建请求数据
        data = {
            "query": query,
            "conversation_id": conversation_id,
            "history_len": history_len,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        # 发送 POST 请求进行对话
        response = self.post("/chat/chat", json=data, stream=True, **kwargs)
        # 将响应结果转换为生成器并返回
        return self._httpx_stream2generator(response, as_json=True)

    # 弃用警告
    @deprecated(
        since="0.3.0",
        message="自定义Agent问答将于 Langchain-Chatchat 0.3.x重写, 0.2.x中相关功能将废弃",
        removal="0.3.0")
    # 定义一个方法用于与代理进行对话
    def agent_chat(
            self,
            query: str,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
    ):
        '''
        对应api.py/chat/agent_chat 接口
        '''
        # 构建包含对话信息的数据字典
        data = {
            "query": query,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        # 发送数据到指定的接口，并获取响应
        response = self.post("/chat/agent_chat", json=data, stream=True)
        # 将响应转换为生成器并返回
        return self._httpx_stream2generator(response, as_json=True)

    # 定义一个方法用于与知识库进行对话
    def knowledge_base_chat(
            self,
            query: str,
            knowledge_base_name: str,
            top_k: int = VECTOR_SEARCH_TOP_K,
            score_threshold: float = SCORE_THRESHOLD,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
    ):
        '''
        对应api.py/chat/knowledge_base_chat接口
        '''
        # 构建包含请求数据的字典
        data = {
            "query": query,
            "knowledge_base_name": knowledge_base_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        # 发送 POST 请求到指定的接口，并传递数据
        response = self.post(
            "/chat/knowledge_base_chat",
            json=data,
            stream=True,
        )
        # 将响应流转换为生成器，并返回
        return self._httpx_stream2generator(response, as_json=True)

    # 上传临时文档
    def upload_temp_docs(
            self,
            files: List[Union[str, Path, bytes]],
            knowledge_id: str = None,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE,
            zh_title_enhance=ZH_TITLE_ENHANCE,
        '''
        对应api.py/knowledge_base/upload_tmep_docs接口
        '''

        # 定义一个函数，用于将不同类型的文件转换为统一格式
        def convert_file(file, filename=None):
            # 如果文件是字节流，则直接使用
            if isinstance(file, bytes):  # raw bytes
                file = BytesIO(file)
            # 如果文件具有read方法，则将其视为文件对象
            elif hasattr(file, "read"):  # a file io like object
                filename = filename or file.name
            # 如果文件是本地路径，则打开并读取文件内容
            else:  # a local path
                file = Path(file).absolute().open("rb")
                filename = filename or os.path.split(file.name)[-1]
            return filename, file

        # 对输入的多个文件进行转换
        files = [convert_file(file) for file in files]
        # 构建请求数据
        data = {
            "knowledge_id": knowledge_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
        }

        # 发送POST请求，上传临时文档
        response = self.post(
            "/knowledge_base/upload_temp_docs",
            data=data,
            files=[("files", (filename, file)) for filename, file in files],
        )
        # 获取响应结果
        return self._get_response_value(response, as_json=True)

    # 定义一个函数，用于进行文件聊天
    def file_chat(
            self,
            query: str,
            knowledge_id: str,
            top_k: int = VECTOR_SEARCH_TOP_K,
            score_threshold: float = SCORE_THRESHOLD,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
    ):
        '''
        对应api.py/chat/file_chat接口
        '''
        # 构建包含请求参数的字典
        data = {
            "query": query,
            "knowledge_id": knowledge_id,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        # 发起 POST 请求到指定的接口
        response = self.post(
            "/chat/file_chat",
            json=data,
            stream=True,
        )
        # 将响应转换为生成器并返回
        return self._httpx_stream2generator(response, as_json=True)

    @deprecated(
        since="0.3.0",
        message="搜索引擎问答将于 Langchain-Chatchat 0.3.x重写, 0.2.x中相关功能将废弃",
        removal="0.3.0"
    )
    def search_engine_chat(
            self,
            query: str,
            search_engine_name: str,
            top_k: int = SEARCH_ENGINE_TOP_K,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
            split_result: bool = False,
    ):
        '''
        对应api.py/chat/search_engine_chat接口
        '''
        # 构建包含请求参数的字典
        data = {
            "query": query,
            "search_engine_name": search_engine_name,
            "top_k": top_k,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
            "split_result": split_result,
        }

        # 发起 POST 请求到指定的接口
        response = self.post(
            "/chat/search_engine_chat",
            json=data,
            stream=True,
        )
        # 将响应转换为生成器并返回
        return self._httpx_stream2generator(response, as_json=True)

    # 知识库相关操作
    # 列出所有知识库的名称
    def list_knowledge_bases(
            self,
    ):
        '''
        对应api.py/knowledge_base/list_knowledge_bases接口
        '''
        # 发起GET请求获取知识库列表
        response = self.get("/knowledge_base/list_knowledge_bases")
        # 解析响应数据，返回数据部分作为JSON格式
        return self._get_response_value(response,
                                        as_json=True,
                                        value_func=lambda r: r.get("data", []))

    # 创建知识库
    def create_knowledge_base(
            self,
            knowledge_base_name: str,
            vector_store_type: str = DEFAULT_VS_TYPE,
            embed_model: str = EMBEDDING_MODEL,
    ):
        '''
        对应api.py/knowledge_base/create_knowledge_base接口
        '''
        # 构建请求数据
        data = {
            "knowledge_base_name": knowledge_base_name,
            "vector_store_type": vector_store_type,
            "embed_model": embed_model,
        }

        # 发起POST请求创建知识库
        response = self.post(
            "/knowledge_base/create_knowledge_base",
            json=data,
        )
        # 解析响应数据，返回JSON格式数据
        return self._get_response_value(response, as_json=True)

    # 删除知识库
    def delete_knowledge_base(
            self,
            knowledge_base_name: str,
    ):
        '''
        对应api.py/knowledge_base/delete_knowledge_base接口
        '''
        # 发起POST请求删除指定名称的知识库
        response = self.post(
            "/knowledge_base/delete_knowledge_base",
            json=f"{knowledge_base_name}",
        )
        # 解析响应数据，返回JSON格式数据
        return self._get_response_value(response, as_json=True)

    # 列出知识库文档
    def list_kb_docs(
            self,
            knowledge_base_name: str,
    ):
        '''
        对应api.py/knowledge_base/list_files接口
        '''
        # 发起GET请求获取指定知识库的文档列表
        response = self.get(
            "/knowledge_base/list_files",
            params={"knowledge_base_name": knowledge_base_name}
        )
        # 解析响应数据，返回数据部分作为JSON格式
        return self._get_response_value(response,
                                        as_json=True,
                                        value_func=lambda r: r.get("data", []))
    # 定义一个方法用于搜索知识库文档
    def search_kb_docs(
            self,
            knowledge_base_name: str,
            query: str = "",
            top_k: int = VECTOR_SEARCH_TOP_K,
            score_threshold: int = SCORE_THRESHOLD,
            file_name: str = "",
            metadata: dict = {},
    ) -> List:
        '''
        对应api.py/knowledge_base/search_docs接口
        '''
        # 构建请求数据
        data = {
            "query": query,
            "knowledge_base_name": knowledge_base_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "file_name": file_name,
            "metadata": metadata,
        }

        # 发送POST请求
        response = self.post(
            "/knowledge_base/search_docs",
            json=data,
        )
        # 获取并返回响应值
        return self._get_response_value(response, as_json=True)

    # 定义一个方法用于根据文档ID更新文档
    def update_docs_by_id(
            self,
            knowledge_base_name: str,
            docs: Dict[str, Dict],
    ) -> bool:
        '''
        对应api.py/knowledge_base/update_docs_by_id接口
        '''
        # 构建请求数据
        data = {
            "knowledge_base_name": knowledge_base_name,
            "docs": docs,
        }
        # 发送POST请求
        response = self.post(
            "/knowledge_base/update_docs_by_id",
            json=data
        )
        # 获取并返回响应值
        return self._get_response_value(response)

    # 定义一个方法用于上传知识库文档
    def upload_kb_docs(
            self,
            files: List[Union[str, Path, bytes]],
            knowledge_base_name: str,
            override: bool = False,
            to_vector_store: bool = True,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE,
            zh_title_enhance=ZH_TITLE_ENHANCE,
            docs: Dict = {},
            not_refresh_vs_cache: bool = False,
        '''
        对应api.py/knowledge_base/upload_docs接口
        '''

        # 定义一个函数，用于将不同类型的文件转换为统一格式
        def convert_file(file, filename=None):
            # 如果文件是字节流，则直接使用
            if isinstance(file, bytes):  # raw bytes
                file = BytesIO(file)
            # 如果文件具有read方法，则将其转换为文件对象
            elif hasattr(file, "read"):  # a file io like object
                filename = filename or file.name
            # 如果文件是本地路径，则打开并读取文件内容
            else:  # a local path
                file = Path(file).absolute().open("rb")
                filename = filename or os.path.split(file.name)[-1]
            return filename, file

        # 对输入的文件列表进行转换
        files = [convert_file(file) for file in files]
        
        # 构建请求数据
        data = {
            "knowledge_base_name": knowledge_base_name,
            "override": override,
            "to_vector_store": to_vector_store,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
            "docs": docs,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        # 如果docs是字典类型，则转换为JSON字符串
        if isinstance(data["docs"], dict):
            data["docs"] = json.dumps(data["docs"], ensure_ascii=False)
        
        # 发送POST请求，上传文件和数据
        response = self.post(
            "/knowledge_base/upload_docs",
            data=data,
            files=[("files", (filename, file)) for filename, file in files],
        )
        
        # 返回响应结果
        return self._get_response_value(response, as_json=True)

    # 删除知识库中的文档
    def delete_kb_docs(
            self,
            knowledge_base_name: str,
            file_names: List[str],
            delete_content: bool = False,
            not_refresh_vs_cache: bool = False,
    ):
        '''
        对应api.py/knowledge_base/delete_docs接口
        '''
        # 构建包含删除文档所需参数的数据字典
        data = {
            "knowledge_base_name": knowledge_base_name,
            "file_names": file_names,
            "delete_content": delete_content,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        # 发送 POST 请求到指定的接口，传入数据字典
        response = self.post(
            "/knowledge_base/delete_docs",
            json=data,
        )
        # 返回响应结果的 JSON 格式
        return self._get_response_value(response, as_json=True)

    def update_kb_info(self, knowledge_base_name, kb_info):
        '''
        对应api.py/knowledge_base/update_info接口
        '''
        # 构建包含更新知识库信息所需参数的数据字典
        data = {
            "knowledge_base_name": knowledge_base_name,
            "kb_info": kb_info,
        }

        # 发送 POST 请求到指定的接口，传入数据字典
        response = self.post(
            "/knowledge_base/update_info",
            json=data,
        )
        # 返回响应结果的 JSON 格式
        return self._get_response_value(response, as_json=True)

    def update_kb_docs(
            self,
            knowledge_base_name: str,
            file_names: List[str],
            override_custom_docs: bool = False,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE,
            zh_title_enhance=ZH_TITLE_ENHANCE,
            docs: Dict = {},
            not_refresh_vs_cache: bool = False,
    ):
        '''
        对应api.py/knowledge_base/update_docs接口
        '''
        # 构建包含更新文档所需参数的字典
        data = {
            "knowledge_base_name": knowledge_base_name,
            "file_names": file_names,
            "override_custom_docs": override_custom_docs,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
            "docs": docs,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        # 如果文档是字典类型，则转换为 JSON 字符串
        if isinstance(data["docs"], dict):
            data["docs"] = json.dumps(data["docs"], ensure_ascii=False)

        # 发送 POST 请求更新文档
        response = self.post(
            "/knowledge_base/update_docs",
            json=data,
        )
        # 返回响应值
        return self._get_response_value(response, as_json=True)

    def recreate_vector_store(
            self,
            knowledge_base_name: str,
            allow_empty_kb: bool = True,
            vs_type: str = DEFAULT_VS_TYPE,
            embed_model: str = EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE,
            zh_title_enhance=ZH_TITLE_ENHANCE,
    ):
        '''
        对应api.py/knowledge_base/recreate_vector_store接口
        '''
        # 构建包含重新创建向量存储所需参数的字典
        data = {
            "knowledge_base_name": knowledge_base_name,
            "allow_empty_kb": allow_empty_kb,
            "vs_type": vs_type,
            "embed_model": embed_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
        }

        # 发送 POST 请求重新创建向量存储
        response = self.post(
            "/knowledge_base/recreate_vector_store",
            json=data,
            stream=True,
            timeout=None,
        )
        # 将 HTTPX 流转换为生成器并返回
        return self._httpx_stream2generator(response, as_json=True)

    # LLM模型相关操作
    def list_running_models(
            self,
            controller_address: str = None,
        ):
        '''
        获取Fastchat中正运行的模型列表
        '''
        # 构建包含控制器地址的数据字典
        data = {
            "controller_address": controller_address,
        }

        # 如果日志详细信息开启，则记录数据信息
        if log_verbose:
            logger.info(f'{self.__class__.__name__}:data: {data}')

        # 发送 POST 请求获取正在运行的模型列表
        response = self.post(
            "/llm_model/list_running_models",
            json=data,
        )
        # 返回响应中"data"字段的值，作为 JSON 格式，如果没有则返回空列表
        return self._get_response_value(response, as_json=True, value_func=lambda r: r.get("data", []))
    # 定义一个方法，用于获取默认的LLM模型
    def get_default_llm_model(self, local_first: bool = True) -> Tuple[str, bool]:
        '''
        从服务器上获取当前运行的LLM模型。
        当 local_first=True 时，优先返回运行中的本地模型，否则优先按LLM_MODELS配置顺序返回。
        返回类型为（model_name, is_local_model）
        '''

        # 定义一个同步方法，用于返回LLM模型
        def ret_sync():
            # 获取当前运行的模型列表
            running_models = self.list_running_models()
            if not running_models:
                return "", False

            model = ""
            # 遍历LLM_MODELS配置的模型
            for m in LLM_MODELS:
                if m not in running_models:
                    continue
                is_local = not running_models[m].get("online_api")
                if local_first and not is_local:
                    continue
                else:
                    model = m
                    break

            if not model:  # LLM_MODELS中配置的模型都不在running_models里
                model = list(running_models)[0]
            is_local = not running_models[model].get("online_api")
            return model, is_local

        # 定义一个异步方法，用于返回LLM模型
        async def ret_async():
            # 异步获取当前运行的模型列表
            running_models = await self.list_running_models()
            if not running_models:
                return "", False

            model = ""
            # 遍历LLM_MODELS配置的模型
            for m in LLM_MODELS:
                if m not in running_models:
                    continue
                is_local = not running_models[m].get("online_api")
                if local_first and not is_local:
                    continue
                else:
                    model = m
                    break

            if not model:  # LLM_MODELS中配置的模型都不在running_models里
                model = list(running_models)[0]
            is_local = not running_models[model].get("online_api")
            return model, is_local

        # 根据是否使用异步返回相应的LLM模型
        if self._use_async:
            return ret_async()
        else:
            return ret_sync()

    # 定义一个方法，用于列出配置的模型
    def list_config_models(
            self,
            types: List[str] = ["local", "online"],
    def get_model_list(self, types: List[str]) -> Dict[str, Dict]:
        '''
        获取服务器configs中配置的模型列表，返回形式为{"type": {model_name: config}, ...}。
        '''
        # 构建请求数据，包含模型类型列表
        data = {
            "types": types,
        }
        # 发送POST请求到指定接口，获取响应
        response = self.post(
            "/llm_model/list_config_models",
            json=data,
        )
        # 解析响应数据，返回模型列表信息
        return self._get_response_value(response, as_json=True, value_func=lambda r: r.get("data", {}))

    def get_model_config(
            self,
            model_name: str = None,
    ) -> Dict:
        '''
        获取服务器上模型配置
        '''
        # 构建请求数据，包含模型名称
        data = {
            "model_name": model_name,
        }
        # 发送POST请求到指定接口，获取响应
        response = self.post(
            "/llm_model/get_model_config",
            json=data,
        )
        # 解析响应数据，返回模型配置信息
        return self._get_response_value(response, as_json=True, value_func=lambda r: r.get("data", {}))

    def list_search_engines(self) -> List[str]:
        '''
        获取服务器支持的搜索引擎
        '''
        # 发送POST请求到指定接口，获取响应
        response = self.post(
            "/server/list_search_engines",
        )
        # 解析响应数据，返回搜索引擎列表
        return self._get_response_value(response, as_json=True, value_func=lambda r: r.get("data", {}))

    def stop_llm_model(
            self,
            model_name: str,
            controller_address: str = None,
    ):
        '''
        停止某个LLM模型。
        注意：由于Fastchat的实现方式，实际上是把LLM模型所在的model_worker停掉。
        '''
        # 构建请求数据，包含模型名称和控制器地址
        data = {
            "model_name": model_name,
            "controller_address": controller_address,
        }
        # 发送POST请求到指定接口，停止指定LLM模型
        response = self.post(
            "/llm_model/stop",
            json=data,
        )
        # 解析响应数据，返回停止操作结果
        return self._get_response_value(response, as_json=True)

    def change_llm_model(
            self,
            model_name: str,
            new_model_name: str,
            controller_address: str = None,
    ):
        '''
        更改LLM模型的名称。
        '''
        # 构建请求数据，包含原模型名称、新模型名称和控制器地址
        data = {
            "model_name": model_name,
            "new_model_name": new_model_name,
            "controller_address": controller_address,
        }
        # 发送POST请求到指定接口，更改LLM模型名称
        response = self.post(
            "/llm_model/change_name",
            json=data,
        )
        # 解析响应数据，返回更改操作结果
        return self._get_response_value(response, as_json=True)

    def embed_texts(
            self,
            texts: List[str],
            embed_model: str = EMBEDDING_MODEL,
            to_query: bool = False,
        ):
        '''
        将文本嵌入到向量空间中。
        '''
        # 构建请求数据，包含文本列表、嵌入模型和是否转换为查询向量的标志
        data = {
            "texts": texts,
            "embed_model": embed_model,
            "to_query": to_query,
        }
        # 发送POST请求到指定接口，将文本嵌入到向量空间中
        response = self.post(
            "/llm_model/embed_texts",
            json=data,
        )
        # 解析响应数据，返回嵌入结果
        return self._get_response_value(response, as_json=True)
    ) -> List[List[float]]:
        '''
        对文本进行向量化，可选模型包括本地 embed_models 和支持 embeddings 的在线模型
        '''
        # 准备要发送的数据，包括文本内容、嵌入模型和查询方式
        data = {
            "texts": texts,
            "embed_model": embed_model,
            "to_query": to_query,
        }
        # 发送 POST 请求到指定路径，传递数据
        resp = self.post(
            "/other/embed_texts",
            json=data,
        )
        # 从响应中获取数据并以 JSON 格式返回
        return self._get_response_value(resp, as_json=True, value_func=lambda r: r.get("data"))

    def chat_feedback(
            self,
            message_id: str,
            score: int,
            reason: str = "",
    ) -> int:
        '''
        反馈对话评价
        '''
        # 准备要发送的数据，包括消息 ID、评分和反馈原因
        data = {
            "message_id": message_id,
            "score": score,
            "reason": reason,
        }
        # 发送 POST 请求到指定路径，传递数据
        resp = self.post("/chat/feedback", json=data)
        # 从响应中获取数据并返回
        return self._get_response_value(resp)
class AsyncApiRequest(ApiRequest):
    # 异步 API 请求类，继承自 ApiRequest 类
    def __init__(self, base_url: str = api_address(), timeout: float = HTTPX_DEFAULT_TIMEOUT):
        # 初始化方法，设置基础 URL 和超时时间
        super().__init__(base_url, timeout)
        # 设置使用异步请求标志为 True
        self._use_async = True


def check_error_msg(data: Union[str, dict, list], key: str = "errorMsg") -> str:
    '''
    return error message if error occured when requests API
    '''
    # 检查数据是否为字典类型
    if isinstance(data, dict):
        # 检查指定键是否在数据中
        if key in data:
            return data[key]
        # 检查数据中是否包含"code"键且其值不为200
        if "code" in data and data["code"] != 200:
            return data["msg"]
    return ""


def check_success_msg(data: Union[str, dict, list], key: str = "msg") -> str:
    '''
    return error message if error occured when requests API
    '''
    # 检查数据是否为字典类型，并且包含指定键和"code"键值为200
    if (isinstance(data, dict)
            and key in data
            and "code" in data
            and data["code"] == 200):
        return data[key]
    return ""


if __name__ == "__main__":
    # 创建 ApiRequest 实例
    api = ApiRequest()
    # 创建 AsyncApiRequest 实例
    aapi = AsyncApiRequest()

    # 注释掉以下代码块
    # with api.chat_chat("你好") as r:
    #     for t in r.iter_text(None):
    #         print(t)

    # 注释掉以下代码块
    # r = api.chat_chat("你好", no_remote_api=True)
    # for t in r:
    #     print(t)

    # 注释掉以下代码块
    # r = api.duckduckgo_search_chat("室温超导最新研究进展", no_remote_api=True)
    # for t in r:
    #     print(t)

    # 注释掉以下代码块
    # print(api.list_knowledge_bases())
```